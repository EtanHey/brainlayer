// w300 recorder: ScreenCaptureKit at the display's 120 Hz, one screen region, cursor drawn, H.264 into .mov.
// Every delivered frame's presentation time (host clock seconds = CACurrentMediaTime) goes to <out>.pts, so the
// recorder's own cadence is measured, not assumed. minimumFrameInterval is 1/240 s on purpose: at exactly 1/120 s SCK
// beat against the display's vsync and skipped ~1 frame in 10 (measured against the reference strip); at 1/240 s it
// delivers every 120 Hz frame (SCK_RATE overrides). SCK delivers a frame only when the region changed (idle frames
// carry no image), so cadence is read inside motion windows.
//   sck <out.mov> <seconds> <x> <y> <w> <h> [scale=2] [bgra|420v]     (points, global, top-left origin)
import AVFoundation
import CoreMedia
import ScreenCaptureKit

let a = CommandLine.arguments
let out = URL(fileURLWithPath: a[1])
let seconds = Double(a[2])!
let region = CGRect(x: Double(a[3])!, y: Double(a[4])!, width: Double(a[5])!, height: Double(a[6])!)
let scale = a.count > 7 ? Double(a[7])! : 2.0
let yuv = a.count > 8 && a[8] == "420v"

final class Rec: NSObject, SCStreamOutput, @unchecked Sendable {
    let writer: AVAssetWriter
    let input: AVAssetWriterInput
    var started = false
    var pts: [Double] = []
    let q = DispatchQueue(label: "rec")
    init(url: URL, w: Int, h: Int) throws {
        try? FileManager.default.removeItem(at: url)
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: w, AVVideoHeightKey: h,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 80_000_000, AVVideoExpectedSourceFrameRateKey: 120,
                                              AVVideoMaxKeyFrameIntervalKey: 120],
        ])
        input.expectsMediaDataInRealTime = true
        writer.add(input)
    }
    func stream(_ stream: SCStream, didOutputSampleBuffer sb: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .screen, sb.isValid,
              let att = (CMSampleBufferGetSampleAttachmentsArray(sb, createIfNecessary: false) as? [[SCStreamFrameInfo: Any]])?.first,
              let raw = att[.status] as? Int, SCFrameStatus(rawValue: raw) == .complete else { return }
        let t = CMSampleBufferGetPresentationTimeStamp(sb)
        if !started { writer.startWriting(); writer.startSession(atSourceTime: t); started = true }
        if input.isReadyForMoreMediaData { input.append(sb); pts.append(t.seconds) } else { pts.append(-t.seconds) }
    }
}

let sem = DispatchSemaphore(value: 0)
Task {
    do {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: true)
        let display = content.displays.first { $0.displayID == CGMainDisplayID() }!
        let filter = SCContentFilter(display: display, excludingWindows: [])
        let cfg = SCStreamConfiguration()
        cfg.sourceRect = region
        cfg.width = Int(region.width * scale); cfg.height = Int(region.height * scale)
        cfg.minimumFrameInterval = CMTime(value: 1, timescale: Int32(ProcessInfo.processInfo.environment["SCK_RATE"] ?? "240")!)
        cfg.showsCursor = true
        cfg.queueDepth = 8
        cfg.pixelFormat = yuv ? kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange : kCVPixelFormatType_32BGRA
        let rec = try Rec(url: out, w: cfg.width, h: cfg.height)
        let stream = SCStream(filter: filter, configuration: cfg, delegate: nil)
        try stream.addStreamOutput(rec, type: .screen, sampleHandlerQueue: rec.q)
        try await stream.startCapture()
        FileHandle.standardError.write("recording started \(CACurrentMediaTime())\n".data(using: .utf8)!)
        try await Task.sleep(nanoseconds: UInt64(seconds * 1e9))
        try await stream.stopCapture()
        rec.q.sync {}
        rec.input.markAsFinished()
        await rec.writer.finishWriting()
        let txt = rec.pts.map { String(format: "%.6f", $0) }.joined(separator: "\n") + "\n"
        try txt.write(to: out.deletingPathExtension().appendingPathExtension("pts"), atomically: true, encoding: .utf8)
        FileHandle.standardError.write("frames \(rec.pts.count) dropped-by-writer \(rec.pts.filter { $0 < 0 }.count) status \(rec.writer.status.rawValue)\n".data(using: .utf8)!)
    } catch { FileHandle.standardError.write("error \(error)\n".data(using: .utf8)!) }
    sem.signal()
}
sem.wait()

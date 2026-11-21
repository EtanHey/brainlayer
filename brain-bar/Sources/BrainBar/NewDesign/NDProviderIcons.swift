import AppKit
import SwiftUI

// Provider icon paths from T3 Code (https://github.com/pingdotgg/t3code), used under its MIT License:
//
// MIT License
//
// Copyright (c) 2026 T3 Tools Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// Sources (in the T3 Code repository): one per icon below. Drawn monochrome in the label colour (L34).

/// R1.4 (L34): the four providers BrainLayer reads, as vector shapes. BrainLayer's key `gemini` is Antigravity: its watch
/// root is Antigravity's own folder.
enum NDProvider {
    struct Icon: Sendable {
        let viewBox: CGRect
        let path: String
        let evenOdd: Bool
    }

    static func name(_ family: AgentFamily) -> String {
        switch family {
        case .gemini: BrainBarMenuPresenter.SurfaceSay.antigravity
        default: family.label
        }
    }

    static func icon(_ family: AgentFamily) -> Icon {
        switch family {
        case .claude: claude
        case .codex: codex
        case .cursor: cursor
        case .gemini: antigravity
        }
    }

    /// T3 Code: apps/marketing/public/harnesses/claude-ai-icon.svg.
    static let claude = Icon(viewBox: CGRect(x: 0, y: 0, width: 256, height: 257), path: "m 50.228 170.321 50.357 -28.257 .843 -2.463 -.843 -1.361 h -2.462 l -8.426 -.518 -28.775 -.778 -24.952 -1.037 -24.175 -1.296 -6.092 -1.297 L 0 125.796 l .583 -3.759 5.12 -3.434 7.324 .648 16.202 1.101 24.304 1.685 17.629 1.037 26.118 2.722 h 4.148 l .583 -1.685 -1.426 -1.037 -1.101 -1.037 -25.147 -17.045 -27.22 -18.017 -14.258 -10.37 -7.713 -5.25 -3.888 -4.925 -1.685 -10.758 7 -7.713 9.397 .649 2.398 .648 9.527 7.323 20.35 15.75 L 94.817 91.9 l 3.889 3.24 1.555 -1.102 .195 -.777 -1.75 -2.917 -14.453 -26.118 -15.425 -26.572 -6.87 -11.018 -1.814 -6.61 c -.648 -2.723 -1.102 -4.991 -1.102 -7.778 l 7.972 -10.823 L 71.42 0 82.05 1.426 l 4.472 3.888 6.61 15.101 10.694 23.786 16.591 32.34 4.861 9.592 2.592 8.879 .973 2.722 h 1.685 v -1.556 l 1.36 -18.211 2.528 -22.36 2.463 -28.776 .843 -8.1 4.018 -9.722 7.971 -5.25 6.222 2.981 5.12 7.324 -.713 4.73 -3.046 19.768 -5.962 30.98 -3.889 20.739 h 2.268 l 2.593 -2.593 10.499 -13.934 17.628 -22.036 7.778 -8.749 9.073 -9.657 5.833 -4.601 h 11.018 l 8.1 12.055 -3.628 12.443 -11.342 14.388 -9.398 12.184 -13.48 18.147 -8.426 14.518 .778 1.166 2.01 -.194 30.46 -6.481 16.462 -2.982 19.637 -3.37 8.88 4.148 .971 4.213 -3.5 8.62 -20.998 5.184 -24.628 4.926 -36.682 8.685 -.454 .324 .519 .648 16.526 1.555 7.065 .389 h 17.304 l 32.21 2.398 8.426 5.574 5.055 6.805 -.843 5.184 -12.962 6.611 -17.498 -4.148 -40.83 -9.721 -14 -3.5 h -1.944 v 1.167 l 11.666 11.406 21.387 19.314 26.767 24.887 1.36 6.157 -3.434 4.86 -3.63 -.518 -23.526 -17.693 -9.073 -7.972 -20.545 -17.304 h -1.36 v 1.814 l 4.73 6.935 25.017 37.59 1.296 11.536 -1.814 3.76 -6.481 2.268 -7.13 -1.297 -14.647 -20.544 -15.1 -23.138 -12.185 -20.739 -1.49 .843 -7.194 77.448 -3.37 3.953 -7.778 2.981 -6.48 -4.925 -3.436 -7.972 3.435 -15.749 4.148 -20.544 3.37 -16.333 3.046 -20.285 1.815 -6.74 -.13 -.454 -1.49 .194 -15.295 20.999 -23.267 31.433 -18.406 19.702 -4.407 1.75 -7.648 -3.954 .713 -7.064 4.277 -6.286 25.47 -32.405 15.36 -20.092 9.917 -11.6 -.065 -1.686 h -.583 L 44.07 198.125 l -12.055 1.555 -5.185 -4.86 .648 -7.972 2.463 -2.593 20.35 -13.999 -.064 .065 Z", evenOdd: false)
    /// T3 Code: apps/marketing/public/harnesses/openai_dark.svg.
    static let codex = Icon(viewBox: CGRect(x: 100, y: 100, width: 411, height: 411), path: "M 252.794 108.802 C 289.191 99.0484 326.265 110.305 351.148 135.135 C 385.113 126.072 422.85 134.862 449.492 161.505 C 476.136 188.149 484.925 225.888 475.862 259.85 V 259.854 C 500.696 284.735 511.95 321.81 502.198 358.207 C 492.447 394.602 464.161 421.084 430.215 430.217 C 421.083 464.162 394.603 492.448 358.206 502.199 C 321.812 511.951 284.734 500.693 259.852 475.864 C 225.887 484.927 188.15 476.137 161.507 449.495 C 134.864 422.851 126.073 385.111 135.136 351.149 C 110.304 326.266 99.0496 289.192 108.801 252.795 C 118.552 216.4 146.84 189.918 180.784 180.785 C 189.917 146.841 216.396 118.553 252.794 108.802 Z M 374.292 407.145 C 374.292 411.271 372.092 415.086 368.517 417.148 L 283.723 466.102 C 302.487 480.585 327.555 486.459 352.217 479.852 C 386.997 470.532 410.068 439.312 410.555 405.006 V 317.717 C 410.555 315.08 409.125 312.621 406.843 311.303 L 374.292 292.509 V 407.145 Z M 251.868 415.897 C 248.296 417.959 243.893 417.959 240.317 415.897 L 155.526 366.942 C 152.366 390.436 159.811 415.08 177.866 433.136 H 177.863 C 203.325 458.594 241.896 462.962 271.85 446.232 L 347.449 402.586 C 349.735 401.268 351.148 398.8 351.148 396.163 V 358.579 L 251.868 415.897 Z M 368.602 220.628 C 366.319 219.309 363.474 219.318 361.191 220.637 L 328.641 239.431 L 427.921 296.749 C 431.496 298.811 433.697 302.627 433.697 306.752 V 404.661 C 455.622 395.654 473.244 376.881 479.851 352.218 C 489.169 317.442 473.668 281.85 444.201 264.274 L 368.602 220.628 Z M 177.303 206.34 C 155.377 215.348 137.756 234.122 131.148 258.783 C 121.832 293.561 137.331 329.153 166.799 346.727 L 242.398 390.373 C 244.68 391.692 247.525 391.684 249.807 390.366 L 282.357 371.572 L 183.078 314.253 C 179.504 312.189 177.303 308.375 177.303 304.251 V 206.34 Z M 259.849 279.145 V 331.858 L 305.5 358.213 L 351.15 331.858 V 279.145 L 305.5 252.789 L 259.849 279.145 Z M 327.276 144.9 C 308.512 130.418 283.445 124.543 258.782 131.15 C 224.002 140.471 200.931 171.691 200.445 205.995 V 293.286 C 200.445 295.923 201.875 298.381 204.158 299.7 L 236.707 318.493 V 203.856 C 236.707 199.731 238.909 195.916 242.483 193.853 L 327.276 144.9 Z M 433.137 177.867 C 407.675 152.407 369.103 148.038 339.149 164.769 L 263.55 208.415 C 261.265 209.734 259.852 212.202 259.852 214.838 V 252.423 L 359.132 195.105 C 362.703 193.041 367.108 193.041 370.682 195.105 L 455.473 244.06 C 458.635 220.567 451.189 195.922 433.135 177.867 H 433.137 Z", evenOdd: true)
    /// T3 Code: apps/marketing/public/harnesses/cursor_light.svg.
    static let cursor = Icon(viewBox: CGRect(x: 0, y: 0, width: 466.73, height: 532.09), path: "M 457.43 125.94 L 244.42 2.96 c -6.84 -3.95 -15.28 -3.95 -22.12 0 L 9.3 125.94 c -5.75 3.32 -9.3 9.46 -9.3 16.11 v 247.99 c 0 6.65 3.55 12.79 9.3 16.11 l 213.01 122.98 c 6.84 3.95 15.28 3.95 22.12 0 l 213.01 -122.98 c 5.75 -3.32 9.3 -9.46 9.3 -16.11 v -247.99 c 0 -6.65 -3.55 -12.79 -9.3 -16.11 h -.01 Z M 444.05 151.99 l -205.63 356.16 c -1.39 2.4 -5.06 1.42 -5.06 -1.36 v -233.21 c 0 -4.66 -2.49 -8.97 -6.53 -11.31 L 24.87 145.67 c -2.4 -1.39 -1.42 -5.06 1.36 -5.06 h 411.26 c 5.84 0 9.49 6.33 6.57 11.39 h -.01 Z", evenOdd: false)
    /// T3 Code: apps/marketing/public/harnesses/antigravity.svg (the silhouette path of its mask).
    static let antigravity = Icon(viewBox: CGRect(x: 0, y: 0, width: 24, height: 24), path: "M 21.751 22.607 c 1.34 1.005 3.35 .335 1.508 -1.508 C 17.73 15.74 18.904 1 12.037 1 5.17 1 6.342 15.74 .815 21.1 c -2.01 2.009 .167 2.511 1.507 1.506 5.192 -3.517 4.857 -9.714 9.715 -9.714 4.857 0 4.522 6.197 9.714 9.715 z", evenOdd: false)
}

/// A minimal SVG path reader for the icon data above: M L H V C Z, absolute and relative (the commands those paths use),
/// plus S and Q for safety. Scaled into `rect`, keeping the aspect ratio, centred.
enum NDSVGPath {
    static func path(_ d: String, viewBox: CGRect, in rect: CGRect) -> Path {
        let scale = min(rect.width / viewBox.width, rect.height / viewBox.height)
        let ox = rect.midX - viewBox.midX * scale, oy = rect.midY - viewBox.midY * scale
        func map(_ p: CGPoint) -> CGPoint { CGPoint(x: ox + p.x * scale, y: oy + p.y * scale) }
        var out = Path()
        var tokens = tokenize(d)[...]
        var cmd: Character = "M"
        var cur = CGPoint.zero, start = CGPoint.zero, lastControl: CGPoint?
        func num() -> CGFloat? {
            guard case let .number(n)? = tokens.first else { return nil }
            tokens = tokens.dropFirst()
            return n
        }
        while let token = tokens.first {
            if case let .command(c) = token { cmd = c; tokens = tokens.dropFirst() }
            let rel = cmd.isLowercase
            let base = rel ? cur : .zero
            switch Character(cmd.uppercased()) {
            case "M":
                guard let x = num(), let y = num() else { return out }
                cur = CGPoint(x: base.x + x, y: base.y + y); start = cur
                out.move(to: map(cur))
                cmd = rel ? "l" : "L"
                lastControl = nil
            case "L":
                guard let x = num(), let y = num() else { return out }
                cur = CGPoint(x: base.x + x, y: base.y + y)
                out.addLine(to: map(cur)); lastControl = nil
            case "H":
                guard let x = num() else { return out }
                cur = CGPoint(x: (rel ? cur.x : 0) + x, y: cur.y)
                out.addLine(to: map(cur)); lastControl = nil
            case "V":
                guard let y = num() else { return out }
                cur = CGPoint(x: cur.x, y: (rel ? cur.y : 0) + y)
                out.addLine(to: map(cur)); lastControl = nil
            case "C":
                guard let x1 = num(), let y1 = num(), let x2 = num(), let y2 = num(), let x = num(), let y = num() else { return out }
                let c1 = CGPoint(x: base.x + x1, y: base.y + y1), c2 = CGPoint(x: base.x + x2, y: base.y + y2)
                cur = CGPoint(x: base.x + x, y: base.y + y)
                out.addCurve(to: map(cur), control1: map(c1), control2: map(c2)); lastControl = c2
            case "S":
                guard let x2 = num(), let y2 = num(), let x = num(), let y = num() else { return out }
                let c1 = lastControl.map { CGPoint(x: 2 * cur.x - $0.x, y: 2 * cur.y - $0.y) } ?? cur
                let c2 = CGPoint(x: base.x + x2, y: base.y + y2)
                cur = CGPoint(x: base.x + x, y: base.y + y)
                out.addCurve(to: map(cur), control1: map(c1), control2: map(c2)); lastControl = c2
            case "Q":
                guard let x1 = num(), let y1 = num(), let x = num(), let y = num() else { return out }
                let c = CGPoint(x: base.x + x1, y: base.y + y1)
                cur = CGPoint(x: base.x + x, y: base.y + y)
                out.addQuadCurve(to: map(cur), control: map(c)); lastControl = nil
            case "Z":
                out.closeSubpath(); cur = start; lastControl = nil
                if case .number? = tokens.first { cmd = "L" }
            default:
                return out
            }
        }
        return out
    }

    enum Token: Equatable { case command(Character), number(CGFloat) }

    static func tokenize(_ d: String) -> [Token] {
        var out: [Token] = []
        var number = ""
        func flush() {
            if !number.isEmpty, let v = Double(number) { out.append(.number(CGFloat(v))) }
            number = ""
        }
        for ch in d {
            if ch.isLetter && ch != "e" && ch != "E" {
                flush(); out.append(.command(ch))
            } else if ch == "-" {
                if let last = number.last, last == "e" || last == "E" { number.append(ch) } else { flush(); number.append(ch) }
            } else if ch == "." {
                if number.contains(".") && !number.contains("e") { flush() }
                number.append(ch)
            } else if ch.isNumber || ch == "e" || ch == "E" {
                number.append(ch)
            } else {
                flush()
            }
        }
        flush()
        return out
    }
}

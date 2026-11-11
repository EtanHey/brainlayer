#!/usr/bin/env python3
"""Force a queued telemetry write across fixture teardown and check it cannot recreate the fixture."""

import argparse
import json
import pathlib
import re
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before-ref", required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.before_ref):
        parser.error("--before-ref must be a full commit SHA")
    base = pathlib.Path(__file__).resolve().parents[4]
    output = base / ".build.noindex/caller-model-cleanup-proof"
    output.mkdir(parents=True, exist_ok=True)
    producer = (base / "brain-bar/Sources/BrainBar/DeferredStoreTelemetry.swift").read_text()
    wait = "static func waitForWritesForTesting() { queue.sync {} }"
    write = "do { try write(item, path: path) }"
    assert producer.count(wait) == producer.count(write) == 1, "Telemetry scheduling seam changed"
    producer = producer.replace(wait, "static func waitForWritesForTesting() { probeRelease(); queue.sync {} }")
    producer = producer.replace(write, "do { try write(item, path: path); probeWritten.signal() }")
    hook = """
extension DeferredStoreTelemetry {
    static let probeResume = DispatchSemaphore(value: 1)
    static let probeWritten = DispatchSemaphore(value: 0)
    static func probeHold() { queue.suspend() }
    static func probeRelease() {
        if probeResume.wait(timeout: .now()) == .success { queue.resume() }
    }
}
"""
    path = "brain-bar/Tests/BrainBarTests/StoreCallerModelTests.swift"
    old = subprocess.check_output(["git", "show", args.before_ref + ":" + path], cwd=base, text=True)
    current = (base / path).read_text()
    for name, source in (("before", old), ("after", current)):
        match = re.search(r"override func tearDownWithError\(\) throws \{\n(.*?)\n    \}", source, re.S)
        assert match, "Fixture teardown seam changed"
        program = (
            producer
            + hook
            + """
let tempDir = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
setenv("BRAINLAYER_WRITER_TELEMETRY", "1", 1)
setenv("BRAINLAYER_DEFERRED_TELEMETRY_PATH", tempDir.appendingPathComponent("logs/deferred-stores.jsonl").path, 1)
DeferredStoreTelemetry.probeHold()
DeferredStoreTelemetry.record("queued", dbPath: tempDir.appendingPathComponent("brainbar.db").path,
                             queueID: "fixture", chunkID: "fixture", origin: nil, queuedAt: nil)
func cleanup() throws {
"""
            + match.group(1)
            + """
}
try cleanup()
DeferredStoreTelemetry.probeRelease()
DeferredStoreTelemetry.waitForWritesForTesting()
let wrote = DeferredStoreTelemetry.probeWritten.wait(timeout: .now()) == .success
let recreated = FileManager.default.fileExists(atPath: tempDir.path)
print("writer_completed=\\(wrote) fixture_recreated=\\(recreated)")
if recreated { try FileManager.default.removeItem(at: tempDir) }
exit(wrote && !recreated ? 0 : 1)
"""
        )
        swift = output / f"telemetry-cleanup-{name}.swift"
        binary = swift.with_suffix("")
        swift.write_text(program)
        subprocess.run(["swiftc", str(swift), "-o", str(binary)], check=True)
        with tempfile.TemporaryDirectory(prefix="brainbar-caller-model-proof-", dir=output) as root:
            result = subprocess.run([str(binary), root], capture_output=True, text=True)
        expected = 1 if name == "before" else 0
        receipt = {
            "case": name,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr,
        }
        print(json.dumps(receipt), flush=True)
        assert result.returncode == expected, receipt
        assert result.stdout.strip() == f"writer_completed=true fixture_recreated={str(name == 'before').lower()}", (
            receipt
        )


if __name__ == "__main__":
    main()

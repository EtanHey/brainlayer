// HotkeyManager.swift — Global hotkey detection via CGEventTap.
//
// Ported from VoiceBar (PR #87). Uses .listenOnly tap (Input Monitoring
// permission) with F4 as the default hotkey for BrainBar quick capture.
//
// Gesture state machine: single tap = toggle panel, hold = not used (reserved).

import CoreGraphics
import Foundation

// MARK: - Gesture State Machine

/// Detects hold vs tap vs double-tap from raw key events.
/// - Single tap: toggle quick capture panel
/// - Hold (250ms+): reserved for future use
/// - Double-tap (within 400ms): reserved for future use
final class GestureStateMachine {
    enum State: Sendable, Equatable {
        case idle
        case waitingForHoldThreshold
        case holding
        case waitingForDoubleTap
    }

    private(set) var state: State = .idle
    private var holdTimer: DispatchWorkItem?
    private var doubleTapTimer: DispatchWorkItem?

    static let holdThresholdMs: Int = 250
    static let doubleTapWindowMs: Int = 400

    var onHoldStart: () -> Void = {}
    var onHoldEnd: () -> Void = {}
    var onSingleTap: () -> Void = {}
    var onDoubleTap: () -> Void = {}

    func handleKeyDown() {
        switch state {
        case .waitingForDoubleTap:
            doubleTapTimer?.cancel()
            state = .idle
            onDoubleTap()
        case .idle:
            state = .waitingForHoldThreshold
            let timer = DispatchWorkItem { [weak self] in
                guard let self, state == .waitingForHoldThreshold else { return }
                state = .holding
                onHoldStart()
            }
            holdTimer = timer
            DispatchQueue.main.asyncAfter(
                deadline: .now() + .milliseconds(Self.holdThresholdMs),
                execute: timer
            )
        default:
            break
        }
    }

    func handleKeyUp() {
        switch state {
        case .waitingForHoldThreshold:
            holdTimer?.cancel()
            state = .waitingForDoubleTap
            let timer = DispatchWorkItem { [weak self] in
                guard let self, state == .waitingForDoubleTap else { return }
                state = .idle
                onSingleTap()
            }
            doubleTapTimer = timer
            DispatchQueue.main.asyncAfter(
                deadline: .now() + .milliseconds(Self.doubleTapWindowMs),
                execute: timer
            )
        case .holding:
            state = .idle
            onHoldEnd()
        default:
            break
        }
    }

    func reset() {
        holdTimer?.cancel()
        doubleTapTimer?.cancel()
        state = .idle
    }
}

// MARK: - Tap Context

private final class TapContext: @unchecked Sendable {
    let gesture: GestureStateMachine
    let targetKeycodes: Set<Int64>
    let useModifierMode: Bool
    var tap: CFMachPort?

    init(gesture: GestureStateMachine, keycodes: Set<Int64>, modifierMode: Bool) {
        self.gesture = gesture
        targetKeycodes = keycodes
        useModifierMode = modifierMode
    }
}

// MARK: - C Callback

private func hotkeyCallback(
    _: CGEventTapProxy,
    type: CGEventType,
    event: CGEvent,
    userInfo: UnsafeMutableRawPointer?
) -> Unmanaged<CGEvent>? {
    guard let userInfo else { return Unmanaged.passUnretained(event) }
    let ctx = Unmanaged<TapContext>.fromOpaque(userInfo).takeUnretainedValue()

    if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
        if let tap = ctx.tap {
            CGEvent.tapEnable(tap: tap, enable: true)
            NSLog("[BrainBar.Hotkey] Re-enabled event tap after system disable")
        }
        return Unmanaged.passUnretained(event)
    }

    let keycode = event.getIntegerValueField(.keyboardEventKeycode)

    if ctx.useModifierMode {
        guard ctx.targetKeycodes.contains(keycode) else {
            return Unmanaged.passUnretained(event)
        }
        let isDown = event.flags.contains(.maskCommand)
        DispatchQueue.main.async {
            if isDown { ctx.gesture.handleKeyDown() }
            else { ctx.gesture.handleKeyUp() }
        }
    } else {
        guard ctx.targetKeycodes.contains(keycode) else {
            return Unmanaged.passUnretained(event)
        }
        let autorepeat = event.getIntegerValueField(.keyboardEventAutorepeat)
        guard autorepeat == 0 else { return Unmanaged.passUnretained(event) }
        let isDown = (type == .keyDown)
        DispatchQueue.main.async {
            if isDown { ctx.gesture.handleKeyDown() }
            else { ctx.gesture.handleKeyUp() }
        }
    }

    return Unmanaged.passUnretained(event)
}

// MARK: - Hotkey Manager

final class HotkeyManager {
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var targetKeycodes: Set<Int64> = [118, 129] // F4 standard + F4 media
    private var useModifierMode: Bool = false
    private let gesture: GestureStateMachine
    private var tapContext: TapContext?

    init(gesture: GestureStateMachine) {
        self.gesture = gesture
    }

    static func hasPermission() -> Bool { CGPreflightListenEventAccess() }
    static func requestPermission() { CGRequestListenEventAccess() }

    func start() -> Bool {
        guard HotkeyManager.hasPermission() else {
            NSLog("[BrainBar.Hotkey] Input Monitoring permission not granted")
            HotkeyManager.requestPermission()
            return false
        }

        let mask = if useModifierMode {
            CGEventMask(1 << CGEventType.flagsChanged.rawValue)
        } else {
            CGEventMask(
                (1 << CGEventType.keyDown.rawValue) |
                (1 << CGEventType.keyUp.rawValue)
            )
        }

        let ctx = TapContext(gesture: gesture, keycodes: targetKeycodes, modifierMode: useModifierMode)
        tapContext = ctx
        let ctxPtr = Unmanaged.passUnretained(ctx).toOpaque()

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .listenOnly,
            eventsOfInterest: mask,
            callback: hotkeyCallback,
            userInfo: ctxPtr
        ) else {
            NSLog("[BrainBar.Hotkey] Failed to create CGEventTap — check permissions")
            return false
        }

        eventTap = tap
        ctx.tap = tap

        let source = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)!
        CFRunLoopAddSource(CFRunLoopGetMain(), source, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        runLoopSource = source

        NSLog("[BrainBar.Hotkey] Event tap started — F4 keycodes: %@", String(describing: targetKeycodes))
        return true
    }

    func stop() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
            CFMachPortInvalidate(tap)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetMain(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil
        tapContext = nil
    }

    func configure(keycodes: Set<Int64>, useModifierMode: Bool) {
        let wasRunning = eventTap != nil
        if wasRunning { stop() }
        targetKeycodes = keycodes
        self.useModifierMode = useModifierMode
        if wasRunning { _ = start() }
    }
}

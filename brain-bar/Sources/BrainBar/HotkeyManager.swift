// HotkeyManager.swift — Optional global hotkey via CGEventTap (fallback path).
//
// Primary hotkeys are expected from Karabiner Elements → `open brainbar://toggle`
// or `brainbar://search` (see brain-bar/karabiner/brainbar-f4.json). This tap
// is only started when the user enables “CGEventTap fallback” in the status
// popover (Input Monitoring / Listen Events required).
//
// Gesture state machine: single tap = toggle panel, hold = not used (reserved).

import ApplicationServices
import CoreGraphics
import Foundation

struct HotkeyPermissionStatus: Equatable {
    let inputMonitoringGranted: Bool
    let accessibilityGranted: Bool

    var isSatisfied: Bool {
        inputMonitoringGranted && accessibilityGranted
    }

    var missingPermissionsMessage: String {
        switch (inputMonitoringGranted, accessibilityGranted) {
        case (false, false):
            return "Input Monitoring and Accessibility"
        case (false, true):
            return "Input Monitoring"
        case (true, false):
            return "Accessibility"
        case (true, true):
            return "None"
        }
    }
}

final class HotkeyDebouncer {
    private let window: TimeInterval
    private var lastProcessedKeyDownAt: Date?

    init(windowMs: Int = 300) {
        window = Double(windowMs) / 1000.0
    }

    func shouldProcessKeyDown(at now: Date = Date()) -> Bool {
        if let lastProcessedKeyDownAt, now.timeIntervalSince(lastProcessedKeyDownAt) < window {
            return false
        }
        lastProcessedKeyDownAt = now
        return true
    }
}

enum HotkeyAction: Equatable {
    case none
    case keyDown
    case keyUp
}

struct HotkeyEventDecision: Equatable {
    let matchesHotkey: Bool
    let shouldConsumeEvent: Bool
    let action: HotkeyAction

    static func make(
        type: CGEventType,
        keycode: Int64,
        autorepeat: Int64,
        targetKeycodes: Set<Int64>,
        useModifierMode: Bool,
        debouncer: HotkeyDebouncer,
        now: Date = Date()
    ) -> Self {
        guard targetKeycodes.contains(keycode) else {
            return Self(matchesHotkey: false, shouldConsumeEvent: false, action: .none)
        }

        if useModifierMode {
            return Self(matchesHotkey: true, shouldConsumeEvent: true, action: .keyDown)
        }

        if autorepeat != 0 {
            return Self(matchesHotkey: true, shouldConsumeEvent: false, action: .none)
        }

        switch type {
        case .keyDown:
            guard debouncer.shouldProcessKeyDown(at: now) else {
                return Self(matchesHotkey: true, shouldConsumeEvent: false, action: .none)
            }
            return Self(matchesHotkey: true, shouldConsumeEvent: true, action: .keyDown)
        case .keyUp:
            return Self(matchesHotkey: true, shouldConsumeEvent: true, action: .keyUp)
        default:
            return Self(matchesHotkey: true, shouldConsumeEvent: false, action: .none)
        }
    }
}

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
    let debouncer: HotkeyDebouncer
    var tap: CFMachPort?

    init(gesture: GestureStateMachine, keycodes: Set<Int64>, modifierMode: Bool, debouncer: HotkeyDebouncer) {
        self.gesture = gesture
        targetKeycodes = keycodes
        useModifierMode = modifierMode
        self.debouncer = debouncer
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
        return nil
    }

    let decision = HotkeyEventDecision.make(
        type: type,
        keycode: keycode,
        autorepeat: event.getIntegerValueField(.keyboardEventAutorepeat),
        targetKeycodes: ctx.targetKeycodes,
        useModifierMode: ctx.useModifierMode,
        debouncer: ctx.debouncer
    )

    guard decision.matchesHotkey else {
        return Unmanaged.passUnretained(event)
    }

    DispatchQueue.main.async {
        switch decision.action {
        case .keyDown:
            ctx.gesture.handleKeyDown()
        case .keyUp:
            ctx.gesture.handleKeyUp()
        case .none:
            break
        }
    }

    return decision.shouldConsumeEvent ? nil : Unmanaged.passUnretained(event)
}

// MARK: - Hotkey Manager

final class HotkeyManager {
    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var targetKeycodes: Set<Int64> = [118, 129] // F4 standard + F4 media
    private var useModifierMode: Bool = false
    private let gesture: GestureStateMachine
    private var tapContext: TapContext?
    private let debouncer = HotkeyDebouncer()

    init(gesture: GestureStateMachine) {
        self.gesture = gesture
    }

    static func permissionStatus() -> HotkeyPermissionStatus {
        HotkeyPermissionStatus(
            inputMonitoringGranted: CGPreflightListenEventAccess(),
            accessibilityGranted: AXIsProcessTrusted()
        )
    }

    static func hasPermission() -> Bool { permissionStatus().isSatisfied }
    static func requestPermission() { CGRequestListenEventAccess() }

    func start() -> Bool {
        let permissions = HotkeyManager.permissionStatus()
        guard permissions.isSatisfied else {
            NSLog("[BrainBar.Hotkey] Missing permission: %@", permissions.missingPermissionsMessage)
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

        let ctx = TapContext(
            gesture: gesture,
            keycodes: targetKeycodes,
            modifierMode: useModifierMode,
            debouncer: debouncer
        )
        tapContext = ctx
        let ctxPtr = Unmanaged.passUnretained(ctx).toOpaque()

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
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

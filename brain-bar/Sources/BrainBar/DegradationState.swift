import Foundation

/// Tracks read-path degradation for BrainBar UI surfaces (graph + injections).
///
/// After PR #312 removed the FastAPI daemon, the UI process opens SQLite
/// directly and contends with the Python enrich-supervisor + drain for the
/// writer pidfile (PR #309). Transient ReadOnly / busy / locked errors can
/// surface here. Instead of blanking the UI or silently swallowing the error,
/// surfaces report a degradation state so the user sees a small badge and
/// knows data may be stale rather than absent.
///
/// Etan-mandate 2026-05-22 ~18:00 IDT: "WITHOUT DEGRATION" [sic] — no blank screens,
/// no "warming memory" lingering after data should be available; degraded ≠
/// hidden.
enum DegradationState: Equatable, Sendable {
    case healthy
    case degraded(reason: String)

    var isDegraded: Bool {
        if case .degraded = self { return true }
        return false
    }

    var reason: String? {
        if case .degraded(let reason) = self { return reason }
        return nil
    }
}

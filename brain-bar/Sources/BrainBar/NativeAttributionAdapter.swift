import Foundation

/// Projects only persisted native-record declarations. It deliberately does not infer a person,
/// transport sender, relay, or original author from provider/session envelopes.
enum NativeAttributionAdapter {
    private static let maximumDeclaredFieldBytes = 4_096

    static func project(
        contexts: [[String: Any]],
        contextStatus: String,
        evidenceKeys: [String]
    ) -> [String: Any] {
        let label = declaredField(
            "attributionAgent",
            contexts: contexts,
            contextStatus: contextStatus,
            missingStatus: "unknown_missing_declared_agent_label",
            malformedStatus: "unknown_malformed_declared_agent_label",
            conflictStatus: "unknown_conflicting_declared_agent_labels"
        )
        let role = declaredField(
            "declared_message_role",
            contexts: contexts,
            contextStatus: contextStatus,
            missingStatus: "unknown_missing_raw_declared_role",
            malformedStatus: "unknown_malformed_raw_declared_role",
            conflictStatus: "unknown_conflicting_raw_declared_roles"
        )
        let hasDeclaredField = label.value is String || role.value is String
        let hasDispute = label.status.contains("conflicting") || label.status.contains("malformed")
            || role.status.contains("conflicting") || role.status.contains("malformed")

        return [
            "recorded_emitter": [
                "principal": NSNull(),
                "principal_status": "unknown_no_distinct_identity_binding",
                "declared_agent_label": label.value,
                "declared_agent_label_status": label.status,
                "raw_declared_role": role.value,
                "raw_declared_role_status": role.status,
                "status": hasDispute
                    ? "unknown_conflicting_or_malformed_record_fields"
                    : (hasDeclaredField ? "declared_fields_no_principal_binding" : "unknown_no_recorded_emitter_fields"),
                "evidence_kind": contexts.isEmpty ? NSNull() : "record_context",
                "evidence_keys": evidenceKeys,
            ] as [String: Any],
            "observed_sender": [
                "principal": NSNull(),
                "status": "unknown_no_transport_sender_evidence",
            ] as [String: Any],
            "original_content_author": [
                "principal": NSNull(),
                "status": "unknown_no_authored_source_lineage",
            ] as [String: Any],
            "relay": [
                "status": "unknown_no_delivery_receipt",
            ] as [String: Any],
        ]
    }

    static var omittedResponseBudget: [String: Any] {
        ["status": "omitted_response_budget"]
    }

    private static func declaredField(
        _ field: String,
        contexts: [[String: Any]],
        contextStatus: String,
        missingStatus: String,
        malformedStatus: String,
        conflictStatus: String
    ) -> (value: Any, status: String) {
        if contextStatus == "omitted_byte_cap" || contextStatus == "omitted_match_cap" {
            return (NSNull(), "unknown_omitted_record_context")
        }
        if contextStatus == "unavailable_legacy_schema" {
            return (NSNull(), "unknown_legacy_record_context_schema")
        }
        if contextStatus == "unavailable_missing_physical_index" {
            return (NSNull(), "unknown_missing_record_context_physical_index")
        }
        if contextStatus == "unavailable_database_error" {
            return (NSNull(), "unknown_record_context_read_error")
        }
        if contextStatus == "unavailable_missing" || contextStatus == "unavailable_field_only" {
            return (NSNull(), "unknown_no_record_context")
        }

        var values = Set<String>()
        var sawMalformed = false
        for context in contexts where context.keys.contains(field) {
            guard let value = context[field] as? String,
                  !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                  value.utf8.count <= maximumDeclaredFieldBytes,
                  !value.utf8.contains(0) else {
                sawMalformed = true
                continue
            }
            values.insert(value)
        }
        guard !sawMalformed else { return (NSNull(), malformedStatus) }
        guard !values.isEmpty else { return (NSNull(), missingStatus) }
        guard values.count == 1, let value = values.first else { return (NSNull(), conflictStatus) }
        return (value, "declared_unverified")
    }
}

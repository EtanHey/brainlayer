import XCTest
@testable import BrainBar

final class BrainLayerConfigTests: XCTestCase {
    func testDefaultConfigURLUsesUnifiedBrainLayerEnvPath() {
        let home = URL(fileURLWithPath: "/Users/example", isDirectory: true)

        XCTAssertEqual(
            BrainLayerConfigStore.defaultConfigURL(homeDirectory: home).path,
            "/Users/example/.config/brainlayer/brainlayer.env"
        )
    }

    func testParsesUnifiedConfigWithoutLeakingPlainSecret() throws {
        let document = try BrainLayerEnvDocument(
            text: """
            # BrainLayer private config.
            GOOGLE_API_KEY='plain-secret'
            BRAINLAYER_SYSTEM_ENABLED=1
            BRAINLAYER_ENRICH_ENABLED=off
            BRAINLAYER_ENRICH_MODE=local
            BRAINLAYER_ENRICH_PROVIDER=gemini
            BRAINLAYER_ENRICH_BACKEND=ollama
            BRAINLAYER_LAUNCHD_DRAIN_ENABLED=0
            """
        )

        XCTAssertEqual(document.config.googleAPIKey.kind, .plainPresent)
        XCTAssertEqual(document.config.googleAPIKey.displayText, "Stored in config file")
        XCTAssertEqual(document.config.enrichmentEnabled, false)
        XCTAssertEqual(document.config.enrichmentMode, .local)
        XCTAssertEqual(document.config.enrichmentProvider, .gemini)
        XCTAssertEqual(document.config.enrichmentBackend, "ollama")
        XCTAssertEqual(document.config.launchdJobs[.drain]?.enabled, false)
        XCTAssertFalse(document.config.googleAPIKey.displayText.contains("plain-secret"))
    }

    func testParsesOnePasswordGoogleKeyReference() throws {
        let document = try BrainLayerEnvDocument(
            text: """
            GOOGLE_API_KEY="$(op read 'op://Private/Google AI/Gemini API key')"
            BRAINLAYER_ENRICH_PROVIDER=gemini
            BRAINLAYER_ENRICH_BACKEND=gemini
            """
        )

        XCTAssertEqual(document.config.googleAPIKey.kind, .onePasswordReference)
        XCTAssertEqual(document.config.googleAPIKey.displayText, "1Password reference")
    }

    func testUpdatingConfigPreservesCommentsAndUnmanagedKeys() throws {
        var document = try BrainLayerEnvDocument(
            text: """
            # keep this
            CUSTOM_FLAG=keep
            GOOGLE_API_KEY='old-secret'
            BRAINLAYER_ENRICH_ENABLED=1
            BRAINLAYER_ENRICH_MODE=remote
            BRAINLAYER_ENRICH_PROVIDER=gemini
            BRAINLAYER_ENRICH_BACKEND=gemini
            BRAINLAYER_ENRICH_RATE=99
            BRAINLAYER_ENRICH_CONCURRENCY=7
            BRAINLAYER_MAX_COMMIT_BATCH=88
            BRAINLAYER_GEMINI_SERVICE_TIER=standard
            BRAINLAYER_DISABLED_SLEEP_SECONDS=42
            BRAINLAYER_LAUNCHD_DRAIN_ENABLED=1
            """
        )

        document.update { config in
            config.googleAPIKey = .onePasswordReference("op://Private/Google AI/Gemini API key")
            config.enrichmentEnabled = false
            config.enrichmentMode = .local
            config.enrichmentBackend = "mlx"
            config.launchdJobs[.drain]?.enabled = false
        }

        let rendered = document.rendered()
        XCTAssertTrue(rendered.contains("# keep this"))
        XCTAssertTrue(rendered.contains("CUSTOM_FLAG=keep"))
        XCTAssertTrue(rendered.contains("GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\""))
        XCTAssertTrue(rendered.contains("BRAINLAYER_ENRICH_ENABLED=0"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_ENRICH_MODE=local"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_ENRICH_BACKEND=mlx"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_ENRICH_RATE=99"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_ENRICH_CONCURRENCY=7"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_MAX_COMMIT_BATCH=88"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_GEMINI_SERVICE_TIER=standard"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_DISABLED_SLEEP_SECONDS=42"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_LAUNCHD_DRAIN_ENABLED=0"))
        XCTAssertFalse(rendered.contains("old-secret"))
    }

    func testSaveMigratesLegacyGoogleKeyAliasToCanonicalKeyAndClearsAlias() throws {
        var document = try BrainLayerEnvDocument(
            text: """
            GOOGLE_GENERATIVE_AI_API_KEY='legacy-secret'
            BRAINLAYER_ENRICH_PROVIDER=gemini
            """
        )

        XCTAssertEqual(document.config.googleAPIKey.kind, .plainPresent)

        document.update { config in
            config.googleAPIKey = .missing
        }

        let rendered = document.rendered()
        XCTAssertTrue(rendered.contains("GOOGLE_API_KEY="))
        XCTAssertTrue(rendered.contains("GOOGLE_GENERATIVE_AI_API_KEY="))
        XCTAssertFalse(rendered.contains("legacy-secret"))
    }

    func testWritesMissingConfigWithFullSchemaDefaults() throws {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent("brainbar-config-\(UUID().uuidString)", isDirectory: true)
        let configURL = directory.appendingPathComponent("brainlayer.env")
        let store = BrainLayerConfigStore(configURL: configURL)
        defer { try? FileManager.default.removeItem(at: directory) }

        try store.save(BrainLayerConfig.defaultConfig)

        let content = try String(contentsOf: configURL, encoding: .utf8)
        XCTAssertTrue(content.contains("BRAINLAYER_ENRICH_ENABLED=1"))
        XCTAssertTrue(content.contains("BRAINLAYER_ENRICH_MODE=remote"))
        XCTAssertTrue(content.contains("BRAINLAYER_ENRICH_PROVIDER=gemini"))
        XCTAssertTrue(content.contains("BRAINLAYER_ENRICH_BACKEND=gemini"))
        XCTAssertTrue(content.contains("BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED=1"))
        XCTAssertTrue(content.contains("BRAINLAYER_LAUNCHD_DRAIN_ENABLED=1"))
    }
}

import XCTest
@testable import BrainBar

final class BrainLayerConfigTests: XCTestCase {
    func testRetiredRetrievalKeyRemainsUnmanagedOnRender() throws {
        let input = "BRAINLAYER_SHOW_RETRIEVAL_TOOLS=1\nBRAINLAYER_SYSTEM_ENABLED=1\n"
        let document = try BrainLayerEnvDocument(text: input)
        let rendered = document.rendered()
        XCTAssertTrue(rendered.contains("BRAINLAYER_SHOW_RETRIEVAL_TOOLS=1"))
        XCTAssertEqual(rendered.components(separatedBy: "BRAINLAYER_SHOW_RETRIEVAL_TOOLS=").count, 2)
    }

    func testRetiredEnrichmentJobKeyRemainsUnmanagedOnRender() throws {
        var document = try BrainLayerEnvDocument(text: "BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED=0\n")
        document.update { $0.systemEnabled = false }
        let rendered = document.rendered()
        XCTAssertTrue(rendered.contains("BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED=0"))
        XCTAssertEqual(rendered.components(separatedBy: "BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED=").count, 2)
    }

    func testLaunchctlProbeDistinguishesMissingJobFromProbeFailure() {
        let missing = BrainLayerLaunchdStatusProvider(
            commandRunner: { _ in
                BrainLayerLaunchdCommandResult(
                    terminationStatus: 113,
                    output: "Could not find service com.brainlayer.watch"
                )
            },
            uidProvider: { 501 }
        )
        XCTAssertEqual(missing.sample()[.watch], .unloaded)

        let failed = BrainLayerLaunchdStatusProvider(
            commandRunner: { _ in
                BrainLayerLaunchdCommandResult(
                    terminationStatus: 1,
                    output: "operation timed out"
                )
            },
            uidProvider: { 501 }
        )
        XCTAssertEqual(failed.sample()[.watch], .probeError("launchctl exited 1"))
    }

    func testSecretDescriptionsStayRedacted() {
        let secret = "super-secret-fixture-value"
        let value = BrainLayerGoogleAPIKey.plain(secret)
        var config = BrainLayerConfig.defaultConfig
        config.googleAPIKey = value

        XCTAssertFalse(String(describing: value).contains(secret))
        XCTAssertFalse(String(reflecting: value).contains(secret))
        XCTAssertFalse(String(describing: config).contains(secret))
        XCTAssertFalse(String(reflecting: config).contains(secret))
        XCTAssertEqual(String(describing: value), "Stored in config file")
    }

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

    func testPlainGoogleKeyWithApostropheRoundTripsThroughManagedShellRendering() throws {
        let expected = BrainLayerGoogleAPIKey.plain("fixture'key")
        var document = BrainLayerEnvDocument(config: .defaultConfig)
        document.update { $0.googleAPIKey = expected }

        let reloaded = try BrainLayerEnvDocument(text: document.rendered()).config

        XCTAssertEqual(reloaded.googleAPIKey, expected)
        XCTAssertTrue(reloaded.persistedValuesEqual(to: document.config))
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
            config.launchdJobs[.drain]?.enabled = false
        }

        let rendered = document.rendered()
        XCTAssertTrue(rendered.contains("# keep this"))
        XCTAssertTrue(rendered.contains("CUSTOM_FLAG=keep"))
        XCTAssertTrue(rendered.contains("GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\""))
        XCTAssertTrue(rendered.contains("BRAINLAYER_ENRICH_ENABLED=1"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_ENRICH_MODE=remote"))
        XCTAssertTrue(rendered.contains("BRAINLAYER_ENRICH_BACKEND=gemini"))
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
        XCTAssertFalse(content.contains("BRAINLAYER_ENRICH_ENABLED="))
        XCTAssertFalse(content.contains("BRAINLAYER_ENRICH_MODE="))
        XCTAssertFalse(content.contains("BRAINLAYER_ENRICH_PROVIDER="))
        XCTAssertFalse(content.contains("BRAINLAYER_ENRICH_BACKEND="))
        XCTAssertFalse(content.contains("BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED="))
        XCTAssertTrue(content.contains("BRAINLAYER_LAUNCHD_DRAIN_ENABLED=1"))
    }

    func testLegacyEnrichmentKeysAndGoogleKeyRoundTripWithoutRewriting() throws {
        let legacy = """
        # existing enrichment configuration
        BRAINLAYER_ENRICH_ENABLED=1
        BRAINLAYER_ENRICH_MODE=remote
        BRAINLAYER_ENRICH_PROVIDER=unsupported-legacy-provider
        BRAINLAYER_ENRICH_BACKEND='legacy backend'
        BRAINLAYER_ENRICH_RATE=99
        BRAINLAYER_ENRICH_CONCURRENCY=7
        BRAINLAYER_MAX_COMMIT_BATCH=88
        BRAINLAYER_GEMINI_SERVICE_TIER=standard
        BRAINLAYER_DISABLED_SLEEP_SECONDS=42
        BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED=0
        """
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let configURL = directory.appendingPathComponent("legacy.env")
        let store = BrainLayerConfigStore(configURL: configURL)
        for google in ["GOOGLE_API_KEY='synthetic-fixture-key'", "GOOGLE_API_KEY=\"$(op read 'op://Fixture/Google/key')\""] {
            try (google + "\n" + legacy + "\nBRAINLAYER_SYSTEM_ENABLED=1\n").write(to: configURL, atomically: true, encoding: .utf8)
            var config = try store.loadDocument().config
            config.systemEnabled = false
            try store.save(config)
            let rendered = try String(contentsOf: configURL, encoding: .utf8)
            for line in (google + "\n" + legacy).split(separator: "\n") {
                XCTAssertTrue(rendered.split(separator: "\n").contains(line), String(line))
            }
            let reloaded = try BrainLayerEnvDocument(text: rendered).config
            XCTAssertEqual(reloaded.googleAPIKey, config.googleAPIKey)
            XCTAssertFalse(reloaded.systemEnabled)
            XCTAssertTrue(reloaded.persistedValuesEqual(to: config))
        }
    }

    func testNewConfigDoesNotCreateEnrichmentSettings() {
        let rendered = BrainLayerEnvDocument(config: .defaultConfig).rendered()
        for key in ["BRAINLAYER_ENRICH_ENABLED", "BRAINLAYER_ENRICH_MODE", "BRAINLAYER_ENRICH_PROVIDER", "BRAINLAYER_ENRICH_BACKEND",
                    "BRAINLAYER_ENRICH_RATE", "BRAINLAYER_ENRICH_CONCURRENCY", "BRAINLAYER_MAX_COMMIT_BATCH",
                    "BRAINLAYER_GEMINI_SERVICE_TIER", "BRAINLAYER_DISABLED_SLEEP_SECONDS", "BRAINLAYER_LAUNCHD_ENRICHMENT_ENABLED"] {
            XCTAssertFalse(rendered.contains(key + "="), key)
        }
        XCTAssertTrue(rendered.contains("GOOGLE_API_KEY="))
    }

}

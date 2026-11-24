"""Declarative layout contracts; no hosting view, window, logs or database."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "brain-bar/Sources/BrainBar"


def test_initial_status_menu_is_populated_before_appkit_can_open_it():
    source = (ROOT / "BrainBarStatusPopoverController.swift").read_text()
    setup = source.split("private func configureContextMenu() {", 1)[1].split("@objc private func openDashboard", 1)[0]
    assert "newMenuBuilder.populate(contextMenuForTesting)" in setup
    menu = (ROOT / "NewDesign/NDMenu.swift").read_text()
    assert "action: .settings" in menu
    assert "action: .quit" in menu


def test_shared_settings_has_no_design_or_old_menu_choice():
    source = (ROOT / "BrainBarSettingsView.swift").read_text()
    assert "designRow" not in source
    assert 'Picker("Design"' not in source
    assert "BrainBarNewMenu" not in source


def test_drive_state_is_after_spacer():
    source = (ROOT / "NewDesign/NDSettings.swift").read_text()
    row = source.split("row(Say.stageDrive) {", 1)[1].split("ND.hair", 1)[0]
    assert row.index("Spacer()") < row.index("Text(driveState)") < row.index("driveAction")


def test_backups_axis_draws_now_tick_at_shared_time_position():
    source = (ROOT / "NewDesign/NDPages.swift").read_text()
    axis = source.split("struct NDRailAxis:", 1)[1].split("struct NDLegendItem", 1)[0]
    assert "tick.hours == 0" in axis
    assert ".offset(x: x - 0.5" in axis


def test_older_run_label_drops_left_chevron():
    source = (ROOT / "NewDesign/NDVisuals.swift").read_text()
    older = source.split("if e.hours < -past {", 1)[1].split("if e.hours > ahead", 1)[0]
    assert "chevron(left: true" not in older


def test_frequency_uses_schedule_evidence():
    source = (ROOT / "NewDesign/NDPages.swift").read_text()
    assert "NDJobsPage.frequent.contains(lane.job)" not in source
    assert "input?.runsFrequently" in source
    inputs = (ROOT / "NewDesign/NDInputs.swift").read_text()
    assert "< 3600" in inputs


def test_next_run_hover_is_attached_at_ring():
    source = (ROOT / "NewDesign/NDVisuals.swift").read_text()
    rail = source.split("struct NDRail:", 1)[1].split("// MARK: Numbers", 1)[0]
    assert "event.kind == .next" in rail
    assert ".help(help)" in rail
    assert ".position(x: x(event.hours, geo.size.width), y: 7)" in rail


def test_new_settings_has_no_design_choice_or_switch_back_caption():
    settings = (ROOT / "NewDesign/NDSettings.swift").read_text()
    assert "row(Say.design)" not in settings
    assert "BrainBarDesign" not in settings
    assert "classicCaption" not in settings
    sentences = (ROOT / "BrainBarMenuSentences.swift").read_text()
    assert "You can switch back at any time." not in sentences


def test_activity_labels_carry_each_halfs_scale():
    source = (ROOT / "NewDesign/NDWindow.swift").read_text()
    assert "scale: known ? scales?.saves : nil" in source
    assert "scale: known ? scales?.reads : nil" in source
    assert "Text(scale).font(NDType.caption).foregroundStyle(ND.t3)" in source


def test_old_design_preferences_cannot_select_another_design():
    source = (ROOT / "NewDesign/BrainBarDesign.swift").read_text()
    assert "static let defaultValue: BrainBarDesign = .new" in source
    assert "case classic" not in source
    assert "defaults.string(forKey:" not in source


def test_activity_header_stays_22_points_and_scale_stays_inline():
    source = (ROOT / "NewDesign/NDWindow.swift").read_text()
    assert "static let headerHeight: CGFloat = 22" in source
    assert ".frame(height: Self.headerHeight)" in source
    header = source.split("@ViewBuilder private func chart(", 1)[1].split(
        ".accessibilityElement(children: .combine)", 1
    )[0]
    assert ".lineLimit(1)" in header
    legend = source.split("private func legend(color:", 1)[1].split("/// The hover readout", 1)[0]
    assert "VStack" not in legend
    assert "Text(scale).font(NDType.caption).foregroundStyle(ND.t3)" in legend


def test_queue_and_storage_help_does_not_promise_a_fixed_cadence():
    source = (ROOT / "BrainBarMenuSentences.swift").read_text()
    help_rows = source.split("static let jobHelp:", 1)[1].split("// Station popovers", 1)[0]
    assert "Every few minutes." not in help_rows
    pages = (ROOT / "NewDesign/NDPages.swift").read_text()
    assert ".help(Self.jobHelp(input," in pages
    assert "input?.runsFrequently" in pages.split("static func jobHelp(", 1)[1]


def test_scale_wording_uses_slash_with_bucket_units():
    source = (ROOT / "BrainBarMenuSentences.swift").read_text()
    formatter = source.split("static func upTo(", 1)[1].split("// Revision", 1)[0]
    assert "/min" in formatter and "/\\(minutes) min" in formatter
    assert " per " not in formatter


def test_stored_design_regression_checks_new_menu_and_icon():
    source = (ROOT.parents[1] / "Tests/BrainBarTests/BrainBarNewDesignTests.swift").read_text()
    assert "testEveryStoredDesignShowsTheNewMenuAndIcon" in source
    assert "XCTAssertEqual(actual.image, expected.image)" in source
    assert "XCTAssertGreaterThanOrEqual(actual.hostedNew, 5)" in source


def test_outside_timeline_legend_matches_bare_age():
    source = (ROOT / "NewDesign/NDPages.swift").read_text()
    sample = source.split("NDLegendItem(words: Say.outsideTimeline)", 1)[1].split(".frame(height:", 1)[0]
    assert 'Text("6 d")' in sample
    assert 'Text("…")' not in sample


def test_retention_errors_cannot_be_housekeeping():
    source = (ROOT / "NewDesign/NDBackupHistory.swift").read_text()
    assert 'guard row["event"] == nil, row["attempted_at"] != nil else { continue }' not in source
    assert "case damage" in source
    assert 'row["error"]' in source
    producer = (ROOT.parents[2] / "src/brainlayer/jsonl_backup.py").read_text()
    assert 'report("jsonl-retention", result.get("local_retention_error"))' in producer
    notifications = (ROOT / "BrainBarFaultNotifications.swift").read_text()
    assert "next[id] = .init(id: id, title: message, causes: [cause])" in notifications


def test_invented_retention_error_stays_in_log_and_active_job_alert(tmp_path):
    import json

    from brainlayer import job_alerts, jsonl_backup

    message = "invented inventory unavailable"
    log = tmp_path / "invented-backup.log"
    alerts = tmp_path / "invented-alerts.json"
    jsonl_backup._retention_warning(log, message)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row == {"event": "local_retention_error", "error": message}
    assert job_alerts.report("jsonl-retention", row["error"], path=alerts, notify=lambda _: None)
    assert job_alerts.active_alerts(alerts)["jsonl-retention"] == message

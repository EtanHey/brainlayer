"""Declarative layout contracts; no hosting view, window, logs or database."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "brain-bar/Sources/BrainBar"


def test_classic_design_is_first_settings_row():
    source = (ROOT / "BrainBarSettingsView.swift").read_text()
    content = source.split("var body: some View {", 1)[1]
    stack = content.split("VStack(alignment: .leading, spacing: 24) {", 1)[1]
    assert stack.index("designRow") < stack.index("header")
    advanced = source.split("case .advanced:")[-1].split("private var designRow", 1)[0]
    assert 'Picker("Design"' not in advanced


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
    assert "input?.scheduleInterval" in source
    assert "< 3600" in source


def test_next_run_hover_is_attached_at_ring():
    source = (ROOT / "NewDesign/NDVisuals.swift").read_text()
    rail = source.split("struct NDRail:", 1)[1].split("// MARK: Numbers", 1)[0]
    assert "event.kind == .next" in rail
    assert ".help(help)" in rail
    assert ".position(x: x(event.hours, geo.size.width), y: 7)" in rail


def test_new_settings_design_is_first_row_and_caption_keeps_switch_back():
    settings = (ROOT / "NewDesign/NDSettings.swift").read_text()
    assert settings.index("row(Say.design)") < settings.index("row(Say.stageDrive)")
    assert "Text(Say.classicCaption)" in settings
    sentences = (ROOT / "BrainBarMenuSentences.swift").read_text()
    assert "You can switch back at any time." in sentences


def test_activity_labels_carry_each_halfs_scale():
    source = (ROOT / "NewDesign/NDWindow.swift").read_text()
    assert "scale: known ? scales?.saves : nil" in source
    assert "scale: known ? scales?.reads : nil" in source
    assert "Text(scale).font(NDType.caption).foregroundStyle(ND.t3)" in source


def test_new_default_keeps_stored_choice_precedence():
    source = (ROOT / "NewDesign/BrainBarDesign.swift").read_text()
    assert "static let defaultValue: BrainBarDesign = .new" in source
    assert "defaults.string(forKey: key).flatMap(Self.init(rawValue:)) ?? defaultValue" in source

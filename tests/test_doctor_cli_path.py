import pytest

from brainlayer import doctor


@pytest.mark.parametrize("order,severity", [("shadow,keg", "fatal"), ("keg,shadow", "warning"), ("keg", None)])
@pytest.mark.parametrize(
    "editable,user_site,user_owned", [(True, False, False), (False, True, False), (False, True, True)]
)
def test_cli_path_order(tmp_path, monkeypatch, order, severity, editable, user_site, user_owned):
    monkeypatch.setenv("HOME", str(tmp_path))
    if user_site:
        (tmp_path / "Library/Python/3.13/lib/python/site-packages/brainlayer-1.0.dist-info").mkdir(parents=True)
    keg = tmp_path / "Cellar/brainlayer/1.5.45"
    monkeypatch.setattr(doctor.sys, "prefix", str(keg / "libexec/venv"))
    paths = {}
    for name, prefix in [("keg", keg), ("shadow", tmp_path / "framework")]:
        executable = prefix / "bin/brainlayer"
        if name == "shadow" and user_owned:
            executable = tmp_path / ".local/bin/brainlayer"
        executable.parent.mkdir(parents=True)
        executable.write_text(f"#!{prefix}/bin/python3.13 -E\n")
        executable.chmod(0o755)
        paths[name] = executable
    metadata = tmp_path / "framework/lib/python3.13/site-packages/brainlayer-1.0.dist-info/direct_url.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"dir_info": {"editable": %s}}' % str(editable).lower())
    monkeypatch.setenv("PATH", ":".join(str(paths[name].parent) for name in order.split(",")))
    result = doctor.DoctorResult("", True, 0)
    doctor._check_cli_path_shadow(result)
    assert result.cli_path_shadow["state"] == (severity or "pass")
    if severity:
        issue = result.issues[0]
        assert issue.code == "cli_path_shadow"
        assert issue.severity == severity
        assert str(paths["shadow"]) in issue.message
        assert ("editable install" if editable else "plain script") in issue.message
        env = "PYTHONNOUSERSITE=1 " if user_site else ""
        expected = f"sudo {env}{tmp_path}/framework/bin/pip3 uninstall brainlayer"
        if user_owned:
            expected = f"{tmp_path}/framework/bin/python3.13 -m pip uninstall brainlayer"
        assert issue.details["remediation"] == expected
    else:
        assert not result.issues


def test_cli_path_aliases_and_skip(tmp_path, monkeypatch):
    keg = tmp_path / "Cellar/brainlayer/1.5.45"
    executable = keg / "bin/brainlayer"
    executable.parent.mkdir(parents=True)
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o755)
    alias = tmp_path / "bin"
    alias.mkdir()
    (alias / "brainlayer").symlink_to(executable)
    monkeypatch.setenv("PATH", f"{alias}:{executable.parent}")
    monkeypatch.setattr(doctor.sys, "prefix", str(keg / "libexec/venv"))
    result = doctor.DoctorResult("", True, 0)
    doctor._check_cli_path_shadow(result)
    assert result.cli_path_shadow["state"] == "pass"
    assert len(result.cli_path_shadow["paths"]) == 1
    executable.chmod(0o644)
    doctor._check_cli_path_shadow(result)
    assert result.cli_path_shadow["state"] == "warning"
    assert result.issues[-1].severity == "warning"
    monkeypatch.setattr(doctor.sys, "prefix", str(tmp_path / "dev/venv"))
    result = doctor.DoctorResult("", True, 0)
    doctor._check_cli_path_shadow(result)
    assert result.cli_path_shadow["state"] == "skipped"
    assert result.cli_path_shadow["reason"]


def test_cli_path_unsearchable_entry_and_opt_prefix(tmp_path, monkeypatch):
    keg = tmp_path / "Cellar/brainlayer/1.5.45"
    script = keg / "bin/brainlayer"
    script.parent.mkdir(parents=True)
    script.write_text("#!/bin/sh\n")
    script.chmod(0o755)
    opt = tmp_path / "opt/brainlayer"
    opt.parent.mkdir()
    opt.symlink_to(keg, target_is_directory=True)
    denied = tmp_path / "denied"
    denied.mkdir(mode=0o000)
    monkeypatch.setattr(doctor.sys, "prefix", str(opt / "libexec/venv"))
    monkeypatch.setenv("PATH", f"{denied}:{script.parent}")
    original = type(script).is_file

    def probe(path):
        if path == denied / "brainlayer":
            raise PermissionError("unsearchable PATH directory")
        return original(path)

    monkeypatch.setattr(type(script), "is_file", probe)
    result = doctor.DoctorResult("", True, 0)
    try:
        doctor._check_cli_path_shadow(result)
    finally:
        denied.chmod(0o700)
    assert result.cli_path_shadow["paths"] == [str(script)]
    assert result.cli_path_shadow["prefix"] == str(keg)
    assert result.issues[0].severity == "warning"
    assert "unsearchable PATH directory" in result.issues[0].message


def test_cli_path_only_first_shadow_fatal_and_user_version(tmp_path, monkeypatch):
    paths = [tmp_path / name / "bin" for name in ("first", "later")]
    for directory in paths:
        directory.mkdir(parents=True)
        script = directory / "brainlayer"
        script.write_text(f"#!{directory}/python3.13 -E\n")
        script.chmod(0o755)
    (tmp_path / "Library/Python/3.12/lib/python/site-packages/brainlayer-1.0.dist-info").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("PATH", ":".join(map(str, paths)))
    monkeypatch.setattr(doctor.sys, "prefix", str(tmp_path / "Cellar/brainlayer/1.5.45/libexec/venv"))
    result = doctor.DoctorResult("", True, 0)
    doctor._check_cli_path_shadow(result)
    assert [i.severity for i in result.issues] == ["fatal", "warning"]
    assert all("PYTHONNOUSERSITE" not in i.details["remediation"] for i in result.issues)
    assert all("pip3 uninstall brainlayer" in i.details["remediation"] for i in result.issues)

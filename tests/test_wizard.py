import pytest

from brainlayer.cli import wizard
from brainlayer.cli.wizard import WizardConfig, detect_environment


def test_detect_claude_code_conversations():
    env = detect_environment()
    assert "claude_projects_dir" in env
    assert isinstance(env["conversation_count"], int)


def test_wizard_config_defaults():
    config = WizardConfig()
    assert not hasattr(config, "enrich_backend")
    assert isinstance(config.extras, list)


def test_default_gemini_env_file_location(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    assert wizard.get_default_env_file() == tmp_path / ".config" / "brainlayer" / "brainlayer.env"


def test_write_gemini_env_file_rejects_plaintext_google_key(tmp_path):
    env_path = tmp_path / "brainlayer.env"

    with pytest.raises(ValueError, match="1Password"):
        wizard.write_gemini_env_file(env_path, google_api_key="test-secret", secret_source="plain")

    assert not env_path.exists()


def test_write_gemini_env_file_refuses_to_overwrite_existing_key_without_confirmation(tmp_path):
    env_path = tmp_path / "brainlayer.env"
    env_path.write_text("GOOGLE_API_KEY=existing\nBRAINLAYER_ENRICH_RATE=5\n", encoding="utf-8")

    with pytest.raises(FileExistsError):
        wizard.write_gemini_env_file(
            env_path, google_api_key="op://Private/Google AI/Gemini API key", secret_source="1password", overwrite=False
        )

    assert "GOOGLE_API_KEY=existing" in env_path.read_text(encoding="utf-8")


def test_write_gemini_env_file_preserves_existing_config_values_on_key_update(tmp_path):
    env_path = tmp_path / "brainlayer.env"
    env_path.write_text(
        "\n".join(
            [
                "# keep this comment",
                "GOOGLE_API_KEY=existing",
                "BRAINLAYER_ENRICH_RATE=7",
                "export BRAINLAYER_LAUNCHD_DRAIN_ENABLED=0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    wizard.write_gemini_env_file(
        env_path,
        google_api_key="op://Private/Google AI/Gemini API key",
        secret_source="1password",
        overwrite=True,
    )

    content = env_path.read_text(encoding="utf-8")
    assert "GOOGLE_API_KEY=existing" not in content
    assert "GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\"" in content
    assert "# keep this comment" in content
    assert "BRAINLAYER_ENRICH_RATE=7" in content
    assert "BRAINLAYER_ENRICH_RATE=15" not in content
    assert "export BRAINLAYER_LAUNCHD_DRAIN_ENABLED=0" in content
    assert "BRAINLAYER_LAUNCHD_DRAIN_ENABLED=1" not in content
    assert "BRAINLAYER_ENRICH_CONCURRENCY" not in content


def test_write_gemini_env_file_can_source_google_key_from_1password_reference(tmp_path):
    env_path = tmp_path / "brainlayer.env"

    wizard.write_gemini_env_file(
        env_path,
        google_api_key="op://Private/Google AI/Gemini API key",
        secret_source="1password",
    )

    content = env_path.read_text(encoding="utf-8")
    assert "GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\"" in content


@pytest.fixture(autouse=True)
def isolated_wizard_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(wizard, "get_db_path", lambda: tmp_path / "fixture.db")


def test_new_env_defaults_offer_only_local_services(tmp_path):
    env_path = tmp_path / "brainlayer.env"
    wizard.write_gemini_env_file(
        env_path, google_api_key="op://Private/Google AI/Gemini API key", secret_source="1password"
    )
    text = env_path.read_text()
    assert "GOOGLE_API_KEY=" in text
    assert "BRAINLAYER_LAUNCHD_WATCH_ENABLED=1" in text
    assert "BRAINLAYER_LAUNCHD_DRAIN_ENABLED=1" in text
    assert "ENRICH" not in text
    assert "BRAINLAYER_GEMINI_SERVICE_TIER" not in text
    assert "BRAINLAYER_MAX_COMMIT_BATCH" not in text


def test_wizard_offers_local_extras_without_model_or_enrichment_setup(monkeypatch, tmp_path, capsys):
    from rich.prompt import Confirm, Prompt

    env = {
        "conversation_count": 3,
        "existing_db": False,
        "claude_projects_dir": tmp_path,
        "gemini_env_file": tmp_path / "brainlayer.env",
        "gemini_env_file_has_key": False,
        "op_available": False,
        "ollama_available": True,
        "is_apple_silicon": True,
    }
    monkeypatch.setattr(wizard, "detect_environment", lambda: env)
    questions = []
    monkeypatch.setattr(Confirm, "ask", lambda text, **kw: questions.append(text) or False)
    monkeypatch.setattr(Prompt, "ask", lambda *a, **kw: pytest.fail("unexpected model/backend prompt"))
    config = wizard.run_wizard()
    assert config.claude_projects_dir == tmp_path
    assert len(questions) == 3
    assert "enrichment" not in capsys.readouterr().out.lower()


def test_environment_detection_does_not_probe_model_runtime(monkeypatch):
    import shutil
    import subprocess

    monkeypatch.setattr(shutil, "which", lambda command: "/bin/true")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("model runtime probe"))
    assert isinstance(wizard.detect_environment()["conversation_count"], int)


@pytest.mark.parametrize("key_name", ["GOOGLE_API_KEY", "GOOGLE_GENERATIVE_AI_API_KEY"])
def test_google_key_alias_is_preserved_until_explicit_replacement(tmp_path, key_name):
    env_path = tmp_path / "brainlayer.env"
    original = f"{key_name}=legacy-fixture\nBRAINLAYER_ENRICH_RATE=7\n"
    env_path.write_text(original)
    assert wizard.env_file_has_google_key(env_path)
    with pytest.raises(FileExistsError):
        wizard.write_gemini_env_file(
            env_path, google_api_key="op://Private/Google AI/Gemini API key", secret_source="1password"
        )
    assert env_path.read_text() == original


def test_setup_creates_local_defaults_and_keeps_existing_file(tmp_path):
    from brainlayer.setup import ensure_brainlayer_env

    env_path = tmp_path / "brainlayer.env"
    assert ensure_brainlayer_env(env_path) == env_path
    text = env_path.read_text()
    assert "ENRICH" not in text and "cloud enrichment" not in text
    assert "--google-api-key-op-ref" in text
    assert env_path.stat().st_mode & 0o777 == 0o600
    original = "GOOGLE_GENERATIVE_AI_API_KEY=fixture\nBRAINLAYER_ENRICH_RATE=7\n"
    env_path.write_text(original)
    ensure_brainlayer_env(env_path)
    assert env_path.read_text() == original

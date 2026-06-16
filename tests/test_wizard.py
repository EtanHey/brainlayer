import pytest

from brainlayer.cli import wizard
from brainlayer.cli.wizard import WizardConfig, detect_environment


def test_detect_ollama_running():
    env = detect_environment()
    assert "ollama_available" in env
    assert isinstance(env["ollama_available"], bool)


def test_detect_claude_code_conversations():
    env = detect_environment()
    assert "claude_projects_dir" in env
    assert isinstance(env["conversation_count"], int)


def test_detect_apple_silicon():
    env = detect_environment()
    assert "is_apple_silicon" in env
    assert isinstance(env["is_apple_silicon"], bool)


def test_wizard_config_defaults():
    config = WizardConfig()
    assert config.enrich_backend in ("ollama", "mlx", "none")
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
    assert "BRAINLAYER_ENRICH_CONCURRENCY=4" in content


def test_write_gemini_env_file_can_source_google_key_from_1password_reference(tmp_path):
    env_path = tmp_path / "brainlayer.env"

    wizard.write_gemini_env_file(
        env_path,
        google_api_key="op://Private/Google AI/Gemini API key",
        secret_source="1password",
    )

    content = env_path.read_text(encoding="utf-8")
    assert "GOOGLE_API_KEY=\"$(op read 'op://Private/Google AI/Gemini API key')\"" in content

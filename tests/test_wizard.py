from brainlayer.cli.wizard import detect_environment, WizardConfig


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

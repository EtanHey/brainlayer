"""Shared Groq model selection and availability checks."""

from typing import Any

import requests

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"


class GroqModelUnavailableError(RuntimeError):
    """The configured Groq model cannot serve requests."""


class GroqServiceUnavailableError(RuntimeError):
    """Groq's catalog could not be queried, so model status is unknown."""


def _models_url(completions_url: str) -> str:
    prefix, separator, _ = completions_url.rstrip("/").rpartition("/chat/completions")
    if separator:
        return f"{prefix}/models"
    return "https://api.groq.com/openai/v1/models"


def raise_for_groq_response(response: Any, model: str) -> None:
    """Raise a model-specific error only when Groq identifies model_not_found."""
    try:
        error = response.json().get("error", {})
    except (AttributeError, TypeError, ValueError):
        error = {}
    if response.status_code == 404 and isinstance(error, dict) and error.get("code") == "model_not_found":
        raise GroqModelUnavailableError(f"Groq model {model!r} is unavailable (HTTP 404 model_not_found)")
    response.raise_for_status()


def validate_groq_model(api_key: str, model: str, completions_url: str, timeout: int = 5) -> None:
    """Refuse startup when the configured model is absent from Groq's catalog."""
    try:
        response = requests.get(
            _models_url(completions_url),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        catalog = response.json()
        if not isinstance(catalog, dict) or not isinstance(catalog.get("data"), list):
            raise TypeError("Groq /models response is not an object with a data list")
        model_ids = set()
        for item in catalog["data"]:
            if not isinstance(item, dict) or not isinstance(item.get("id"), str):
                raise TypeError("Groq /models response contains an entry without a string id")
            model_ids.add(item["id"])
    except (requests.RequestException, KeyError, TypeError, ValueError) as exc:
        raise GroqServiceUnavailableError(
            f"Groq model {model!r} could not be checked because the Groq service is unavailable "
            "or rejected the /models request"
        ) from exc
    if model not in model_ids:
        raise GroqModelUnavailableError(f"Groq model {model!r} is unavailable: not listed by the /models endpoint")

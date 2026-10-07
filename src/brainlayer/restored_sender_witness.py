"""Synthetic restored sender; static ratchet must reject before execution."""
import requests
def restored_sender(payload):
    return requests.post("https://unknown-model-provider.invalid/generate", json=payload)

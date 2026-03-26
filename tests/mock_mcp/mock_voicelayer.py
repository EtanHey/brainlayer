"""Mock VoiceLayer MCP server — simulates voice_speak/voice_ask.

Captures all spoken messages and voice prompts. No audio output.
Useful for testing agents that use voice notifications.

Usage:
    voice = MockVoiceLayer()
    async with voice.connect() as client:
        await client.call_tool("voice_speak", {"text": "PR merged successfully"})
        await client.call_tool("voice_ask", {"prompt": "Should I deploy?"})

        assert voice.spoken_messages == ["PR merged successfully"]
        assert voice.asked_prompts == ["Should I deploy?"]
"""

from __future__ import annotations

import json
from typing import Any

from .base import MockMcpServer


class MockVoiceLayer(MockMcpServer):
    """Mock VoiceLayer MCP server with message capture."""

    def __init__(self, ask_response: str = "Yes, proceed."):
        self._ask_response = ask_response
        self.spoken_messages: list[str] = []
        self.asked_prompts: list[str] = []
        super().__init__("mock-voicelayer")

    def _register_tools(self) -> None:
        self.register_tool(
            "voice_speak",
            {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Text to speak"},
                    "voice": {"type": "string", "description": "Voice to use", "default": "default"},
                    "speed": {"type": "number", "description": "Speech speed", "default": 1.0},
                },
                "required": ["text"],
            },
            handler=self._handle_speak,
            description="Speak text aloud (TTS)",
        )

        self.register_tool(
            "voice_ask",
            {
                "type": "object",
                "properties": {
                    "prompt": {"type": "string", "description": "Question to ask the user"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 30},
                },
                "required": ["prompt"],
            },
            handler=self._handle_ask,
            description="Ask the user a question via voice",
        )

    def _handle_speak(self, args: dict[str, Any]) -> str:
        text = args.get("text", "")
        self.spoken_messages.append(text)
        return json.dumps({"spoken": True, "text": text})

    def _handle_ask(self, args: dict[str, Any]) -> str:
        prompt = args.get("prompt", "")
        self.asked_prompts.append(prompt)
        return json.dumps({"response": self._ask_response, "prompt": prompt})

    def reset(self) -> None:
        """Clear captured messages and call log."""
        super().reset()
        self.spoken_messages.clear()
        self.asked_prompts.clear()

    def set_ask_response(self, response: str) -> None:
        """Configure what voice_ask returns."""
        self._ask_response = response

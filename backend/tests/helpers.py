"""
Shared test helpers: mock Anthropic response builders.
Importable from any test file.
"""
from unittest.mock import MagicMock


def make_tool_use_response(tool_name: str, tool_input: dict, tool_id: str = "tu_abc123"):
    """Build a mock Anthropic response with stop_reason='tool_use'."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.id = tool_id
    tool_block.name = tool_name
    tool_block.input = tool_input

    response = MagicMock()
    response.stop_reason = "tool_use"
    response.content = [tool_block]
    return response


def make_text_response(text: str):
    """Build a mock Anthropic response with stop_reason='end_turn'."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.stop_reason = "end_turn"
    response.content = [text_block]
    return response

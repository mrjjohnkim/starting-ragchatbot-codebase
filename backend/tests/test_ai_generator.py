"""
Tests for AIGenerator.generate_response() and _handle_tool_execution().

Focus areas:
1. Tools are forwarded to the Anthropic API when provided
2. When stop_reason == 'tool_use', _handle_tool_execution is invoked and the
   tool manager's execute_tool() is called with the correct arguments
3. The tool result is appended as a 'user' message before the follow-up call
4. The final text response is returned (not a tool block)
5. Direct (non-tool) responses are returned immediately
"""
import pytest
from unittest.mock import MagicMock, patch, call

# conftest.py adds backend/ to sys.path
from ai_generator import AIGenerator
from helpers import make_tool_use_response, make_text_response


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_generator(mock_client):
    """Return an AIGenerator whose internal Anthropic client is replaced."""
    with patch("ai_generator.anthropic.Anthropic", return_value=mock_client):
        gen = AIGenerator(api_key="fake-key", model="claude-test-model")
    return gen


# ─── Direct response (no tool use) ────────────────────────────────────────────

class TestDirectResponse:

    def test_returns_text_when_no_tool_use(self, mock_anthropic_client):
        """generate_response() should return content[0].text on end_turn."""
        mock_anthropic_client.messages.create.return_value = make_text_response(
            "Python is a language."
        )
        gen = _make_generator(mock_anthropic_client)
        result = gen.generate_response(query="What is Python?")

        assert result == "Python is a language."

    def test_does_not_call_tool_manager_on_direct_response(self, mock_anthropic_client):
        """No tool_manager calls should happen when stop_reason is end_turn."""
        mock_anthropic_client.messages.create.return_value = make_text_response("OK")
        gen = _make_generator(mock_anthropic_client)
        tool_manager = MagicMock()

        gen.generate_response(query="General question", tool_manager=tool_manager)
        tool_manager.execute_tool.assert_not_called()

    def test_passes_system_prompt_to_api(self, mock_anthropic_client):
        """The SYSTEM_PROMPT must be included in the API call."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        gen = _make_generator(mock_anthropic_client)
        gen.generate_response(query="Hello")

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "system" in call_kwargs
        assert len(call_kwargs["system"]) > 0


# ─── Tools are forwarded to the API ───────────────────────────────────────────

class TestToolForwarding:

    def test_tools_included_in_api_call_when_provided(self, mock_anthropic_client):
        """When tools are passed, they must appear in the API call parameters."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        gen = _make_generator(mock_anthropic_client)

        fake_tools = [{"name": "search_course_content", "description": "...", "input_schema": {}}]
        gen.generate_response(query="test", tools=fake_tools)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "tools" in call_kwargs
        assert call_kwargs["tools"] == fake_tools

    def test_tool_choice_set_to_auto_when_tools_provided(self, mock_anthropic_client):
        """tool_choice must be {'type': 'auto'} whenever tools are supplied."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        gen = _make_generator(mock_anthropic_client)

        gen.generate_response(query="test", tools=[{"name": "t"}])

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert call_kwargs.get("tool_choice") == {"type": "auto"}

    def test_no_tools_key_when_tools_is_none(self, mock_anthropic_client):
        """When tools=None, the 'tools' key must not appear in the API params."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        gen = _make_generator(mock_anthropic_client)
        gen.generate_response(query="test", tools=None)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "tools" not in call_kwargs


# ─── Tool execution flow ───────────────────────────────────────────────────────

class TestToolExecution:

    def _setup_tool_call(self, mock_anthropic_client, tool_name, tool_input,
                         tool_result_text, final_answer):
        """Wire the mock client to return a tool_use then a text response."""
        tool_response = make_tool_use_response(tool_name, tool_input, tool_id="tu_001")
        text_response = make_text_response(final_answer)
        mock_anthropic_client.messages.create.side_effect = [tool_response, text_response]

    def test_execute_tool_is_called_when_stop_reason_is_tool_use(self, mock_anthropic_client):
        """When Claude returns stop_reason='tool_use', execute_tool must be called."""
        self._setup_tool_call(
            mock_anthropic_client,
            tool_name="search_course_content",
            tool_input={"query": "neural networks"},
            tool_result_text="[Course] Neural networks content...",
            final_answer="Neural networks are...",
        )
        gen = _make_generator(mock_anthropic_client)
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "[Course] Neural networks content..."

        result = gen.generate_response(
            query="What are neural networks?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="neural networks"
        )

    def test_returns_final_text_after_tool_execution(self, mock_anthropic_client):
        """The string returned after tool execution must be the final text block."""
        self._setup_tool_call(
            mock_anthropic_client,
            tool_name="search_course_content",
            tool_input={"query": "activation functions"},
            tool_result_text="ReLU is...",
            final_answer="Activation functions transform layer outputs.",
        )
        gen = _make_generator(mock_anthropic_client)
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "ReLU is..."

        result = gen.generate_response(
            query="What are activation functions?",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_manager,
        )

        assert result == "Activation functions transform layer outputs."

    def test_api_called_twice_during_tool_execution(self, mock_anthropic_client):
        """There must be exactly two API calls: initial + follow-up."""
        self._setup_tool_call(
            mock_anthropic_client,
            tool_name="search_course_content",
            tool_input={"query": "q"},
            tool_result_text="result",
            final_answer="answer",
        )
        gen = _make_generator(mock_anthropic_client)
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "result"

        gen.generate_response(
            query="q", tools=[{"name": "search_course_content"}], tool_manager=tool_manager
        )

        assert mock_anthropic_client.messages.create.call_count == 2

    def test_tool_result_sent_as_user_message_in_follow_up(self, mock_anthropic_client):
        """The follow-up API call must contain a user message with the tool result."""
        self._setup_tool_call(
            mock_anthropic_client,
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_result_text="some content",
            final_answer="final",
        )
        gen = _make_generator(mock_anthropic_client)
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "some content"

        gen.generate_response(
            query="test", tools=[{"name": "x"}], tool_manager=tool_manager
        )

        # Second call is the follow-up
        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
        messages = second_call_kwargs["messages"]

        # Last message must be the user message with tool results
        last_msg = messages[-1]
        assert last_msg["role"] == "user"

        # Content must be a list of tool_result dicts
        assert isinstance(last_msg["content"], list)
        tool_result_content = last_msg["content"][0]
        assert tool_result_content["type"] == "tool_result"
        assert tool_result_content["tool_use_id"] == "tu_001"
        assert tool_result_content["content"] == "some content"

    def test_follow_up_call_does_not_include_tools(self, mock_anthropic_client):
        """The second (follow-up) API call must NOT include tools (prevents re-use)."""
        self._setup_tool_call(
            mock_anthropic_client,
            tool_name="search_course_content",
            tool_input={"query": "test"},
            tool_result_text="content",
            final_answer="final",
        )
        gen = _make_generator(mock_anthropic_client)
        tool_manager = MagicMock()
        tool_manager.execute_tool.return_value = "content"

        gen.generate_response(
            query="test", tools=[{"name": "x"}], tool_manager=tool_manager
        )

        second_call_kwargs = mock_anthropic_client.messages.create.call_args_list[1].kwargs
        assert "tools" not in second_call_kwargs

    def test_generate_response_without_tool_manager_ignores_tool_use(self, mock_anthropic_client):
        """
        If stop_reason is 'tool_use' but no tool_manager is given,
        generate_response() should fall through to content[0].text.
        This tests an edge case in the current code.
        """
        tool_response = make_tool_use_response(
            "search_course_content", {"query": "x"}, tool_id="tu_002"
        )
        # Add a .text attribute so the fallback .content[0].text doesn't crash
        tool_response.content[0].text = "fallback text"
        mock_anthropic_client.messages.create.return_value = tool_response

        gen = _make_generator(mock_anthropic_client)

        # With no tool_manager the code takes the else branch
        result = gen.generate_response(query="test", tools=[{"name": "t"}], tool_manager=None)
        # Should return content[0].text — even if the block is a ToolUseBlock
        assert result == "fallback text"


# ─── Conversation history ──────────────────────────────────────────────────────

class TestConversationHistory:

    def test_history_appended_to_system_prompt_when_provided(self, mock_anthropic_client):
        """Conversation history should be appended to the system content."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        gen = _make_generator(mock_anthropic_client)
        gen.generate_response(query="follow-up", conversation_history="User: hi\nAssistant: hello")

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "Previous conversation" in call_kwargs["system"]
        assert "User: hi" in call_kwargs["system"]

    def test_no_history_in_system_prompt_when_none(self, mock_anthropic_client):
        """When conversation_history is None, system must not contain 'Previous conversation'."""
        mock_anthropic_client.messages.create.return_value = make_text_response("ok")
        gen = _make_generator(mock_anthropic_client)
        gen.generate_response(query="fresh question", conversation_history=None)

        call_kwargs = mock_anthropic_client.messages.create.call_args.kwargs
        assert "Previous conversation" not in call_kwargs["system"]

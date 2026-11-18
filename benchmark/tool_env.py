"""Attach local retrieval tools to Tinker's tool-use environment (no training launch).

Requires a compatible tinker-cookbook. The caller supplies a renderer for its
model and a trusted reward callback; the plain rubric recipe cannot run tools.
"""
import json
from pathlib import Path
from typing import Annotated

from tinker_cookbook.tool_use import ToolResult, simple_tool_result, tool, build_agent_tool_env
from manual_tools import ManualTools


class TinkerManualTools:
    def __init__(self, corpus_path: Path):
        self.manuals = ManualTools(corpus_path)

    @tool
    async def search_manual(
        self,
        query: Annotated[str, "Words describing the underwriting rule or definition to find"],
        source_id: Annotated[str, "Assigned source ID: tx, ma, allianz, or fema"],
    ) -> ToolResult:
        """Search a frozen source and return passages with stable section IDs."""
        try:
            result = self.manuals.search_manual(query, source_id)
        except ValueError as error:
            result = {"error": str(error)}
        return simple_tool_result(json.dumps(result, ensure_ascii=False))

    @tool
    async def read_section(
        self,
        source_id: Annotated[str, "Assigned source ID"],
        section_id: Annotated[str, "Exact section_id returned by search_manual"],
    ) -> ToolResult:
        """Read the complete source section to check context and exceptions."""
        try:
            result = self.manuals.read_section(source_id, section_id)
        except ValueError as error:
            result = {"error": str(error)}
        return simple_tool_result(json.dumps(result, ensure_ascii=False))


def make_env(record: dict, renderer, reward_fn, corpus_path: Path | None = None, max_turns: int = 8):
    """One fixed-prefix tool episode; not the native three-user-turn benchmark.

    reward_fn receives the actual history including tool results. Grade its final
    answer with record['rubric_items']; never show those items to the policy.
    The caller must retain histories and use independent grading credentials.
    """
    tools = TinkerManualTools(corpus_path or Path(__file__).with_name("corpus.json"))
    functions = [tools.search_manual, tools.read_section]
    convo = record["convo"]
    if not convo or convo[0]["role"] != "system":
        raise ValueError("Expected an exported benchmark prefix with a system message")
    prefix = renderer.create_conversation_prefix_with_tools(
        tools=[function.to_spec() for function in functions],
        system_prompt=convo[0]["content"],
    )
    return build_agent_tool_env(
        renderer=renderer, tools=functions, initial_messages=prefix + convo[1:],
        reward_fn=reward_fn, max_turns=max_turns, max_trajectory_tokens=32768,
    )

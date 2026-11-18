#!/usr/bin/env python3
"""Run the frozen underwriting conversation set through Tinker BenchmarkBuilder."""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import importlib.metadata
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
RECORD_DIR = REPO_ROOT / "study findings" / "records" / "inkling-scored-20261115"
PROTOCOL_PATH = RECORD_DIR / "PROTOCOL.md"
sys.path.insert(0, str(ROOT))

import build  # noqa: E402
import grade  # noqa: E402
import run_eval  # noqa: E402
from tinker_cookbook.eval.benchmarks import (  # noqa: E402
    BenchmarkBuilder,
    BenchmarkConfig,
    REGISTRY,
    register,
    run_benchmark,
)
from tinker_cookbook.renderers import get_text_content  # noqa: E402
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult  # noqa: E402


BENCHMARK_NAME = "inkling_underwriting_3turn"
MAX_BUDGET_USD = 10.0
CANDIDATE_INPUT_CONTEXT_CAP = 65536
MAX_TRAJECTORY_TOKENS = CANDIDATE_INPUT_CONTEXT_CAP + run_eval.OUTPUT_TOKEN_CAP


class FatalPaidRequestAbort(BaseException):
    """Stop run_benchmark before any later case can submit a paid request."""


def _sample_parameters(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return {
        name: getattr(value, name)
        for name in ("max_tokens", "temperature", "seed", "stop")
        if hasattr(value, name)
    }


class CandidateSamplingProxy:
    """Route framework candidate samples through the durable request ledger."""

    def __init__(self, sample_fn: Callable[..., Any], renderer: Any, abort_state: dict[str, str]):
        self.sample_fn = sample_fn
        self.renderer = renderer
        self.abort_state = abort_state
        self.case_id = ""
        self.turn_index = 1
        self.call_index = 1

    def set_context(self, case_id: str, turn_index: int, call_index: int) -> None:
        self.case_id, self.turn_index, self.call_index = case_id, turn_index, call_index

    async def sample(self, *, prompt: Any, num_samples: int, sampling_params: Any) -> Any:
        if self.abort_state:
            raise FatalPaidRequestAbort(next(iter(self.abort_state.values())))
        params = _sample_parameters(sampling_params)
        if (
            num_samples != 1
            or params.get("max_tokens") != run_eval.OUTPUT_TOKEN_CAP
            or float(params.get("temperature", -1)) != run_eval.CANDIDATE_TEMP
            or run_eval.jsonable(params.get("stop")) != run_eval.jsonable(self.renderer.get_stop_sequences())
        ):
            raise run_eval.EvaluationError("Framework changed the frozen candidate sampling parameters")
        try:
            response = await asyncio.to_thread(
                self.sample_fn,
                role="candidate",
                case_id=self.case_id,
                turn_index=self.turn_index,
                call_index=self.call_index,
                prompt=prompt,
            )
        except asyncio.CancelledError as exc:
            reason = "candidate sampling was cancelled; request status may be uncertain"
            self.abort_state["reason"] = reason
            raise FatalPaidRequestAbort(reason) from exc
        if self.abort_state:
            raise FatalPaidRequestAbort(next(iter(self.abort_state.values())))
        if not response.sequences or str(response.sequences[0].stop_reason) != "stop":
            raise run_eval.EvaluationError("Candidate response hit the output limit or returned no sample")
        return response


async def _execute_tool(tool: Any, tool_call: Any, assigned_source_id: str) -> list[Any]:
    del assigned_source_id
    from tinker_cookbook.tool_use.tools import handle_tool_call

    return (await handle_tool_call({tool.name: tool}, tool_call)).messages


class UnderwritingMessageEnv(MessageEnv):
    """Reveal and grade one user turn at a time while retaining actual history."""

    def __init__(
        self,
        *,
        case: dict,
        corpus: dict,
        candidate_renderer: Any,
        judge_renderer: Any,
        tools: list[Any],
        sample_proxy: CandidateSamplingProxy,
        judge_client: Any,
        sample_fn: Callable[..., Any],
        tool_executor: Callable[..., Any] = _execute_tool,
        save_dir: Path,
        turn_grader: Callable[..., Any] | None = None,
    ):
        if len(case["turns"]) != 3:
            raise run_eval.EvaluationError(f"{case['id']} must contain exactly three user turns")
        self.turn_grader = turn_grader
        self.case = case
        self.corpus = corpus
        self.candidate_renderer = candidate_renderer
        self.judge_renderer = judge_renderer
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.sample_proxy = sample_proxy
        self.judge_client = judge_client
        self.sample_fn = sample_fn
        self.tool_executor = tool_executor
        self.save_dir = save_dir
        self.messages: list[dict[str, Any]] = []
        self.actual_answers: list[str] = []
        self.turn_results: list[dict[str, Any]] = []
        self.turn_index = 1
        self.sample_calls = 0
        self.tool_calls = 0
        self.final_answer_only = False
        self.example_id = case["id"]

    async def initial_observation(self) -> list[dict[str, Any]]:
        if self.sample_proxy.abort_state:
            raise FatalPaidRequestAbort(next(iter(self.sample_proxy.abort_state.values())))
        specs = [tool.to_spec() for tool in self.tools]
        self.messages = self.candidate_renderer.create_conversation_prefix_with_tools(
            tools=specs, system_prompt=run_eval.make_candidate_system_prompt(self.case)
        )
        self.messages.append({"role": "user", "content": self.case["turns"][0]["user"]})
        self.sample_proxy.set_context(self.case["id"], 1, 1)
        return self.messages

    async def step(self, message: dict[str, Any]) -> MessageStepResult:
        self.messages.append(message)
        self.sample_calls += 1
        if message.get("unparsed_tool_calls"):
            error = "Candidate response contained malformed tool calls"
            run_eval.save_json(self.save_dir / "turns" / f"{self.case['id']}-turn-{self.turn_index}.json", {
                "case_id": self.case["id"], "turn_index": self.turn_index,
                "status": "candidate_parse_error", "error": error,
                "candidate_sample_calls": self.sample_calls,
                "actual_prior_answers": list(self.actual_answers),
                "actual_message_history": run_eval.jsonable(self.messages),
            })
            raise run_eval.EvaluationError(error)
        tool_calls = list(message.get("tool_calls") or [])
        if tool_calls and self.final_answer_only:
            # Do not execute more tools or sample again after the reserved answer attempt.
            self.messages.extend(run_eval._blocked_tool_message(call, run_eval._tool_call_name_args(call)[0]) for call in tool_calls)
            answer = ""
        elif tool_calls:
            for tool_call in tool_calls:
                name, args = run_eval._tool_call_name_args(tool_call)
                if self.tool_calls >= run_eval.TOOL_CALLS_PER_TURN:
                    results = [run_eval._blocked_tool_message(tool_call, name)]
                elif args.get("source_id") != self.case["source_id"] or name not in self.tool_map:
                    results = [run_eval._blocked_tool_message(tool_call, name)]
                    self.tool_calls += 1
                else:
                    results = await self.tool_executor(self.tool_map[name], tool_call, self.case["source_id"])
                    self.tool_calls += 1
                self.messages.extend(results)

            if self.sample_calls >= run_eval.SAMPLE_CALLS_PER_TURN or self.tool_calls >= run_eval.TOOL_CALLS_PER_TURN:
                self.final_answer_only = True
                self.messages.append({"role": "user", "content": (
                    "Environment notice: retrieval is now closed for this question. "
                    "You have one final-answer response remaining. Do not call tools. "
                    "Answer the current question using the evidence already returned, cite each section ID "
                    "in its own brackets, and state any missing evidence or uncertainty."
                )})
            self.sample_proxy.set_context(self.case["id"], self.turn_index, self.sample_calls + 1)
            return MessageStepResult(0.0, False, self.messages, logs={
                "case_id": self.case["id"], "turn_index": self.turn_index,
            })
        else:
            answer = get_text_content(message)

        partial_record = {
            "case_id": self.case["id"], "turn_index": self.turn_index,
            "user": self.case["turns"][self.turn_index - 1]["user"],
            "answer": answer, "tool_calls": self.tool_calls,
            "candidate_sample_calls": self.sample_calls, "judge_call": "pending",
            "reserved_final_answer_used": self.final_answer_only,
            "final_answer_tool_violation": bool(self.final_answer_only and tool_calls),
            "actual_prior_answers": list(self.actual_answers),
            "actual_message_history": run_eval.jsonable(self.messages), "status": "pending_grade",
        }
        turn_path = self.save_dir / "turns" / f"{self.case['id']}-turn-{self.turn_index}.json"
        run_eval.save_json(turn_path, partial_record)
        turn_record = await self._grade_turn(answer, partial_record)
        self.turn_results.append(turn_record)
        self.actual_answers.append(answer)
        run_eval.save_json(turn_path, turn_record)
        this_turn = self.turn_index
        self.turn_index += 1
        self.sample_calls = self.tool_calls = 0
        self.final_answer_only = False

        if self.turn_index <= 3:
            self.messages.append({"role": "user", "content": self.case["turns"][this_turn]["user"]})
            self.sample_proxy.set_context(self.case["id"], self.turn_index, 1)
            return MessageStepResult(
                0.0, False, self.messages,
                metrics={"turn_passed": float(turn_record["verdict"].get("passed") is True)},
                logs={"case_id": self.case["id"], "turn_index": this_turn,
                      "turn_verdict_json": json.dumps(turn_record["verdict"], ensure_ascii=False)},
            )

        passed = len(self.turn_results) == 3 and all(row["verdict"].get("passed") is True for row in self.turn_results)
        unresolved = any(row["verdict"].get("passed") is None for row in self.turn_results)
        case_record = {"case_id": self.case["id"], "status": "unresolved" if unresolved else "completed", "passed": None if unresolved else passed, "turns": self.turn_results}
        run_eval.save_json(self.save_dir / "cases" / f"{self.case['id']}.json", case_record)
        return MessageStepResult(
            float(passed), True, self.messages,
            metrics={"conversation_passed": float(passed),
                     "turn_passed": float(turn_record["verdict"].get("passed") is True)},
            logs={"case_id": self.case["id"], "turn_index": 3,
                  "conversation_passed": float(passed),
                  "turn_verdict_json": json.dumps(turn_record["verdict"], ensure_ascii=False)},
        )

    async def _grade_turn(self, answer: str, turn_record: dict[str, Any]) -> dict[str, Any]:
        if self.turn_grader is not None:
            verdict = await asyncio.to_thread(self.turn_grader, self.case, self.turn_index, list(self.actual_answers), answer)
            return {**turn_record, "judge_call": "completed", "verdict": verdict, "status": verdict["status"]}
        turn = self.case["turns"][self.turn_index - 1]
        overlay = build.overlay(self.case, self.turn_index)
        evidence, citation_error = grade.citation_evidence(answer, self.corpus, self.case["source_id"])
        discarded_quotes: list[dict[str, Any]] = []
        judge_call = "not_called"
        if citation_error:
            verdict = {"reward": 0.0, "passed": False, "critical_fail": True,
                       "citation_error": citation_error, "criteria": []}
        else:
            overlay["cited_source_sections"] = evidence
            messages = grade.build_messages(overlay, self.case["source_text"], self.actual_answers, answer)
            messages[0]["content"] += (
                ' Return the JSON object without Markdown fences. Use the permitted empty-string '
                'option for EVERY evidence_quote: "evidence_quote":"". Explain each score in '
                'the reason field; do not generate verbatim quotes. The full raw answer is saved '
                'separately for audit.'
            )
            prompt = self.judge_renderer.build_generation_prompt(messages)
            if self.sample_proxy.abort_state:
                raise FatalPaidRequestAbort(next(iter(self.sample_proxy.abort_state.values())))
            try:
                response = await asyncio.to_thread(
                    self.sample_fn, role="judge", case_id=self.case["id"],
                    turn_index=self.turn_index, call_index=1, prompt=prompt,
                )
            except asyncio.CancelledError as exc:
                reason = "judge sampling was cancelled; request status may be uncertain"
                self.sample_proxy.abort_state["reason"] = reason
                raise FatalPaidRequestAbort(reason) from exc
            sequence = response.sequences[0]
            judge_message, termination = self.judge_renderer.parse_response([int(token) for token in sequence.tokens])
            if str(sequence.stop_reason) != "stop" or not termination.is_clean or judge_message.get("unparsed_tool_calls"):
                raise run_eval.EvaluationError("Judge response was truncated or malformed")
            judge_text = run_eval.message_text(judge_message)
            fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", judge_text.strip(), re.DOTALL)
            if fenced:
                judge_text = fenced.group(1)
            judge_payload = json.loads(judge_text)
            if isinstance(judge_payload, dict) and isinstance(judge_payload.get("criteria"), list):
                for row in judge_payload["criteria"]:
                    if isinstance(row, dict) and isinstance(row.get("evidence_quote"), str):
                        quote = row["evidence_quote"]
                        if quote and quote not in answer:
                            discarded_quotes.append({"criterion_id": row.get("id"), "discarded_quote": quote})
                            row["evidence_quote"] = ""
            scores = grade.parse_scores(json.dumps(judge_payload, ensure_ascii=False), turn["rubric_items"], answer)
            verdict = grade.score_verdict(scores, turn["rubric_items"])
            verdict["discarded_unverifiable_quotes"] = discarded_quotes
            judge_call = "completed"

        verdict.update({"case_id": self.case["id"], "turn_index": self.turn_index})
        return {**turn_record, "judge_call": judge_call, "verdict": verdict, "status": "graded"}


class UnderwritingBenchmark(BenchmarkBuilder):
    name = BENCHMARK_NAME
    multi_turn = True
    requires_judge = True
    recommended_timeout = 1800

    def __init__(
        self, *, cases: list[dict], corpus: dict, save_dir: Path,
        candidate_renderer: Any, judge_renderer: Any, tools: list[Any],
        sample_proxy: CandidateSamplingProxy, judge_client: Any,
        sample_fn: Callable[..., Any], tool_executor: Callable[..., Any] = _execute_tool,
        turn_grader: Callable[..., Any] | None = None,
    ):
        self.turn_grader = turn_grader
        self.requires_judge = turn_grader is None
        self.cases, self.corpus, self.save_dir = cases, corpus, save_dir
        self.candidate_renderer, self.judge_renderer = candidate_renderer, judge_renderer
        self.tools, self.sample_proxy, self.judge_client = tools, sample_proxy, judge_client
        self.sample_fn, self.tool_executor = sample_fn, tool_executor

    def make_envs(self, renderer: Any, config: BenchmarkConfig) -> list[Any]:
        selected = self.cases[:config.max_examples] if config.max_examples is not None else self.cases
        return [
            EnvFromMessageEnv(
                renderer=renderer,
                message_env=UnderwritingMessageEnv(
                    case=case, corpus=self.corpus,
                    candidate_renderer=self.candidate_renderer, judge_renderer=self.judge_renderer,
                    tools=self.tools, sample_proxy=self.sample_proxy, judge_client=self.judge_client,
                    sample_fn=self.sample_fn, tool_executor=self.tool_executor, save_dir=self.save_dir,
                    turn_grader=self.turn_grader,
                ),
                failed_parse_reward=0.0, context_overflow_reward=0.0,
                max_trajectory_tokens=MAX_TRAJECTORY_TOKENS,
                max_generation_tokens=run_eval.OUTPUT_TOKEN_CAP,
            )
            for case in selected
        ]


def _record_manifest(
    *, case_ids: list[str], pricing_file: Path, protocol_file: Path, save_dir: Path, budget_usd: float,
    judge_no_output_cap: bool = False,
) -> dict[str, Any]:
    prices = run_eval.load_prices(pricing_file)
    corpus = build.build_corpus()
    config = {
        "benchmark": BENCHMARK_NAME, "cases": case_ids, "attempts_per_case": 1,
        "candidate": run_eval.CANDIDATE, "candidate_renderer": run_eval.CANDIDATE_RENDERER,
        "candidate_temperature": run_eval.CANDIDATE_TEMP, "candidate_seed": run_eval.CANDIDATE_SEED,
        "judge": run_eval.JUDGE, "judge_renderer": run_eval.JUDGE_RENDERER, "judge_temperature": 0.0,
        "judge_output_token_cap": None if judge_no_output_cap else run_eval.OUTPUT_TOKEN_CAP,
        "judge_context_tokens": 65536,
        "input_context_cap": CANDIDATE_INPUT_CONTEXT_CAP, "output_token_cap": run_eval.OUTPUT_TOKEN_CAP,
        "judge_input_context_cap": run_eval.INPUT_CONTEXT_CAP,
        "sample_calls_per_turn": run_eval.SAMPLE_CALLS_PER_TURN + 1,
        "retrieval_generations_per_turn": run_eval.SAMPLE_CALLS_PER_TURN,
        "reserved_final_answer_generations": 1,
        "tool_calls_per_turn": run_eval.TOOL_CALLS_PER_TURN, "max_trajectory_tokens": MAX_TRAJECTORY_TOKENS,
        "framework_concurrency": 1, "timeout_seconds": 1800, "budget_usd": budget_usd,
    }
    source_hashes = {
        info["file"]: run_eval.sha256_file(build.SOURCES_PATH / info["file"])
        for info in build.SOURCE_INFO.values()
    }
    input_hashes = {
        name: run_eval.sha256_file(ROOT / name)
        for name in ("cases.json", "grade.py", "build.py", "tool_env.py", "manual_tools.py", "run_eval.py", "registered_eval.py")
    }
    input_hashes.update(source_hashes)
    input_hashes["corpus.json"] = run_eval.sha256_file(ROOT / "corpus.json")
    input_hashes["pricing_file"] = run_eval.sha256_file(pricing_file)
    input_hashes["protocol_file"] = run_eval.sha256_file(protocol_file)
    config_hash = run_eval.canonical_hash({"configuration": config, "input_sha256": input_hashes})
    run_eval._check_save_dir(save_dir, config_hash)
    run_eval.save_json(save_dir / "corpus.json", corpus)
    manifest_path = save_dir / "manifest.json"
    if not manifest_path.exists():
        manifest = {
            "schema_version": 1, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_hash": config_hash, "configuration": config, "input_sha256": input_hashes,
            "pricing_file": str(pricing_file), "prices_per_million": prices,
            "execution_environment": {
                "python": sys.version.split()[0], "tinker": importlib.metadata.version("tinker"),
                "tinker_cookbook": importlib.metadata.version("tinker-cookbook"),
            },
            "verifier": {"model": run_eval.JUDGE, "renderer": run_eval.JUDGE_RENDERER,
                         "criteria": "grade.py binary scores; weighted turn reward; critical-fail gate; all-criteria pass on all three turns"},
            "evaluation": {"unit": "three-user-turn case", "metric": "pass@1",
                           "attempts": 1, "denominator": len(case_ids),
                           "sampling": "one newly sampled candidate trajectory per case"},
            "budget_cap_usd": budget_usd,
        }
        run_eval.save_json(manifest_path, manifest)
    return {"prices": prices, "corpus": corpus, "manifest": json.loads(manifest_path.read_text(encoding="utf-8"))}


def _assert_no_unresolved_requests(save_dir: Path) -> None:
    calls_dir = save_dir / "calls"
    if not calls_dir.is_dir():
        return
    unresolved = []
    for path in calls_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("status") in {"in_flight", "charge_uncertain"}:
            unresolved.append(path.name)
    if unresolved:
        raise run_eval.EvaluationError(
            "Unresolved paid request(s) exist; refusing to submit more samples: " + ", ".join(unresolved[:3])
        )


def _summarize(save_dir: Path, *, status: str, budget_usd: float, result: Any | None) -> dict[str, Any]:
    case_files = sorted((save_dir / "cases").glob("uw-*.json")) if (save_dir / "cases").exists() else []
    turn_files = sorted((save_dir / "turns").glob("uw-*-turn-*.json")) if (save_dir / "turns").exists() else []
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in case_files]
    turn_records = [json.loads(path.read_text(encoding="utf-8")) for path in turn_files]
    turns = [row for row in turn_records if isinstance(row.get("verdict"), dict)]
    planned_cases = len(json.loads((save_dir / "manifest.json").read_text())["configuration"]["cases"])
    passed = sum(bool(row["passed"]) for row in cases)
    turn_count = len(turns)
    summary = {
        "status": status, "planned_cases": planned_cases,
        "completed_conversations": len(cases), "passed_cases": passed,
        "pass@1": passed / planned_cases,
        "mean_turn_reward": sum(float(row["verdict"].get("reward", 0)) for row in turns) / turn_count if turn_count else None,
        "critical_failure_rate": sum(bool(row["verdict"].get("critical_fail")) for row in turns) / turn_count if turn_count else None,
        "citation_failure_count": sum(bool(row["verdict"].get("citation_error")) for row in turns),
        "discarded_unverifiable_quote_count": sum(len(row["verdict"].get("discarded_unverifiable_quotes", [])) for row in turns),
        "completed_turns": turn_count,
        "ungraded_or_failed_turns": len(turn_records) - turn_count,
        "tool_calls": sum(int(row.get("tool_calls", 0)) for row in turns),
        "candidate_sample_calls": sum(int(row.get("candidate_sample_calls", 0)) for row in turns),
        "reserved_usd": run_eval._reserved_total(save_dir / "calls"), "budget_cap_usd": budget_usd,
        "jev_agreement": "not measured; the benchmark has no Jev labels",
        "calibration": "not applicable to the plain-text response schema",
    }
    if result is not None:
        summary["framework_result"] = {
            "num_examples": result.num_examples, "num_correct": result.num_correct,
            "num_errors": result.num_errors, "num_truncated": result.num_truncated,
            "metrics": result.metrics,
        }
    run_eval.save_json(save_dir / "summary.json", summary)
    return summary


async def run_registered(
    *, case_ids: list[str], save_dir: Path, pricing_file: Path, protocol_file: Path,
    budget_usd: float = MAX_BUDGET_USD,
    judge_no_output_cap: bool = False,
    candidate_renderer: Any | None = None, judge_renderer: Any | None = None,
    tools: list[Any] | None = None, candidate_client: Any | None = None,
    judge_client: Any | None = None, sample_fn: Callable[..., Any] | None = None,
    tool_executor: Callable[..., Any] = _execute_tool,
) -> dict[str, Any]:
    if not case_ids or len(case_ids) != len(set(case_ids)) or any(cid not in run_eval.ALL_CASE_IDS for cid in case_ids):
        raise run_eval.EvaluationError("case IDs must be a non-empty unique subset of uw-01 through uw-15")
    if not math.isfinite(budget_usd) or budget_usd <= 0 or budget_usd > MAX_BUDGET_USD:
        raise run_eval.EvaluationError(f"budget must be finite, positive, and no more than ${MAX_BUDGET_USD:.2f}")
    _assert_no_unresolved_requests(save_dir)
    prepared = _record_manifest(
        case_ids=case_ids, pricing_file=pricing_file, protocol_file=protocol_file,
        save_dir=save_dir, budget_usd=budget_usd, judge_no_output_cap=judge_no_output_cap,
    )
    cases_all = {row["id"]: row for row in build.load_cases()}
    cases = [cases_all[cid] for cid in case_ids]
    abort_state: dict[str, str] = {}
    service = None

    if sample_fn is None:
        credential_source = run_eval.ensure_tinker_token()
        tinker, RetryConfig, ToolFactory, loaded_candidate, loaded_judge = run_eval._load_runtime()
        candidate_renderer = candidate_renderer or loaded_candidate
        judge_renderer = judge_renderer or loaded_judge
        service = tinker.ServiceClient(display_name=f"inkling-underwriting-{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}")
        retry = RetryConfig(enable_retry_logic=False, progress_timeout=300)
        candidate_client = candidate_client or service.create_sampling_client(base_model=run_eval.CANDIDATE, retry_config=retry).result()
        judge_client = judge_client or service.create_sampling_client(base_model=run_eval.JUDGE, retry_config=retry).result()
        run_eval.disable_sample_submission_retries(service)
        manifest = prepared["manifest"]
        prior_sessions = manifest.get("tinker_session_ids", [])
        if manifest.get("tinker_session_id"):
            prior_sessions.append(manifest["tinker_session_id"])
        manifest["tinker_session_url"] = service.get_console_url()
        manifest["tinker_session_id"] = manifest["tinker_session_url"].rstrip("/").split("/")[-1]
        manifest["tinker_session_ids"] = list(dict.fromkeys(prior_sessions + [manifest["tinker_session_id"]]))
        if tools is None:
            tool_owner = ToolFactory(save_dir / "corpus.json")
            tools = [tool_owner.search_manual, tool_owner.read_section]

        def guarded_sample(*, role: str, case_id: str, turn_index: int, call_index: int, prompt: Any) -> Any:
            if abort_state:
                raise FatalPaidRequestAbort(next(iter(abort_state.values())))
            try:
                _assert_no_unresolved_requests(save_dir)
            except run_eval.EvaluationError as exc:
                abort_state["reason"] = str(exc)
                raise FatalPaidRequestAbort(str(exc)) from exc
            model, renderer_name = (
                (run_eval.CANDIDATE, run_eval.CANDIDATE_RENDERER)
                if role == "candidate" else (run_eval.JUDGE, run_eval.JUDGE_RENDERER)
            )
            renderer = candidate_renderer if role == "candidate" else judge_renderer
            try:
                return run_eval.sample_once(
                    client=candidate_client if role == "candidate" else judge_client,
                    tinker=tinker, model=model, renderer_name=renderer_name, role=role,
                    case_id=case_id, turn_index=turn_index, prompt=prompt,
                    stop_sequences=renderer.get_stop_sequences(),
                    temperature=run_eval.CANDIDATE_TEMP if role == "candidate" else 0.0,
                    seed=run_eval.CANDIDATE_SEED if role == "candidate" else None,
                    prices=prepared["prices"][model], budget_usd=budget_usd, save_dir=save_dir,
                    include_logprobs=role == "candidate",
                    output_token_cap=None if role == "judge" and judge_no_output_cap else run_eval.OUTPUT_TOKEN_CAP,
                    model_context_tokens=65536 if role == "judge" else None,
                    input_context_cap=CANDIDATE_INPUT_CONTEXT_CAP if role == "candidate" else run_eval.INPUT_CONTEXT_CAP,
                )
            except run_eval.BudgetExceeded as exc:
                abort_state["reason"] = str(exc)
                raise FatalPaidRequestAbort(str(exc)) from exc
            except run_eval.EvaluationError as exc:
                if "charge status is uncertain" in str(exc):
                    abort_state["reason"] = str(exc)
                    raise FatalPaidRequestAbort(str(exc)) from exc
                raise

        sample_fn = guarded_sample
        manifest["credential_source"] = credential_source
        run_eval.save_json(save_dir / "manifest.json", manifest)
    elif candidate_renderer is None or judge_renderer is None or tools is None or candidate_client is None or judge_client is None:
        raise run_eval.EvaluationError("Injected offline runs must supply renderers, tools, and fake judge/candidate clients")

    assert candidate_renderer is not None and judge_renderer is not None and tools is not None
    candidate_proxy = CandidateSamplingProxy(sample_fn, candidate_renderer, abort_state)
    builder = UnderwritingBenchmark(
        cases=cases, corpus=prepared["corpus"], save_dir=save_dir,
        candidate_renderer=candidate_renderer, judge_renderer=judge_renderer,
        tools=tools, sample_proxy=candidate_proxy, judge_client=judge_client,
        sample_fn=sample_fn, tool_executor=tool_executor,
    )
    register(builder)
    if REGISTRY.get(BENCHMARK_NAME) is not builder:
        raise run_eval.EvaluationError("Underwriting BenchmarkBuilder registration failed")

    config = BenchmarkConfig(
        max_tokens=run_eval.OUTPUT_TOKEN_CAP, temperature=run_eval.CANDIDATE_TEMP,
        context_window=None, timeout_seconds=1800, concurrency=1, agent_concurrency=1,
        max_trajectory_tokens=MAX_TRAJECTORY_TOKENS,
        max_generation_tokens=run_eval.OUTPUT_TOKEN_CAP, num_samples=1,
        judge_sampling_client=judge_client, judge_renderer=judge_renderer,
        save_dir=str(save_dir),
    )
    run_eval.save_json(save_dir / "framework_config.json", {
        "benchmark_name": BENCHMARK_NAME,
        "config": {
            "max_tokens": config.max_tokens, "temperature": config.temperature,
            "context_window": config.context_window, "timeout_seconds": config.timeout_seconds,
            "concurrency": config.concurrency, "agent_concurrency": config.agent_concurrency,
            "max_trajectory_tokens": config.max_trajectory_tokens,
            "max_generation_tokens": config.max_generation_tokens, "num_samples": config.num_samples,
            "save_dir": config.save_dir, "judge_model": run_eval.JUDGE,
            "judge_renderer": run_eval.JUDGE_RENDERER,
        },
        "candidate_sample_params": {"max_tokens": run_eval.OUTPUT_TOKEN_CAP,
                                     "temperature": run_eval.CANDIDATE_TEMP,
                                     "seed": run_eval.CANDIDATE_SEED},
        "judge_sample_params": {**({} if judge_no_output_cap else {"max_tokens": run_eval.OUTPUT_TOKEN_CAP}), "temperature": 0.0, "seed": None},
        "effective_candidate_input_cap": CANDIDATE_INPUT_CONTEXT_CAP,
    })
    result = None
    status = "completed"
    try:
        result = await run_benchmark(BENCHMARK_NAME, candidate_proxy, candidate_renderer, config)
    except FatalPaidRequestAbort as exc:
        status = "stopped_budget_or_ambiguous_request"
        run_eval.save_json(save_dir / "error.json", {"error_type": type(exc).__name__, "message": str(exc)})
    finally:
        summary = _summarize(save_dir, status=status, budget_usd=budget_usd, result=result)
        if service is not None:
            service.close("success" if status == "completed" else "errored").result()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=run_eval.ALL_CASE_IDS, default=list(run_eval.ALL_CASE_IDS))
    parser.add_argument("--save-dir", type=Path, default=RECORD_DIR / "trial")
    parser.add_argument("--pricing-file", type=Path, default=RECORD_DIR / "pricing.json")
    parser.add_argument("--protocol-file", type=Path, default=PROTOCOL_PATH)
    parser.add_argument("--budget-usd", type=float, default=MAX_BUDGET_USD)
    parser.add_argument("--judge-no-output-cap", action="store_true", help="Omit Qwen max_tokens; reserve cost through its 64K service context boundary")
    args = parser.parse_args()
    try:
        summary = asyncio.run(run_registered(
            case_ids=args.cases, save_dir=args.save_dir, pricing_file=args.pricing_file,
            protocol_file=args.protocol_file, budget_usd=args.budget_usd,
            judge_no_output_cap=args.judge_no_output_cap,
        ))
        print(json.dumps(summary, indent=2))
        return 0 if summary["status"] == "completed" else 2
    except Exception as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

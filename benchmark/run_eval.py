#!/usr/bin/env python3
"""Run a bounded two-case, three-user-turn Tinker tool-use pilot."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Callable


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import build  # noqa: E402
import grade  # noqa: E402


CANDIDATE = "zai-org/GLM-5.3:peft:262144"
CANDIDATE_TOKENIZER = "zai-org/GLM-5.3"
CANDIDATE_RENDERER = "glm5_3_max_reasoning"
JUDGE = "Qwen/Qwen3.5-397B-A17B"
JUDGE_RENDERER = "qwen3_5"
PILOT_CASES = ("uw-02", "uw-08")
ALL_CASE_IDS = tuple(f"uw-{i:02d}" for i in range(1, 16))
INPUT_CONTEXT_CAP = 32768
OUTPUT_TOKEN_CAP = 8192
SAMPLE_CALLS_PER_TURN = 4
TOOL_CALLS_PER_TURN = 4
CANDIDATE_TEMP = 0.6
CANDIDATE_SEED = 20261115
MAX_BUDGET_USD = 5.0


class EvaluationError(RuntimeError):
    pass


class BudgetExceeded(EvaluationError):
    pass


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return jsonable(value.model_dump())
    if hasattr(value, "value") and isinstance(value.value, (str, int, float)):
        return value.value
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(jsonable(value), stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def canonical_hash(value: Any) -> str:
    raw = json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _price_number(value: Any, field: str) -> float:
    match = re.fullmatch(r"\s*\$?\s*(\d+(?:\.\d+)?)\s*", str(value))
    if not match:
        raise EvaluationError(f"Invalid {field} price in pricing snapshot")
    rate = float(match.group(1))
    if not math.isfinite(rate) or rate < 0:
        raise EvaluationError(f"Invalid {field} price in pricing snapshot")
    return rate


def load_prices(path: Path, models: tuple[str, ...] = (CANDIDATE, JUDGE)) -> dict[str, dict[str, float]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows.get("models")
    if not isinstance(rows, list):
        raise EvaluationError("Pricing snapshot must be a model list or contain a models list")
    prices: dict[str, dict[str, float]] = {}
    for model in models:
        matching = [row for row in rows if isinstance(row, dict) and row.get("tinker_id") == model]
        if len(matching) != 1:
            raise EvaluationError(f"Pricing snapshot must contain one exact row for {model}")
        row = matching[0]
        prices[model] = {
            "prefill": _price_number(row.get("prefill"), "prefill"),
            "cached_prefill": _price_number(row.get("cached_prefill"), "cached_prefill"),
            "sample": _price_number(row.get("sample"), "sample"),
        }
    return prices


def _prompt_tokens(prompt: Any) -> list[int]:
    try:
        return [int(token) for token in prompt.to_ints()]
    except AttributeError as exc:
        raise EvaluationError("Renderer did not return a Tinker ModelInput") from exc


def _reserved_total(calls_dir: Path) -> float:
    total = 0.0
    for path in calls_dir.glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        total += float(record.get("reservation_usd", 0.0))
    return total


def sample_once(
    *,
    client: Any,
    tinker: Any,
    model: str,
    renderer_name: str,
    role: str,
    case_id: str,
    turn_index: int,
    prompt: Any,
    stop_sequences: list[str] | list[int],
    temperature: float,
    seed: int | None,
    prices: dict[str, float],
    budget_usd: float,
    save_dir: Path,
    include_logprobs: bool = False,
    output_token_cap: int | None = OUTPUT_TOKEN_CAP,
    model_context_tokens: int | None = None,
    input_context_cap: int = INPUT_CONTEXT_CAP,
) -> Any:
    tokens = _prompt_tokens(prompt)
    if len(tokens) > input_context_cap:
        raise EvaluationError(f"{role} prompt exceeds the {input_context_cap}-token input cap")
    params = {
        "temperature": temperature, "seed": seed,
        "stop": stop_sequences,
    }
    if output_token_cap is not None:
        params["max_tokens"] = output_token_cap
        reserved_output_tokens = output_token_cap
    else:
        if model_context_tokens is None or model_context_tokens <= len(tokens):
            raise EvaluationError("Uncapped output requires a verified model context bound")
        reserved_output_tokens = model_context_tokens - len(tokens)
    request = {
        "model": model,
        "renderer": renderer_name,
        "prompt_tokens": tokens,
        "sampling_params": params,
    }
    request_hash = canonical_hash(request)
    calls_dir = save_dir / "calls"
    call_path = calls_dir / f"{request_hash}.json"
    if call_path.exists():
        saved = json.loads(call_path.read_text(encoding="utf-8"))
        if saved.get("status") != "completed":
            raise EvaluationError(
                f"Unresolved {saved.get('status', 'unknown')} request {request_hash}; refusing replay"
            )
        response = saved["response"]
        sequence = response["sequences"][0]
        print(
            f"[reuse] case={case_id} turn={turn_index} role={role} "
            f"estimated=${saved['estimated_cost_usd']:.5f}",
            flush=True,
        )
        return SimpleNamespace(
            sequences=[SimpleNamespace(
                tokens=sequence["tokens"], stop_reason=sequence["stop_reason"],
                logprobs=sequence.get("logprobs"), sequence_id=sequence.get("sequence_id"),
            )],
            prompt_cache_hit_tokens=response.get("prompt_cache_hit_tokens", 0),
        )

    initial_reservation = (
        len(tokens) * prices["prefill"] + reserved_output_tokens * prices["sample"]
    ) / 1_000_000
    used = _reserved_total(calls_dir)
    if used + initial_reservation > budget_usd + 1e-12:
        raise BudgetExceeded(
            f"Budget guard before {role} request: ${used + initial_reservation:.5f} "
            f"would exceed ${budget_usd:.2f}"
        )

    record = {
        "status": "in_flight",
        "request_hash": request_hash,
        "case_id": case_id,
        "turn_index": turn_index,
        "role": role,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "request": request,
        "initial_reservation_usd": initial_reservation,
        "reservation_usd": initial_reservation,
    }
    save_json(call_path, record)
    print(
        f"[start] case={case_id} turn={turn_index} role={role} reserve=${initial_reservation:.5f}",
        flush=True,
    )
    try:
        response = client.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=tinker.SamplingParams(**params),
        ).result()
    except Exception as exc:
        record.update(
            status="charge_uncertain",
            finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
            error_type=type(exc).__name__,
        )
        save_json(call_path, record)
        raise EvaluationError(
            f"{role} request failed; charge status is uncertain and replay is disabled"
        ) from exc

    raw_sequences = [
        {
            "tokens": [int(token) for token in sequence.tokens],
            "stop_reason": str(sequence.stop_reason),
            **({"logprobs": list(sequence.logprobs) if sequence.logprobs is not None else None}
               if include_logprobs else {}),
            **({"sequence_id": sequence.sequence_id} if include_logprobs else {}),
        }
        for sequence in response.sequences
    ]
    cache_hits = min(len(tokens), max(0, int(getattr(response, "prompt_cache_hit_tokens", 0))))
    completion_tokens = len(raw_sequences[0]["tokens"]) if raw_sequences else 0
    estimated_cost = (
        (len(tokens) - cache_hits) * prices["prefill"]
        + cache_hits * prices["cached_prefill"]
        + completion_tokens * prices["sample"]
    ) / 1_000_000
    conservative_actual = (len(tokens) * prices["prefill"] + completion_tokens * prices["sample"]) / 1_000_000
    record.update(
        status="completed",
        finished_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        response={"sequences": raw_sequences, "prompt_cache_hit_tokens": cache_hits},
        prompt_tokens=len(tokens),
        completion_tokens=completion_tokens,
        estimated_cost_usd=estimated_cost,
        conservative_uncached_cost_usd=conservative_actual,
        reservation_usd=conservative_actual,
    )
    # Persist the raw completion and token accounting before parsing any model output.
    save_json(call_path, record)
    print(
        f"[call] case={case_id} turn={turn_index} role={role} "
        f"reserved=${initial_reservation:.5f} estimated=${estimated_cost:.5f}",
        flush=True,
    )
    if not raw_sequences:
        raise EvaluationError(f"{role} request returned no samples")
    return response


def message_text(message: Any) -> str:
    content = message.get("content", "") if isinstance(message, dict) else ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )
    return ""


def _tool_call_name_args(tool_call: Any) -> tuple[str, dict[str, Any]]:
    function = tool_call.get("function", {}) if isinstance(tool_call, dict) else tool_call.function
    if isinstance(function, dict):
        name, raw = function.get("name", ""), function.get("arguments", "{}")
    else:
        name, raw = function.name, function.arguments
    try:
        args = json.loads(raw) if isinstance(raw, str) else dict(raw)
    except (json.JSONDecodeError, TypeError, ValueError):
        args = {}
    return str(name), args


def _blocked_tool_message(tool_call: Any, name: str) -> dict[str, Any]:
    call_id = getattr(tool_call, "id", "")
    if isinstance(tool_call, dict):
        call_id = tool_call.get("id", "")
    call_id = getattr(tool_call, "id", "")
    if isinstance(tool_call, dict):
        call_id = tool_call.get("id", "")
    return {
        "role": "tool", "content": json.dumps({"error": "Tool or source is not allowed."}),
        "tool_call_id": call_id or "", "name": name,
    }


def make_candidate_system_prompt(case: dict) -> str:
    return (
        "This is a fictional, source-grounded underwriting exercise. Use only the assigned pinned source "
        "and stipulated facts. Do not make a binding insurance decision. Use search_manual and read_section "
        "for the assigned source, check relevant exceptions, and cite at least one relevant returned section "
        "using its full section_id in brackets. Put exactly one unmodified section_id in each bracket. "
        "For multiple pages write [tx:p003] [tx:p004], never a page range or comma-separated IDs. "
        "Put explanations outside brackets: [ma:p052] (item 7), not [ma:p052, item 7]. "
        "These are formatting examples only; cite only sections actually returned for your assigned source.\n\n"
        f"{build.FOLLOW_UP_INSTRUCTION}\n\n"
        f"Scope: {case['scope']}\nAssigned source: {build.assigned_source(case)}\n"
        f"Assigned source_id: {case['source_id']}"
    )


def run_case(
    *,
    case: dict,
    corpus: dict,
    candidate_renderer: Any,
    judge_renderer: Any,
    tools: list[Any],
    sample_fn: Callable[..., Any],
    tool_executor: Callable[[Any, dict[str, Any], str], list[Any]],
    save_dir: Path,
) -> dict:
    tool_map = {tool.name: tool for tool in tools}
    tool_specs = [tool.to_spec() for tool in tools]
    messages = candidate_renderer.create_conversation_prefix_with_tools(
        tools=tool_specs, system_prompt=make_candidate_system_prompt(case)
    )
    actual_answers: list[str] = []
    turn_results: list[dict] = []

    for turn_index, turn in enumerate(case["turns"], start=1):
        messages.append({"role": "user", "content": turn["user"]})
        tool_calls_this_turn = 0
        sample_calls = 0
        answer = ""
        while sample_calls < SAMPLE_CALLS_PER_TURN:
            prompt = candidate_renderer.build_generation_prompt(messages)
            sampled = sample_fn(
                role="candidate", case_id=case["id"], turn_index=turn_index,
                call_index=sample_calls + 1, prompt=prompt,
            )
            sample_calls += 1
            sequence = sampled.sequences[0]
            if sequence.stop_reason != "stop":
                raise EvaluationError("Candidate response hit the output limit; refusing to grade it")
            assistant_message, termination = candidate_renderer.parse_response(
                [int(token) for token in sequence.tokens]
            )
            if not termination.is_clean or assistant_message.get("unparsed_tool_calls"):
                raise EvaluationError("Candidate response was malformed; refusing to grade it")
            messages.append(assistant_message)
            tool_calls = list(assistant_message.get("tool_calls") or [])
            if not tool_calls:
                answer = message_text(assistant_message)
                break
            for tool_call in tool_calls:
                name, args = _tool_call_name_args(tool_call)
                if tool_calls_this_turn >= TOOL_CALLS_PER_TURN:
                    results = [_blocked_tool_message(tool_call, name)]
                elif args.get("source_id") != case["source_id"]:
                    results = [_blocked_tool_message(tool_call, name)]
                    tool_calls_this_turn += 1
                elif name not in tool_map:
                    results = [_blocked_tool_message(tool_call, name)]
                    tool_calls_this_turn += 1
                else:
                    results = tool_executor(tool_map[name], tool_call, case["source_id"])
                    tool_calls_this_turn += 1
                messages.extend(results)
            # If the fourth sampling call requested more tools, no fifth billable call is made.

        overlay = build.overlay(case, turn_index)
        evidence, citation_error = grade.citation_evidence(answer, corpus, case["source_id"])
        if citation_error:
            verdict = {
                "reward": 0.0, "passed": False, "critical_fail": True,
                "citation_error": citation_error, "criteria": [],
            }
            judge_call = "not_called"
        else:
            overlay["cited_source_sections"] = evidence
            judge_messages = grade.build_messages(overlay, case["source_text"], actual_answers, answer)
            judge_messages[0]["content"] += (
                ' Return the JSON object without Markdown fences. Use the permitted empty-string '
                'option for EVERY evidence_quote: "evidence_quote":"". Explain each score in '
                'the reason field; do not generate verbatim quotes. The full raw answer is saved '
                'separately for audit.'
            )
            judge_prompt = judge_renderer.build_generation_prompt(judge_messages)
            judge_response = sample_fn(
                role="judge", case_id=case["id"], turn_index=turn_index,
                call_index=1, prompt=judge_prompt,
            )
            judge_message, _judge_termination = judge_renderer.parse_response(
                [int(token) for token in judge_response.sequences[0].tokens]
            )
            if (
                judge_response.sequences[0].stop_reason != "stop"
                or not _judge_termination.is_clean
                or judge_message.get("unparsed_tool_calls")
            ):
                raise EvaluationError("Judge response was truncated or malformed")
            judge_text = message_text(judge_message)
            # Qwen may wrap otherwise valid JSON in one Markdown code block.
            # Strip only that outer envelope; retain every strict schema/quote check.
            fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", judge_text.strip(), re.DOTALL)
            if fenced:
                judge_text = fenced.group(1)
            # Quotes are optional audit metadata, not a rubric criterion. Never
            # present a fabricated quote as evidence or change the judge's score.
            quote_audit = []
            judge_payload = json.loads(judge_text)
            if isinstance(judge_payload, dict) and isinstance(judge_payload.get("criteria"), list):
                for row in judge_payload["criteria"]:
                    if isinstance(row, dict) and isinstance(row.get("evidence_quote"), str):
                        quote = row["evidence_quote"]
                        if quote and quote not in answer:
                            quote_audit.append({"criterion_id": row.get("id"), "discarded_quote": quote})
                            row["evidence_quote"] = ""
            judge_text = json.dumps(judge_payload, ensure_ascii=False)
            scores = grade.parse_scores(judge_text, turn["rubric_items"], answer)
            verdict = grade.score_verdict(scores, turn["rubric_items"])
            verdict["discarded_unverifiable_quotes"] = quote_audit
            judge_call = "completed"

        verdict.update({"case_id": case["id"], "turn_index": turn_index})
        turn_record = {
            "case_id": case["id"], "turn_index": turn_index, "user": turn["user"],
            "answer": answer, "tool_calls": tool_calls_this_turn,
            "candidate_sample_calls": sample_calls, "judge_call": judge_call,
            "verdict": verdict, "actual_prior_answers": list(actual_answers),
            "actual_message_history": jsonable(messages),
        }
        save_json(save_dir / "turns" / f"{case['id']}-turn-{turn_index}.json", turn_record)
        turn_results.append(turn_record)
        actual_answers.append(answer)

    passed = len(turn_results) == 3 and all(row["verdict"].get("passed") is True for row in turn_results)
    result = {"case_id": case["id"], "status": "completed", "passed": passed, "turns": turn_results}
    save_json(save_dir / "cases" / f"{case['id']}.json", result)
    return result


def execute_tool_call(tool: Any, tool_call: Any, assigned_source_id: str) -> list[Any]:
    del assigned_source_id
    from tinker_cookbook.tool_use.tools import handle_tool_call

    return asyncio.run(handle_tool_call({tool.name: tool}, tool_call)).messages


def disable_sample_submission_retries(service: Any) -> None:
    """Avoid replaying a potentially billable sample POST after an uncertain timeout.

    Tinker 0.39's SamplingClient also calls holder.execute_with_retries around
    `_send_asample_request`; RetryConfig only disables its outer retry handler.
    Keep SDK retries for non-sample work such as polling, but submit each sample
    POST once. A failed submission remains charge-uncertain in the local ledger.
    """
    holder = service.holder
    original = holder.execute_with_retries

    async def execute_once_for_samples(fn: Any, *args: Any, **kwargs: Any) -> Any:
        if getattr(fn, "__name__", "") == "_send_asample_request":
            return await fn(*args, **kwargs)
        return await original(fn, *args, **kwargs)

    holder.execute_with_retries = execute_once_for_samples


def ensure_tinker_token() -> str:
    if os.environ.get("TINKER_API_KEY"):
        return "environment"
    env_file = REPO_ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("TINKER_API_KEY="):
                token = line.split("=", 1)[1].strip().strip('"').strip("'")
                if token:
                    os.environ["TINKER_API_KEY"] = token
                    return "project_env_file"
    from tinker.auth import get_tinker_token

    token = get_tinker_token()
    if token:
        os.environ["TINKER_API_KEY"] = token
        return "tinker_cli_auth"
    raise EvaluationError("No Tinker credential found in environment, project .env, or CLI auth")


def _check_save_dir(save_dir: Path, config_hash: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    manifest = save_dir / "manifest.json"
    existing = list(save_dir.iterdir())
    if not existing:
        return
    if not manifest.exists():
        raise EvaluationError("Save directory is non-empty and has no compatible manifest")
    prior = json.loads(manifest.read_text(encoding="utf-8"))
    if prior.get("config_hash") != config_hash:
        raise EvaluationError("Save directory belongs to a different pilot configuration")


def _load_runtime():
    import tinker
    from tinker.lib.retry_handler import RetryConfig
    from tinker_cookbook import renderers
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tool_env import TinkerManualTools

    candidate_renderer = renderers.get_renderer(
        CANDIDATE_RENDERER, get_tokenizer(CANDIDATE_TOKENIZER), model_name=CANDIDATE
    )
    judge_renderer = renderers.get_renderer(JUDGE_RENDERER, get_tokenizer(JUDGE), model_name=JUDGE)
    return tinker, RetryConfig, TinkerManualTools, candidate_renderer, judge_renderer


def _run_preflight(pricing_file: Path) -> dict:
    prices = load_prices(pricing_file)
    credential_source = ensure_tinker_token()
    tinker, _RetryConfig, _Tools, candidate_renderer, judge_renderer = _load_runtime()
    capabilities = tinker.RestClient().get_server_capabilities().result()
    supported = {model.model_name: model for model in capabilities.supported_models}
    for model in models:
        if model not in supported or not supported[model].sampleable:
            raise EvaluationError(f"Tinker capability check did not confirm sampling access for {model}")
    return {
        "status": "preflight_passed", "models": [CANDIDATE, JUDGE],
        "renderers": [candidate_renderer.__class__.__name__, judge_renderer.__class__.__name__],
        "prices_per_million": prices,
        "credential_source": credential_source,
        "sampling_requests": 0,
    }


def run_pilot(
    *,
    case_ids: list[str],
    save_dir: Path,
    pricing_file: Path,
    budget_usd: float,
    sample_fn: Callable[..., Any] | None = None,
    candidate_renderer: Any | None = None,
    judge_renderer: Any | None = None,
    tools: list[Any] | None = None,
    tool_executor: Callable[[Any, Any, str], list[Any]] | None = None,
) -> dict:
    if not math.isfinite(budget_usd) or budget_usd <= 0:
        raise EvaluationError("budget must be finite and positive")
    if not case_ids or len(case_ids) != len(set(case_ids)) or any(case_id not in ALL_CASE_IDS for case_id in case_ids):
        raise EvaluationError("case IDs must be a non-empty unique subset of uw-01 through uw-15")

    prices = load_prices(pricing_file)
    cases_all = {row["id"]: row for row in build.load_cases()}
    corpus = build.build_corpus()
    cases = [cases_all[case_id] for case_id in case_ids]
    manifest_config = {
        "cases": case_ids, "attempts_per_case": 1, "candidate": CANDIDATE,
        "candidate_renderer": CANDIDATE_RENDERER, "candidate_temperature": CANDIDATE_TEMP,
        "candidate_seed": CANDIDATE_SEED, "judge": JUDGE, "judge_renderer": JUDGE_RENDERER,
        "judge_temperature": 0.0, "input_context_cap": INPUT_CONTEXT_CAP,
        "output_token_cap": OUTPUT_TOKEN_CAP, "sample_calls_per_turn": SAMPLE_CALLS_PER_TURN,
        "tool_calls_per_turn": TOOL_CALLS_PER_TURN, "budget_usd": budget_usd,
    }
    source_hashes = {
        info["file"]: sha256_file(build.SOURCES_PATH / info["file"])
        for info in build.SOURCE_INFO.values()
    }
    input_hashes = {
        "cases.json": sha256_file(ROOT / "cases.json"),
        "grade.py": sha256_file(ROOT / "grade.py"),
        "build.py": sha256_file(ROOT / "build.py"),
        "tool_env.py": sha256_file(ROOT / "tool_env.py"),
        "manual_tools.py": sha256_file(ROOT / "manual_tools.py"),
        "run_eval.py": sha256_file(ROOT / "run_eval.py"),
        **source_hashes,
        "corpus.json": canonical_hash(corpus),
        "pricing_file": sha256_file(pricing_file),
    }
    config_hash = canonical_hash({"configuration": manifest_config, "input_sha256": input_hashes})
    _check_save_dir(save_dir, config_hash)
    save_json(save_dir / "corpus.json", corpus)
    manifest_path = save_dir / "manifest.json"
    if not manifest_path.exists():
        tinker_version = cookbook_version = "injected-test-runtime"
        if sample_fn is None:
            tinker_version = importlib.metadata.version("tinker")
            cookbook_version = importlib.metadata.version("tinker-cookbook")
        manifest = {
            "schema_version": 1, "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "config_hash": config_hash, "configuration": manifest_config,
            "input_sha256": input_hashes,
            "pricing_file": str(pricing_file), "prices_per_million": prices,
            "execution_environment": {"python": sys.version.split()[0], "tinker": tinker_version,
                                      "tinker_cookbook": cookbook_version},
            "verifier": {"model": JUDGE, "renderer": JUDGE_RENDERER,
                         "criteria": "grade.py binary criterion scores; weighted turn reward; critical-fail gate"},
            "evaluation": {"unit": "case with all 3 sequential user turns passing",
                           "metric": "pass@1", "attempts": 1,
                           "denominator": len(case_ids), "sampling": "single seeded candidate attempt per case"},
            "credential_source": "not recorded for injected tests",
            "budget_cap_usd": budget_usd,
        }
        save_json(manifest_path, manifest)

    tinker = None
    service = None
    if sample_fn is None:
        credential_source = ensure_tinker_token()
        tinker, RetryConfig, ToolFactory, loaded_candidate_renderer, loaded_judge_renderer = _load_runtime()
        candidate_renderer = candidate_renderer or loaded_candidate_renderer
        judge_renderer = judge_renderer or loaded_judge_renderer
        service = tinker.ServiceClient(
            display_name=f"inkling-tool-pilot-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
        )
        retry = RetryConfig(enable_retry_logic=False, progress_timeout=300)
        clients = {
            "candidate": service.create_sampling_client(base_model=CANDIDATE, retry_config=retry).result(),
            "judge": service.create_sampling_client(base_model=JUDGE, retry_config=retry).result(),
        }
        disable_sample_submission_retries(service)
        tools_obj = ToolFactory(save_dir / "corpus.json")
        tools = [tools_obj.search_manual, tools_obj.read_section]
        tool_executor = execute_tool_call

        def sample_fn(*, role: str, case_id: str, turn_index: int, call_index: int, prompt: Any) -> Any:
            model, renderer_name = (
                (CANDIDATE, CANDIDATE_RENDERER) if role == "candidate" else (JUDGE, JUDGE_RENDERER)
            )
            temperature = CANDIDATE_TEMP if role == "candidate" else 0.0
            seed = CANDIDATE_SEED if role == "candidate" else None
            return sample_once(
                client=clients[role], tinker=tinker, model=model, renderer_name=renderer_name,
                role=role, case_id=case_id, turn_index=turn_index, prompt=prompt,
                temperature=temperature, seed=seed, prices=prices[model], budget_usd=budget_usd,
                stop_sequences=(candidate_renderer if role == "candidate" else judge_renderer).get_stop_sequences(),
                save_dir=save_dir,
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["credential_source"] = credential_source
        prior_sessions = manifest.get("tinker_session_ids", [])
        if manifest.get("tinker_session_id"):
            prior_sessions.append(manifest["tinker_session_id"])
        manifest["tinker_session_url"] = service.get_console_url()
        manifest["tinker_session_id"] = manifest["tinker_session_url"].rstrip("/").split("/")[-1]
        manifest["tinker_session_ids"] = list(dict.fromkeys(prior_sessions + [manifest["tinker_session_id"]]))
        save_json(manifest_path, manifest)
    elif candidate_renderer is None or judge_renderer is None or tools is None or tool_executor is None:
        raise EvaluationError("Injected test runs must supply renderers, tools, and a tool executor")

    results = []
    status = "completed"
    error_type = None
    try:
        for case in cases:
            results.append(run_case(
                case=case, corpus=corpus, candidate_renderer=candidate_renderer,
                judge_renderer=judge_renderer, tools=tools, sample_fn=sample_fn,
                tool_executor=tool_executor, save_dir=save_dir,
            ))
    except Exception as exc:
        status = "failed"
        error_type = type(exc).__name__
        save_json(save_dir / "error.json", {"error_type": error_type, "message": str(exc)})
        raise
    finally:
        completed_files = list((save_dir / "cases").glob("uw-*.json")) if (save_dir / "cases").exists() else []
        saved_cases = [json.loads(path.read_text(encoding="utf-8")) for path in completed_files]
        turn_files = list((save_dir / "turns").glob("uw-*.json")) if (save_dir / "turns").exists() else []
        turn_rows = [json.loads(path.read_text(encoding="utf-8")) for path in turn_files]
        passed = sum(bool(result.get("passed")) for result in saved_cases)
        summary = {
            "status": status, "error_type": error_type, "planned_cases": len(case_ids),
            "completed_cases": len(saved_cases), "passed_cases": passed,
            "pass@1": passed / len(case_ids),
            "mean_turn_reward": (
                sum(float(row["verdict"].get("reward", 0.0)) for row in turn_rows) / len(turn_rows)
                if turn_rows else None
            ),
            "critical_failure_rate": (
                sum(bool(row["verdict"].get("critical_fail", False)) for row in turn_rows) / len(turn_rows)
                if turn_rows else None
            ),
            "completed_turns": len(turn_rows),
            "reserved_usd": _reserved_total(save_dir / "calls"),
            "budget_cap_usd": budget_usd,
            "jev_agreement": "not measured; the benchmark has no Jev labels",
            "calibration": "not applicable to the plain-text response schema",
        }
        save_json(save_dir / "summary.json", summary)
        if service is not None:
            service.close("success" if status == "completed" else "errored").result()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=ALL_CASE_IDS, default=list(PILOT_CASES))
    parser.add_argument("--save-dir", required=True, type=Path)
    parser.add_argument("--pricing-file", required=True, type=Path)
    parser.add_argument("--budget-usd", type=float, default=MAX_BUDGET_USD)
    parser.add_argument("--preflight", action="store_true", help="Check SDK renderers/model access; make no samples")
    args = parser.parse_args()
    try:
        if args.preflight:
            print(json.dumps(_run_preflight(args.pricing_file), indent=2))
            return 0
        summary = run_pilot(
            case_ids=args.cases, save_dir=args.save_dir, pricing_file=args.pricing_file,
            budget_usd=args.budget_usd,
        )
        print(json.dumps(summary, indent=2))
        return 0
    except Exception as exc:
        print(f"evaluation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

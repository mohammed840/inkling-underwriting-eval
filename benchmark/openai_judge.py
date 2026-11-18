"""Direct OpenAI transport for the existing strict judge; no retries or input truncation."""
import hashlib
import json
import os
import threading
import urllib.error
import urllib.request
from pathlib import Path
import grade
import run_eval as runtime

BUDGET_LOCK = threading.Lock()

MODEL = 'gpt-6-luna'
ENDPOINT = 'https://api.openai.com/v1/chat/completions'
# Conservative Standard-tier USD/M rates: input includes the 1.25x cache-write rate.
# Cached reads can be cheaper; retain this upper bound rather than assume a cache hit.
INPUT_PRICE, OUTPUT_PRICE = 0.125, 0.50
BUDGET_USD = 35.0  # User explicitly raised the cumulative cap to $35 on 2026-11-17.


def ensure_key():
    if not os.environ.get('OPENAI_API_KEY'):
        env_path = runtime.REPO_ROOT / '.env'
        for line in (env_path.read_text().splitlines() if env_path.exists() else []):
            if line.startswith('OPENAI_API_KEY='):
                os.environ['OPENAI_API_KEY'] = line.split('=', 1)[1].strip().strip('\"').strip("'")
    if not os.environ.get('OPENAI_API_KEY'):
        raise runtime.EvaluationError('Set OPENAI_API_KEY in environment or project .env')


def object_schema(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def payload(messages, criterion_id, model, reasoning_effort="none", output_tokens=None):
    if reasoning_effort not in ('none', 'low', 'medium', 'high'):
        raise ValueError('Unsupported experimental reasoning effort')
    if output_tokens is not None and not 1024 <= output_tokens <= 16384:
        raise ValueError('Output cap outside bounded experiment range')
    if model != MODEL:
        raise ValueError('This price-pinned pilot supports only ' + MODEL)
    props = {'id': {'type': 'string', 'enum': [criterion_id]},
             'score': {'type': 'integer', 'enum': [0, 1]}, 'reason': {'type': 'string'}}
    if criterion_id == 'grounding_scope':
        evidence = object_schema({
            'answer_passage_ids': {'type': 'array', 'items': {'type': 'string'}},
            'passage_ids': {'type': 'array', 'items': {'type': 'string'}},
            'supported': {'type': 'boolean'},
            'applicability': {'type': 'string', 'enum': ['applies', 'contrast', 'optional_advice', 'not_supported', 'not_a_rule']},
            'applicability_reason': {'type': 'string'}})
        props['evidence'] = {'type': 'array', 'items': evidence}
        data = json.loads(messages[1]['content'])
        if data.get('claim_id'):
            evidence['properties']['answer_passage_ids']['items']['enum'] = [data['claim_id']]
            ids = [p['passage_id'] for p in data['cited_passages']]
            if ids:
                evidence['properties']['passage_ids']['items']['enum'] = ids
    if criterion_id == 'claim_grounding':
        import reusable_grader
        props_schema = reusable_grader.schema(json.loads(messages[1]['content']))
    else:
        props_schema = object_schema(props)
    return {'model': model, 'messages': messages, 'stream': False,
            'max_completion_tokens': output_tokens if output_tokens is not None else (4096 if criterion_id == 'grounding_scope' else 1024),
            'reasoning_effort': reasoning_effort, 'service_tier': 'default', 'store': False,
            'response_format': {'type': 'json_schema', 'json_schema': {
                'name': 'criterion_grade', 'strict': True, 'schema': props_schema}}}


def sample(messages, criterion_id, model, trial, stem, context_tokens, reasoning_effort="none", output_tokens=None):
    ensure_key()
    request = payload(messages, criterion_id, model, reasoning_effort, output_tokens)
    encoded = json.dumps(request, ensure_ascii=False).encode()
    # UTF-8 byte count plus framing allowance conservatively bounds this text input.
    input_bound = len(encoded) + 4096
    if input_bound + request['max_completion_tokens'] > min(context_tokens, 272000):
        raise runtime.EvaluationError('Conservative context/standard-price guard exceeded; no truncation or call')
    reservation = (input_bound * INPUT_PRICE + request['max_completion_tokens'] * OUTPUT_PRICE) / 1e6
    digest = hashlib.sha256(encoded).hexdigest()
    path = trial / 'calls' / (stem + '.json')
    if path.exists():
        row = json.loads(path.read_text())
        if row['request_hash'] != digest or row['status'] != 'completed':
            raise runtime.EvaluationError('Changed or uncertain request; refusing replay')
        response = row['response']
    else:
        with BUDGET_LOCK:
            spent = sum(json.loads(p.read_text())['reservation_usd'] for p in (trial / 'calls').glob('*.json'))
            if spent + reservation > BUDGET_USD:
                raise runtime.EvaluationError(f'Pilot ${BUDGET_USD:g} budget would be exceeded')
            row = {'status': 'uncertain', 'request_hash': digest, 'request': request,
                   'role': 'judge', 'initial_reservation_usd': reservation, 'reservation_usd': reservation,
                   'input_token_upper_bound': input_bound}
            runtime.save_json(path, row)  # Durable reservation BEFORE network; never retry uncertain calls.
        req = urllib.request.Request(ENDPOINT, data=encoded,
            headers={'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'], 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=180) as result:
                response = json.load(result)
        except (OSError, ValueError) as exc:
            # Keep the full uncertain reservation. Never persist exception text,
            # response bodies or headers, which may contain credentials.
            row['transport_error'] = {'type': type(exc).__name__}
            if isinstance(exc, urllib.error.HTTPError):
                row['transport_error']['http_status'] = exc.code
            with BUDGET_LOCK:
                runtime.save_json(path, row)
            raise
        row['response'] = response
        with BUDGET_LOCK:
            runtime.save_json(path, row)
        usage = response.get('usage', {})
        if 'prompt_tokens' not in usage or 'completion_tokens' not in usage or response.get('error'):
            raise runtime.EvaluationError('Missing usage or API error; reservation retained, no retry')
        estimate = (usage['prompt_tokens'] * INPUT_PRICE + usage['completion_tokens'] * OUTPUT_PRICE) / 1e6
        # Direct OpenAI usage has no invoice cost; keep a conservative token-based bound.
        if estimate > row['initial_reservation_usd'] + 1e-9:
            raise runtime.EvaluationError('Usage exceeded reservation; retain bound and stop')
        row.update(status='completed', prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'],
                   estimated_cost_usd=estimate, cost_basis='standard_with_max_cache_write_rate',
                   provider_reported_cost_usd=None, reservation_usd=estimate,
                   reservation_settled_from_usage=True)
        with BUDGET_LOCK:
            runtime.save_json(path, row)
    choice = response['choices'][0]
    message = choice['message']
    text = message.get('content') or ''
    clean = choice.get('finish_reason') == 'stop' and not message.get('tool_calls') and not message.get('refusal')
    runtime.save_json(trial / 'responses' / (stem + '.json'), {'text': text, 'stop_reason': choice.get('finish_reason'), 'clean': clean})
    if not clean:
        raise grade.GradeError('Completed judge output is truncated, refused or malformed')
    return text

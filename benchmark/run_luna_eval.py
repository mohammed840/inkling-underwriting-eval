"""Fresh Inkling conversations through the existing BenchmarkBuilder, with direct OpenAI Luna."""
import argparse
import asyncio
import json
import math
import shutil
from pathlib import Path
import build
import individual_judge as judge
import reusable_grader as grounding
import grade
import openai_judge as luna
import registered_eval as registered
import run_eval as runtime

ROOT = Path(__file__).resolve().parent
CASES = list(runtime.ALL_CASE_IDS)
BUDGET = 40.0
CANDIDATE = 'thinkingmachines/Inkling:peft:262144'
TOKENIZER = 'thinkingmachines/Inkling'
SEED = 20261117


def prior_spend(ledger):
    paths = sorted(ledger.glob('*.json'))
    if not paths:
        raise runtime.EvaluationError('Prior ledger is missing or empty; refusing to reset cumulative spend')
    amounts = [json.loads(p.read_text())['reservation_usd'] for p in paths]
    if any(type(x) not in (int, float) or not math.isfinite(x) or x < 0 for x in amounts):
        raise runtime.EvaluationError('Invalid prior cost reservation')
    return sum(amounts), {p.name: runtime.sha256_file(p) for p in paths}


def summarize(record, status):
    trial = record / 'trial'
    cases = [json.loads(p.read_text()) for p in (trial / 'cases').glob('*.json')]
    turns = [json.loads(p.read_text()) for p in (trial / 'turns').glob('*.json')]
    calls = [json.loads(p.read_text()) for p in (trial / 'calls').glob('*.json')]
    states = {c: 'not_completed' for c in CASES}
    for c in cases: states[c['case_id']] = 'unresolved' if c['passed'] is None else 'pass' if c['passed'] else 'fail'
    passed = sum(v=='pass' for v in states.values()); unresolved = sum(v in ('unresolved','not_completed') for v in states.values())
    result = {'status': status, 'cases': states, 'attempts_per_case': 1, 'unit':'three-user-turn conversation',
        'pass@1': passed/len(CASES) if not unresolved else None,
        'confirmed_pass_fraction': passed/len(CASES), 'pass_at_1_bounds': [passed/len(CASES),(passed+unresolved)/len(CASES)],
        'unresolved_cases':unresolved,'turns_total':len(turns),
        'turn_statuses':{k:sum(t.get('status')==k for t in turns) for k in ['graded','judge_error','insufficient_evidence','candidate_parse_error','pending_grade']},
        'candidate_calls':sum(c.get('role')=='candidate' for c in calls),
        'luna_calls':sum(c['request']['model']==luna.MODEL for c in calls),
        'qwen_calls':sum(c['request']['model']==runtime.JUDGE for c in calls),
        'candidate_estimated_cost_usd':sum(c.get('estimated_cost_usd',0) for c in calls if c.get('role')=='candidate'),
        'luna_estimated_cost_usd':sum(c.get('estimated_cost_usd',0) for c in calls if c.get('role')=='judge'),
        'conservative_spend_and_reservations_usd':sum(c['reservation_usd'] for c in calls),'budget_usd':BUDGET,'prior_bound_usd':sum(c['reservation_usd'] for c in calls if c.get('role')=='carry'),
        'calibration':'not measured; no probability predictions','jev_agreement':'not measured; no Jev labels',
        'framework_note':'Framework zero rewards for unresolved cases are placeholders; use this summary with explicit unresolved outcomes.'}
    runtime.save_json(record/'summary.json',result);return result


def summarize_four(record, summaries):
    states = {}
    for case_id in CASES:
        outcomes = [s['cases'].get(case_id) for s in summaries]
        if 'pass' in outcomes:
            states[case_id] = 'pass'
        elif outcomes == ['fail'] * 4:
            states[case_id] = 'fail'
        else:
            states[case_id] = 'unresolved'
    complete = len(summaries) == 4 and all(s['status'] == 'completed' and s['turns_total'] == 45 for s in summaries)
    count = sum(s == 'pass' for s in states.values())
    result = {'cases': states, 'attempts_per_case': 4, 'attempt_batches_recorded': len(summaries),
              'status': 'completed' if complete else 'incomplete',
              'pass@4': count / len(CASES) if complete and 'unresolved' not in states.values() else None}
    calls = [json.loads(path.read_text()) for path in (record / 'attempts').glob('*/trial/calls/*.json')]
    billed = [call for call in calls if call.get('role') in ('candidate', 'judge')]
    result['completed_calls_estimated_cost_usd'] = sum(call.get('estimated_cost_usd', 0) for call in billed)
    result['run_spend_and_reservations_usd'] = sum(call['reservation_usd'] for call in billed)
    runtime.save_json(record / 'summary.json', result)
    return result


async def run(record, execute=False, *, prior_ledger, attempts=1, seed=SEED, check_parent=None):
    if attempts not in (1, 4):
        raise ValueError('Supported attempt counts are 1 and 4')
    prior_ledger = prior_ledger.resolve()
    if prior_ledger == (record / 'trial/calls').resolve():
        raise runtime.EvaluationError('Prior ledger must be separate from this new trial')
    prior_bound, prior_files = prior_spend(prior_ledger)
    if prior_bound >= BUDGET:
        raise runtime.EvaluationError('Cumulative cap already reached')
    def check_prior_unchanged():
        if check_parent is not None:
            check_parent()
        if prior_spend(prior_ledger) != (prior_bound, prior_files):
            raise runtime.EvaluationError('Prior ledger changed; stop concurrent paid work and prepare a new run')
    from tinker_cookbook.model_info import get_recommended_renderer_name
    renderer_name=get_recommended_renderer_name(TOKENIZER)
    luna.BUDGET_USD=BUDGET
    cases_all={c['id']:c for c in build.load_cases()};cases=[cases_all[c] for c in CASES]
    corpus=json.loads((ROOT/'corpus.json').read_text())
    prices=runtime.load_prices(record/'pricing.json',(CANDIDATE,))[CANDIDATE]
    names=['run_luna_eval.py','registered_eval.py','run_eval.py','individual_judge.py','reusable_grader.py','openai_judge.py','grade.py','build.py','tool_env.py','manual_tools.py','cases.json','corpus.json','source-scope.json']
    files={n:runtime.sha256_file(ROOT/n) for n in names}
    files.update({n:runtime.sha256_file(record/n) for n in ['pricing.json','PROTOCOL.md']})
    freeze={'cases':CASES,'candidate':CANDIDATE,'candidate_renderer':renderer_name,
            'temperature':runtime.CANDIDATE_TEMP,'seed':seed,'output_cap':runtime.OUTPUT_TOKEN_CAP,
            'candidate_input_cap':registered.CANDIDATE_INPUT_CONTEXT_CAP,'judge':luna.MODEL,
            'judge_reasoning':'medium','judge_output_tokens':4096,'prior_bound_usd':prior_bound,'prior_ledger':str(prior_ledger),'prior_ledger_files':prior_files,'judge_contract':'reusable_three_way_claim_v1','attempts_per_case':attempts,'budget_usd':BUDGET,'files':files}
    if attempts == 4:
        freeze['attempt_seeds'] = list(range(seed, seed + attempts))
    record.mkdir(exist_ok=True);(record/'sources').mkdir(exist_ok=True)
    if (record/'freeze.json').exists():assert json.loads((record/'freeze.json').read_text())==freeze
    else:
        for n in names:shutil.copy2(ROOT/n,record/'sources'/n)
        runtime.save_json(record/'freeze.json',freeze)
    print(f'Frozen 15 cases; {attempts} fresh attempt(s) each; {CANDIDATE} + Luna; ${BUDGET:g} cumulative cap.',flush=True)
    if not execute:return
    if attempts == 4:
        batches = []
        ledger = prior_ledger
        # Exclusive start prevents restarting or overwriting a partially paid four-attempt run.
        (record / 'attempts').mkdir()
        try:
            for index in range(4):
                check_prior_unchanged()
                for name, digest in files.items():
                    path = record / name if name in ('pricing.json', 'PROTOCOL.md') else ROOT / name
                    if runtime.sha256_file(path) != digest:
                        raise runtime.EvaluationError('Frozen pass@4 inputs changed')
                batch = record / 'attempts' / str(index + 1)
                batch.mkdir()
                for name in ('pricing.json', 'PROTOCOL.md'):
                    shutil.copy2(record / name, batch / name)
                await run(batch, True, prior_ledger=ledger, seed=seed + index, check_parent=check_prior_unchanged)
                summary = json.loads((batch / 'summary.json').read_text())
                batches.append(summary)
                if summary['status'] != 'completed' or summary['turns_total'] != 45:
                    break
                # The finished ledger includes its carry, so each new batch inherits the full spend.
                ledger = batch / 'trial' / 'calls'
        finally:
            summarize_four(record, batches)
        return
    trial=record/'trial';trial.mkdir(exist_ok=True)
    if any((trial/'cases').glob('*.json')) or any((trial/'turns').glob('*.json')):
        raise runtime.EvaluationError('Partial or completed conversations exist; do not resample attempts')
    runtime.save_json(trial/'calls'/'prior-study-bound.json', {'status':'completed','role':'carry','request':{'model':'prior-study'},'reservation_usd':prior_bound})
    runtime.ensure_tinker_token();luna.ensure_key()
    import tinker
    from tinker.lib.retry_handler import RetryConfig
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer
    from tool_env import TinkerManualTools
    renderer=get_renderer(renderer_name,get_tokenizer(TOKENIZER),model_name=CANDIDATE)
    service=tinker.ServiceClient(display_name='inkling-full-luna-fifteen')
    client=service.create_sampling_client(base_model=CANDIDATE,retry_config=RetryConfig(enable_retry_logic=False,progress_timeout=300)).result()
    runtime.disable_sample_submission_retries(service)
    runtime.save_json(record/'session.json',{'url':service.get_console_url(),'candidate_model':CANDIDATE,'tinker_judge_client_created':False})
    runtime.save_json(trial/'corpus.json',corpus)
    tool_owner=TinkerManualTools(trial/'corpus.json');tools=[tool_owner.search_manual,tool_owner.read_section]
    abort={}
    def sample_fn(*,role,case_id,turn_index,call_index,prompt):
        assert role=='candidate'
        if abort:raise registered.FatalPaidRequestAbort(str(abort))
        try:
            check_prior_unchanged()
            if any(json.loads(p.read_text())['status']!='completed' for p in (trial/'calls').glob('*.json')):
                raise runtime.EvaluationError('Uncertain prior request')
            return runtime.sample_once(client=client,tinker=tinker,model=CANDIDATE,
                renderer_name=renderer_name,role='candidate',case_id=case_id,turn_index=turn_index,
                prompt=prompt,stop_sequences=renderer.get_stop_sequences(),temperature=runtime.CANDIDATE_TEMP,seed=seed,
                prices=prices,budget_usd=BUDGET,save_dir=trial,include_logprobs=True,
                output_token_cap=runtime.OUTPUT_TOKEN_CAP,model_context_tokens=262144,input_context_cap=registered.CANDIDATE_INPUT_CONTEXT_CAP)
        except Exception as exc:
            abort['reason']=str(exc)
            cause=exc.__cause__
            if cause is not None and type(cause).__name__=='RequestFailedError':
                runtime.save_json(record/'provider-error.json',{'message':cause.message,'category':str(cause.category)})
            raise registered.FatalPaidRequestAbort(str(exc)) from exc
    def turn_grader(case,turn_index,prior_answers,answer):
        def ask(messages,criterion_id):
            data=json.loads(messages[1]['content']);claim_id=data.get('claim',{}).get('id')
            stem=f"{case['id']}-turn-{turn_index}-{criterion_id}"+('-'+claim_id.rsplit(':',1)[-1] if claim_id else '')
            def parse(text):
                return grounding.parse(text,data) if claim_id else judge.parse_result(text,{'id':criterion_id},answer,[],case)
            for attempt in (1,2):
                attempt_stem=stem+f'-attempt-{attempt}'
                runtime.save_json(trial/'prompts'/(attempt_stem+'.json'),messages)
                try:
                    check_prior_unchanged()
                    text=luna.sample(messages,criterion_id,luna.MODEL,trial,attempt_stem,272000,'medium',4096)
                    parse(text)
                    return text
                except luna.urllib.error.HTTPError as exc:
                    if exc.code in (401,402,403):
                        raise registered.FatalPaidRequestAbort(f'OpenAI HTTP {exc.code}; stopped') from exc
                    if attempt==2:raise
                except (OSError,grade.GradeError,json.JSONDecodeError):
                    if attempt==2:raise
        ask.claim_workers=1
        try:
            saved=json.loads((trial/'turns'/f"{case['id']}-turn-{turn_index}.json").read_text())
            result=judge.grade_answer(case=case,turn_index=turn_index,prior_answers=prior_answers,answer=answer,corpus=corpus,ask=ask,message_history=saved['actual_message_history'])
            print(case['id'],turn_index,result['status'],result.get('passed'),flush=True)
            return result
        except Exception as exc:
            abort['reason']=str(exc)
            cause=exc.__cause__
            if cause is not None and type(cause).__name__=='RequestFailedError':
                runtime.save_json(record/'provider-error.json',{'message':cause.message,'category':str(cause.category)})
            raise registered.FatalPaidRequestAbort(str(exc)) from exc
    proxy=registered.CandidateSamplingProxy(sample_fn,renderer,abort)
    builder=registered.UnderwritingBenchmark(cases=cases,corpus=corpus,save_dir=trial,candidate_renderer=renderer,
        judge_renderer=None,tools=tools,sample_proxy=proxy,judge_client=None,sample_fn=sample_fn,turn_grader=turn_grader)
    registered.register(builder)
    config=registered.BenchmarkConfig(max_tokens=runtime.OUTPUT_TOKEN_CAP,temperature=runtime.CANDIDATE_TEMP,
        context_window=None,timeout_seconds=1800,concurrency=1,agent_concurrency=1,
        max_trajectory_tokens=registered.MAX_TRAJECTORY_TOKENS,max_generation_tokens=runtime.OUTPUT_TOKEN_CAP,
        num_samples=1,save_dir=str(trial))
    runtime.save_json(record/'framework-config.json',{'benchmark':registered.BENCHMARK_NAME,'judge_via_environment':True,'concurrency':1,'num_samples':1,'timeout_seconds':1800,'seed_enforced_in_transport':seed})
    status='completed'
    try:
        await registered.run_benchmark(registered.BENCHMARK_NAME,proxy,renderer,config)
    except registered.FatalPaidRequestAbort as exc:
        status='stopped';runtime.save_json(record/'error.json',{'error':str(exc)})
    except BaseException as exc:
        status='errored';runtime.save_json(record/'error.json',{'error':str(exc),'type':type(exc).__name__});raise
    finally:
        summarize(record,status)
        service.close('success' if status=='completed' else 'errored').result()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('record',type=Path);parser.add_argument('--run',action='store_true');parser.add_argument('--prior-ledger',type=Path,required=True,help='Authoritative prior trial/calls directory; includes settled costs and uncertain reservations')
    parser.add_argument("--attempts", type=int, choices=(1, 4), default=1)
    args=parser.parse_args();asyncio.run(run(args.record,args.run,prior_ledger=args.prior_ledger,attempts=args.attempts))

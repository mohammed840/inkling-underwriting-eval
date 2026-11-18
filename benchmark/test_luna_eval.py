import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import build
from registered_eval import UnderwritingMessageEnv


class LunaEnvironmentTest(unittest.TestCase):
    def test_four_attempts_require_complete_conversations(self):
        from run_luna_eval import CASES, summarize_four
        batches = [dict(status='completed', turns_total=45,
                        cases=dict.fromkeys(CASES, 'fail')) for _ in range(4)]
        batches[3]['cases'][CASES[0]] = 'pass'
        with tempfile.TemporaryDirectory() as folder:
            score = lambda rows: summarize_four(Path(folder), rows)['pass@4']
            self.assertEqual(score(batches), 1 / 15)
            self.assertIsNone(score(batches[:3]))
            batches[0]['cases'][CASES[1]] = 'unresolved'
            self.assertIsNone(score(batches))

    def test_prior_ledger_counts_uncertain_cost_and_rejects_empty(self):
        from run_luna_eval import prior_spend
        from run_eval import EvaluationError
        with tempfile.TemporaryDirectory() as folder:
            ledger = Path(folder)
            with self.assertRaises(EvaluationError): prior_spend(ledger)
            (ledger / 'settled.json').write_text(json.dumps({'reservation_usd': 24.4}))
            (ledger / 'uncertain.json').write_text(json.dumps({'status': 'uncertain', 'reservation_usd': 0.1}))
            amount, hashes = prior_spend(ledger)
            self.assertAlmostEqual(amount, 24.5)
            self.assertEqual(len(hashes), 2)
            (ledger / 'uncertain.json').write_text(json.dumps({'reservation_usd': -1}))
            with self.assertRaises(EvaluationError): prior_spend(ledger)

    def test_external_judge_sees_actual_turns_and_preserves_unresolved_case(self):
        with tempfile.TemporaryDirectory() as folder:
            seen=[]
            def grader(case,index,prior,answer):
                seen.append((index,prior,answer))
                return {'status':'insufficient_evidence','passed':None,'reward':None,'criteria':[]}
            proxy=SimpleNamespace(abort_state={},set_context=lambda *args:None)
            renderer=SimpleNamespace(create_conversation_prefix_with_tools=lambda **kwargs:[])
            env=UnderwritingMessageEnv(case=build.load_cases()[0],corpus={},candidate_renderer=renderer,
                judge_renderer=None,tools=[],sample_proxy=proxy,judge_client=None,sample_fn=None,
                save_dir=Path(folder),turn_grader=grader)
            async def episode():
                await env.initial_observation()
                for i in range(3):await env.step({'role':'assistant','content':f'actual answer {i+1}'})
            asyncio.run(episode())
            self.assertEqual(seen[2],(3,['actual answer 1','actual answer 2'],'actual answer 3'))
            result=json.loads((Path(folder)/'cases/uw-01.json').read_text())
            self.assertIsNone(result['passed'])
            self.assertEqual(result['status'],'unresolved')

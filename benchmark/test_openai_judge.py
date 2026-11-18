import io
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import openai_judge as judge
import run_eval


class OpenAIJudgeTest(unittest.TestCase):
    def test_transport_failure_logged_without_releasing_reservation(self):
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}):
            error = judge.urllib.error.HTTPError('https://api.openai.com', 402, 'secret diagnostic', {},
                io.BytesIO(b'{"error":{"message":"secret diagnostic","metadata":{"limit_source":"openrouter_credits"}}}'))
            with patch('urllib.request.urlopen', side_effect=error):
                with self.assertRaises(judge.urllib.error.HTTPError):
                    judge.sample([], 'decision', judge.MODEL, Path(d), 'timeout', 272000)
            saved = json.loads((Path(d) / 'calls/timeout.json').read_text())
            self.assertEqual(saved['transport_error'], {'type': 'HTTPError', 'http_status': 402})
            self.assertEqual(saved['status'], 'uncertain')
            self.assertEqual(saved['reservation_usd'], saved['initial_reservation_usd'])
            self.assertNotIn('secret diagnostic', json.dumps(saved))

    def test_saved_response_reused_and_budget_blocks_network(self):
        response = {'choices': [{'finish_reason': 'stop', 'message': {'content': '{"id":"decision","score":1,"reason":"ok"}'}}],
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 12}}
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {'OPENAI_API_KEY': 'test'}):
            trial = Path(d)
            with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as net:
                first = judge.sample([], 'decision', judge.MODEL, trial, 'a', 1050000, 'medium', 4096)
                self.assertEqual(first, judge.sample([], 'decision', judge.MODEL, trial, 'a', 1050000, 'medium', 4096))
                self.assertEqual(net.call_count, 1)
                saved = json.loads((trial / 'calls' / 'a.json').read_text())
                self.assertAlmostEqual(saved['reservation_usd'], 0.00000725)
                sent = net.call_args.args[0]
                payload = json.loads(sent.data)
                self.assertEqual(sent.full_url, judge.ENDPOINT)
                self.assertEqual(payload['model'], 'gpt-6-luna')
                self.assertEqual(payload['reasoning_effort'], 'medium')
                self.assertEqual(payload['max_completion_tokens'], 4096)
                self.assertEqual(payload['service_tier'], 'default')
                self.assertTrue(payload['response_format']['json_schema']['strict'])
                self.assertFalse({'provider', 'transforms', 'max_tokens', 'reasoning'} & payload.keys())
                self.assertIsNone(saved['provider_reported_cost_usd'])
                self.assertGreater(saved['initial_reservation_usd'], saved['reservation_usd'])
                run_eval.save_json(trial / 'calls' / 'spent.json', {'reservation_usd': judge.BUDGET_USD})
                with self.assertRaisesRegex(run_eval.EvaluationError, 'budget'):
                    judge.sample([], 'decision', judge.MODEL, trial, 'b', 1050000)
                self.assertEqual(net.call_count, 1)

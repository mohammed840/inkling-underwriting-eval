import hashlib
import json
from pathlib import Path
import unittest
import build
import grade
import individual_judge as judge
import reusable_grader as reusable


class IndividualJudgeTests(unittest.TestCase):
    def test_reusable_verdicts_and_adapter_preserve_context_and_coverage(self):
        case = build.load_cases()[0]
        case['id'] = 'unseen-case-id'
        corpus = json.loads(Path(__file__).with_name('corpus.json').read_text())
        answer = '**Checks**\n\n1. Conditions still need verification: the property is not vacant; it is not under renovation. [tx:p003]'
        prompts = []
        def ask(messages, criterion_id):
            data = json.loads(messages[1]['content']); prompts.append(data)
            if criterion_id != 'claim_grounding':
                self.assertTrue(data['scope'].endswith(judge.ASSESSMENT_BOUNDARY))
                self.assertEqual(data['candidate_current_answer'], answer)
                self.assertEqual(data['criterion']['id'], criterion_id)
                self.assertNotIn('source_text', data)
                self.assertNotIn('cited_passages', data)
                self.assertNotIn('All claims must', messages[0]['content'])
                return json.dumps({'id': criterion_id, 'score': 1, 'reason': 'test'})
            self.assertEqual(data['target_program'], case['target_program'])
            self.assertTrue(any(x['content'] == 'ACTUAL PRIOR ANSWER' for x in data['context']))
            self.assertNotIn('case_id', data)
            self.assertTrue(any(answer in x['content'] for x in data['context']))
            self.assertTrue(any(case['turns'][1]['rubric_items'][0]['criterion'] in x['content'] for x in data['context']))
            self.assertTrue(any('Benchmark owner-defined assessment boundary: assess supplied evidence only.' in x['content'] for x in data['context']))
            self.assertTrue(any(e['id'].startswith('tx:p004:') for e in data['evidence']))
            self.assertTrue(any(e['id'].startswith('tx:p011:') for e in data['evidence']))
            return json.dumps({'claim_id': data['claim']['id'], 'verdict': 'supported',
                'evidence_ids': [data['evidence'][0]['id']], 'reason': 'test'})
        result = judge.grade_answer(case=case, turn_index=2, prior_answers=['ACTUAL PRIOR ANSWER'], answer=answer, corpus=corpus, ask=ask, message_history=[{'role': 'tool', 'content': '{"section_id":"tx:p011"}'}])
        self.assertTrue(result['passed'])
        claims = judge.claim_catalog(answer)
        self.assertEqual(len(result['criteria'][-1]['claim_results']), len(claims))
        for c in claims:
            self.assertEqual(c['text'], answer[c['start']:c['end']])
            self.assertNotEqual(c['text'], '1.')
        self.assertTrue(any('Conditions still need verification' in c['text'] and 'not vacant' in c['text'] for c in claims))
        enumerated = 'b. Named Storm. (Optional)\n          c. Vandalism. (Optional)\n          d. Sprinkler Leakage.'
        spans = judge.claim_catalog(enumerated)
        self.assertTrue(any(c['text'].startswith('c. Vandalism') for c in spans))
        self.assertFalse(any('Named Storm' in c['text'] and 'c.' in c['text'] for c in spans))
        item = next(x for x in prompts if 'claim' in x)
        for verdict in reusable.VERDICTS:
            row = reusable.parse(json.dumps({'claim_id': item['claim']['id'], 'verdict': verdict, 'evidence_ids': [], 'reason': 'test'}), item)
            self.assertEqual(row['verdict'], verdict)
        self.assertNotIn('not_a_rule', reusable.SYSTEM)
        whole = 'Nothing in the pinned source requires this.'
        all_sections = [s for s in corpus['sections'] if s['source_id'] == 'tx']
        packet = judge.claim_input(case, 1, [], judge.claim_catalog(whole)[0], all_sections[:1], whole, {}, all_sections)
        self.assertTrue(any(e['id'].startswith('tx:p012:') for e in packet['evidence']))
        ma = [s for s in corpus['sections'] if s['source_id'] == 'ma']
        compact = judge.claim_input(build.load_cases()[9], 1, [], judge.claim_catalog(whole)[0], ma[:1], whole, {}, ma)
        original = judge.passage_catalog(ma)
        self.assertEqual('\n\n'.join(p['text'] for p in original), '\n\n'.join(p['text'] for p in compact['evidence']))
        self.assertEqual(len({p['id'].split(':b')[0] for p in compact['evidence']}), len(ma))
        affirmative = 'The proposal mandates automatic decline.'
        az = [s for s in corpus['sections'] if s['source_id'] == 'allianz']
        packet = judge.claim_input(build.load_cases()[11], 3, [], judge.claim_catalog(affirmative)[0], az[:1], affirmative, {}, az)
        self.assertEqual('\n\n'.join(p['text'] for p in judge.passage_catalog(az)), '\n\n'.join(p['text'] for p in packet['evidence']))
        self.assertEqual({p['id'].split(':b')[0] for p in packet['evidence']}, {p['section_id'] for p in judge.passage_catalog(az)})
        metadata = json.loads(next(c['content'].split(': ', 1)[1] for c in packet['context']
                                  if c['content'].startswith('Harness record metadata')))
        self.assertEqual(metadata['provided_sections_with_empty_pinned_text'], [
            {'section_id': s['section_id'], 'sha256': hashlib.sha256(s['text'].encode()).hexdigest()}
            for s in az if not s['text'].strip()])
        self.assertTrue(metadata['provided_sections_with_empty_pinned_text'])
        # The verifier's whole source must not become the candidate's retrieval history.
        visible = {'tool_history_available': True, 'retrieved_section_ids': [az[0]['section_id']]}
        packet = judge.claim_input(build.load_cases()[11], 1, [], judge.claim_catalog(affirmative)[0],
                                   az[:1], affirmative, visible, az)
        meta = json.loads(next(c['content'].split(': ', 1)[1] for c in packet['context']
                              if c['content'].startswith('Harness record metadata')))
        self.assertTrue(meta['whole_pinned_source_provided'])
        expected = [e['id'] for e in packet['evidence'] if e['id'].split(':b')[0] == az[0]['section_id']]
        self.assertEqual(meta['candidate_visible_evidence_ids'], expected)
        self.assertLess(len(expected), len(packet['evidence']))
        # Missing history cannot be treated as proof that no source was retrieved.
        self.assertIsNone(metadata['candidate_visible_evidence_ids'])


    def test_unknown_ids_changed_evidence_and_missing_content_rejected(self):
        item = {'claim': {'id': 'x', 'text': 'Claim'}, 'target_program': 'new-program', 'context': [],
                'evidence': [{'id': 'e', 'text': 'Rule', 'programs': ['new-program'], 'sha256': hashlib.sha256(b'Rule').hexdigest()}]}
        with self.assertRaises(grade.GradeError):
            reusable.parse(json.dumps({'claim_id': 'x', 'verdict': 'supported', 'evidence_ids': ['invented'], 'reason': 'test'}), item)
        item['evidence'][0]['text'] = 'Changed rule'
        with self.assertRaises(ValueError): reusable.messages(item)
        with self.assertRaises(grade.GradeError): judge.claim_catalog('  ')
        self.assertFalse(judge.quote_matches('companies may decline', 'companies must decline'))


if __name__ == '__main__':
    unittest.main()

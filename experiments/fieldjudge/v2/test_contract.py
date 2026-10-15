"""Offline checks: IDs, evidence, and fail-closed subscription accounting."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from common import validate, validate_response, presented
import allowance

class ContractTests(unittest.TestCase):
    def setUp(self):
        self.cases=[dict(id="long-a", topic="x",definition="x",text="Title: Small animal’s cancer risk"),
                    dict(id="long-b", topic="y",definition="y",text="Title: Migration patterns")]
    def verdicts(self):
        return [dict(id="p01",evidence="Small animal's cancer risk",reason="Subject is central.",label="on_topic",relation="central"),
                dict(id="p02",evidence="Migration patterns",reason="Different subject.",label="off_topic",relation="unrelated")]
    def test_reordered_results_matched_by_id(self):
        out=validate_response(self.cases,list(reversed(self.verdicts())))
        self.assertEqual([x['id'] for x in out],['long-a','long-b'])
        self.assertTrue(all(v['evidence_verified'] for v in out))
    def test_missing_id_fails(self):
        with self.assertRaises(ValueError): validate_response(self.cases,self.verdicts()[:1])
    def test_duplicate_id_fails(self):
        v=self.verdicts(); v[1]['id']='p01'
        with self.assertRaises(ValueError): validate_response(self.cases,v)
    def test_unknown_id_fails(self):
        v=self.verdicts(); v[1]['id']='p03'
        with self.assertRaises(ValueError): validate_response(self.cases,v)
    def test_invented_evidence_is_quarantined(self):
        v=self.verdicts(); v[0]['evidence']='cancer is never risky'
        out=validate_response(self.cases,v)
        self.assertFalse(out[0]['evidence_verified'])
        with self.assertRaises(ValueError): validate(presented(self.cases),v)
    def test_relation_mismatch_fails(self):
        v=self.verdicts(); v[0]['relation']='unrelated'
        with self.assertRaises(ValueError): validate_response(self.cases,v)
    def test_provider_failure_stops(self):
        with patch.object(allowance,'get_usage',side_effect=TimeoutError):
            with self.assertRaises(allowance.Pause): allowance.check('sol',force=True)
    def test_usage_rate_limit_waits_without_allowing_calls(self):
        import urllib.error
        error = urllib.error.HTTPError('usage',429,'limited',{},None)
        status={'rate_limit':{'limit_reached':False,'primary_window':{'used_percent':4}},'credits':{'balance':0}}
        with tempfile.TemporaryDirectory() as d, patch.object(allowance,'ROOT',Path(d)), \
             patch.object(allowance,'get_usage',side_effect=[error,status]) as get, patch.object(allowance.time,'sleep') as sleep:
            allowance.check('sol',force=True)
            sleep.assert_called_once_with(300.0)
            self.assertEqual(get.call_count,2)
    def test_checks_all_windows_and_persists_credit_floor(self):
        with tempfile.TemporaryDirectory() as d, patch.object(allowance,'ROOT',Path(d)):
            status={'rate_limit':{'limit_reached':False,'primary_window':{'used_percent':4},'secondary_window':{'used_percent':12}},
                    'credits':{'balance':10}}
            with patch.object(allowance,'get_usage',return_value=status):
                allowance.check('sol',force=True)
                status['rate_limit']['secondary_window']['used_percent']=86
                with self.assertRaises(allowance.Pause): allowance.check('sol',force=True)
                status['rate_limit']['secondary_window']['used_percent']=12
                status['credits']['balance']=9
                with self.assertRaises(allowance.Pause): allowance.check('sol',force=True)
                self.assertEqual(json.loads((Path(d)/'credits-start-sol.json').read_text())['credits'],10)

if __name__=='__main__': unittest.main()

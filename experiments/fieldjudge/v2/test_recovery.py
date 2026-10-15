"""No network/model calls: transient errors retry; data/account gates do not."""
import tempfile
import unittest
import urllib.error
from pathlib import Path
from control import RETRY, BLOCKED, transient, provider_failure, TemporaryFailure, NeedsAttention
from run_stage import execute

class RecoveryTests(unittest.TestCase):
    def test_known_transient_provider_errors(self):
        for message in ('TransportError: connection closed mid-chunk', 'TimeoutError: deadline',
                        'RateLimitError: 429', 'ServerError: 503'):
            self.assertTrue(transient(message))
            with self.assertRaises(TemporaryFailure): provider_failure(message, 'probe')
    def test_safety_and_quality_errors_never_retry(self):
        for message in ('Refusal: content_filter', 'credit balance fell', 'allowance guard: 90%',
                        'AuthError: token expired', 'parse failed', 'screen IDs mismatch',
                        'Refusal with nested TransportError'):
            self.assertFalse(transient(message))
            with self.assertRaises(NeedsAttention): provider_failure(message, 'probe')
    def test_usage_endpoint_network_error_retries_but_auth_does_not(self):
        self.assertTrue(transient('usage unavailable (HTTPError, code=503)'))
        self.assertFalse(transient('usage unavailable (HTTPError, code=401)'))
        self.assertTrue(transient(urllib.error.HTTPError('usage',429,'slow down',{},None)))
        self.assertFalse(transient(urllib.error.HTTPError('usage',403,'forbidden',{},None)))
    def stage(self, text):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'probe.py';p.write_text(text)
            return execute([str(p)])
    def test_wrapped_stage_exit_codes(self):
        self.assertEqual(self.stage('raise ConnectionResetError()')[0], RETRY)
        self.assertEqual(self.stage('raise ValueError("bad evidence")')[0], BLOCKED)
        self.assertEqual(self.stage('raise SystemExit(1)')[0], BLOCKED)
        self.assertEqual(self.stage('pass')[0], 0)

if __name__ == '__main__': unittest.main()

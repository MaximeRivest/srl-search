"""Fast path still enforces identities and idempotent queueing, without model calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import stream_queue
from fast import short_labels, request

class FastTests(unittest.TestCase):
    def rows(self):
        return [dict(id='long-a',work_id=1,topic='bats',definition='flying mammals',text='Title: Bat migration'),
                dict(id='long-b',work_id=2,topic='bats',definition='flying mammals',text='Title: Cricket bat construction')]
    def test_reordering_is_harmless(self):
        actual=short_labels(self.rows(),[{'id':'p02','label':'off_topic'},{'id':'p01','label':'on_topic'}])
        self.assertEqual(actual,[{'id':'long-a','label':'on_topic'},{'id':'long-b','label':'off_topic'}])
    def test_missing_or_duplicate_ids_fail(self):
        for v in ([{'id':'p01','label':'on_topic'}], [{'id':'p01','label':'on_topic'}]*2):
            with self.assertRaises(ValueError):short_labels(self.rows(),v)
    def test_no_model_scores_or_known_labels_in_request(self):
        rows=[{**r,'kimi':1,'mining_scores':{'m2b':.9}} for r in self.rows()]
        req=request({'pairs':rows})
        self.assertEqual(set(req),{'topic','definition','papers'})
        self.assertEqual(set(req['papers'][0].model_dump()),{'id','text'})
    def test_mixed_topics_refused(self):
        rows=self.rows();rows[1]['topic']='cricket'
        with self.assertRaises(ValueError):request({'pairs':rows})
    def test_queue_resumes_without_duplicates(self):
        with tempfile.TemporaryDirectory() as d,patch.object(stream_queue,'FAST',Path(d)):
            stream_queue.enqueue(self.rows(),'test')
            first={p.name:p.read_bytes() for p in (Path(d)/'queue').glob('*.json')}
            stream_queue.enqueue(self.rows(),'test')
            self.assertEqual(first,{p.name:p.read_bytes() for p in (Path(d)/'queue').glob('*.json')})
            self.assertEqual(stream_queue.inventory()[1],2)

if __name__=='__main__':unittest.main()

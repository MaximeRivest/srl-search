"""No provider calls: disjoint balanced assignments and honest single-teacher export."""
import unittest
from partition import make_plan, A, B
from export_fast import resolve

class PartitionTests(unittest.TestCase):
    def batches(self,n=100):
        return {f'b{i:03d}':{'pairs':[{'id':str(j),'family_id':f'f{j//40}'} for j in range(i,min(i+25,n))]}
                for i in range(0,n,25)}
    def assigned(self,p,m):return {k for ids in p['batches'][m].values() for k in ids}
    def test_preserve_existing_labels_and_split_rest(self):
        answers={A:{str(i):'on_topic' for i in range(30)},B:{str(i):'on_topic' for i in range(20)}}
        p=make_plan(self.batches(),answers)
        self.assertEqual(p['already_answered_pairs'],30)
        self.assertEqual(p['assigned_pairs'],{A:35,B:35})
        a,b=self.assigned(p,A),self.assigned(p,B)
        self.assertFalse(a&b)
        self.assertEqual(a|b,{str(i) for i in range(30,100)})
    def test_odd_remainder_differs_by_one(self):
        p=make_plan(self.batches(99),{A:{},B:{}})
        self.assertEqual(p['assigned_pairs'],{A:49,B:50})
    def test_failures_go_to_other_teacher_not_same_teacher(self):
        p=make_plan(self.batches(10),{A:{'0':None,'1':None,'2':'uncertain'},B:{'1':None}})
        self.assertIn('0',self.assigned(p,B))
        self.assertEqual(p['review_only_ids'],['1'])
        self.assertNotIn('2',self.assigned(p,A)|self.assigned(p,B))
    def test_all_done_assigns_nothing(self):
        p=make_plan(self.batches(3),{A:{'0':'on_topic','1':'off_topic','2':'uncertain'},B:{}})
        self.assertEqual(p['remaining_pairs'],0)
        self.assertFalse(self.assigned(p,A)|self.assigned(p,B))
    def test_assignment_is_deterministic(self):
        a=make_plan(self.batches(101),{A:{},B:{}})
        b=make_plan(self.batches(101),{A:{},B:{}})
        self.assertEqual(a['batches'],b['batches'])
    def test_single_teacher_not_called_consensus(self):
        ans={A:{'label':'on_topic'},B:None}
        self.assertEqual(resolve(ans,single_mode=True),('provisional_single_teacher',1,[A]))
        self.assertEqual(resolve(ans,single_mode=False)[0],'pending')
    def test_one_refusal_does_not_discard_other_answer(self):
        ans={A:{'label':None},B:{'label':'off_topic'}}
        self.assertEqual(resolve(ans,single_mode=True),('provisional_single_teacher',0,[B]))
    def test_disagreement_and_uncertainty_stay_unresolved(self):
        self.assertEqual(resolve({A:{'label':'on_topic'},B:{'label':'off_topic'}},single_mode=True)[0],'disagreement')
        self.assertEqual(resolve({A:{'label':'uncertain'},B:None},single_mode=True)[0],'uncertain')
        self.assertEqual(resolve({A:{'label':None},B:None},single_mode=True)[0],'unanswered')
    def test_safety_and_pending_precede_single_label(self):
        a={A:{'label':'on_topic'},B:None}
        self.assertEqual(resolve(a,overlap=True,single_mode=True)[0],'validation_overlap')
        self.assertEqual(resolve(a,old=0,single_mode=True)[0],'reference_disagrees')
        self.assertEqual(resolve(a,single_mode=True,assigned_pending=True)[0],'pending')

if __name__=='__main__':unittest.main()

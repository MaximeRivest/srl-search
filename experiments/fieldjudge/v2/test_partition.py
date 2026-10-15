"""No provider calls: disjoint balanced assignments and honest single-teacher export."""
import unittest
from partition import make_plan
from export_fast import resolve

class PartitionTests(unittest.TestCase):
    def batches(self,n=100):
        return {f'b{i:03d}':{'pairs':[{'id':str(j),'family_id':f'f{j//40}'} for j in range(i,min(i+25,n))]}
                for i in range(0,n,25)}
    def assigned(self,p,m):return {k for ids in p['batches'][m].values() for k in ids}
    def test_preserve_existing_labels_and_split_rest(self):
        answers={'sol':{str(i):'on_topic' for i in range(30)},'opus':{str(i):'on_topic' for i in range(20)}}
        p=make_plan(self.batches(),answers)
        self.assertEqual(p['already_answered_pairs'],30)
        self.assertEqual(p['assigned_pairs'],{'sol':35,'opus':35})
        a,b=self.assigned(p,'sol'),self.assigned(p,'opus')
        self.assertFalse(a&b)
        self.assertEqual(a|b,{str(i) for i in range(30,100)})
    def test_odd_remainder_differs_by_one(self):
        p=make_plan(self.batches(99),{'sol':{},'opus':{}})
        self.assertEqual(p['assigned_pairs'],{'sol':49,'opus':50})
    def test_failures_go_to_other_teacher_not_same_teacher(self):
        p=make_plan(self.batches(10),{'sol':{'0':None,'1':None,'2':'uncertain'},'opus':{'1':None}})
        self.assertIn('0',self.assigned(p,'opus'))
        self.assertEqual(p['review_only_ids'],['1'])
        self.assertNotIn('2',self.assigned(p,'sol')|self.assigned(p,'opus'))
    def test_all_done_assigns_nothing(self):
        p=make_plan(self.batches(3),{'sol':{'0':'on_topic','1':'off_topic','2':'uncertain'},'opus':{}})
        self.assertEqual(p['remaining_pairs'],0)
        self.assertFalse(self.assigned(p,'sol')|self.assigned(p,'opus'))
    def test_assignment_is_deterministic(self):
        a=make_plan(self.batches(101),{'sol':{},'opus':{}})
        b=make_plan(self.batches(101),{'sol':{},'opus':{}})
        self.assertEqual(a['batches'],b['batches'])
    def test_single_teacher_not_called_consensus(self):
        ans={'sol':{'label':'on_topic'},'opus':None}
        self.assertEqual(resolve(ans,single_mode=True),('provisional_single_teacher',1,['sol']))
        self.assertEqual(resolve(ans,single_mode=False)[0],'pending')
    def test_one_refusal_does_not_discard_other_answer(self):
        ans={'sol':{'label':None},'opus':{'label':'off_topic'}}
        self.assertEqual(resolve(ans,single_mode=True),('provisional_single_teacher',0,['opus']))
    def test_disagreement_and_uncertainty_stay_unresolved(self):
        self.assertEqual(resolve({'sol':{'label':'on_topic'},'opus':{'label':'off_topic'}},single_mode=True)[0],'disagreement')
        self.assertEqual(resolve({'sol':{'label':'uncertain'},'opus':None},single_mode=True)[0],'uncertain')
        self.assertEqual(resolve({'sol':{'label':None},'opus':None},single_mode=True)[0],'unanswered')
    def test_safety_and_pending_precede_single_label(self):
        a={'sol':{'label':'on_topic'},'opus':None}
        self.assertEqual(resolve(a,overlap=True,single_mode=True)[0],'validation_overlap')
        self.assertEqual(resolve(a,old=0,single_mode=True)[0],'kimi_disagrees')
        self.assertEqual(resolve(a,single_mode=True,assigned_pending=True)[0],'pending')

if __name__=='__main__':unittest.main()

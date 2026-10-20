"""Conditional fixed-endpoint confirmation, all arms, within same hard budget."""
import copy,time
def run_confirmation(c,plan):
 import run_study as s
 from rgb_final_confirmation.data import load_manifest,select_rows
 s.guard();pf=s.read(s.ROOT/'config/preflight.json');remaining=s.authorized_deadline()-time.time()
 estimated=1.25*1060*pf['frame_seconds']+1200
 if remaining<estimated:
  s.write(s.ROOT/'reports/confirmation_deferred.json',dict(authorized=True,complete=False,reason='Remaining user-authorized budget below measured full872+188 estimate',remaining_seconds=remaining,estimated_seconds=estimated,fixed_endpoint=6000+plan['steps']))
  summary=s.read(s.ROOT/'reports/summary.json');summary['next_action']='stop_confirmation_incomplete_budget';s.write(s.ROOT/'reports/summary.json',summary);return
 scope=dict(protocol_id=plan['protocol_id'],fixed_global_step=6000+plan['steps'],arms=['S','A','B','C'],validation=872,test=188,no_additional_training=True,no_checkpoint_selection=True,deadline_unix=s.authorized_deadline())
 s.write(s.ROOT/'config/confirmation_scope.json',scope);s.status('confirming_fixed_endpoint',**scope)
 manifest=load_manifest(plan['manifest']);summaries={}
 for split in ['validation','test']:
  rows=select_rows(manifest,split);assert len(rows)==(872 if split=='validation' else 188)
  p=copy.deepcopy(plan)
  if split=='test':p['full_index_map']={r.sample_id:i for i,r in enumerate(rows)};p['evaluation_phase']='test43'
  values=s.evaluate_rows(c,p,plan['steps'],[r.sample_id for r in rows],s.ROOT/'metrics'/f'{split}_full.json')
  summaries[split]={a+'-'+b:s.h.compare(values,a,b) for a,b in s.PAIRS}
 s.write(s.ROOT/'reports/confirmation_summary.json',dict(complete=True,scope=scope,comparisons=summaries,historically_explored_validation=True,published_source_overlap_limit=True))
 summary=s.read(s.ROOT/'reports/summary.json');summary['next_action']='stop_confirmation_complete';summary['test']=188;summary['complete_validation']=872;s.write(s.ROOT/'reports/summary.json',summary)

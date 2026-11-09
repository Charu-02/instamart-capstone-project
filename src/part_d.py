from pathlib import Path
import sys,json,sqlite3,time,uuid
from datetime import datetime,timezone
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'; STATE=ROOT/'state'; sys.path.insert(0,str(ROOT)); import mock_llm
class SimulatedCrash(Exception): pass
class PolicyRAG:
 def __init__(self,overlap=10):
  self.chunks=[]
  for path in sorted((ROOT/'data/policies').glob('*.txt')):
   words=path.read_text().split(); step=40-overlap
   for st in range(0,len(words),step):
    part=words[st:st+40]
    if part:self.chunks.append({'source':path.name,'text':' '.join(part)})
    if st+40>=len(words):break
  self.v=TfidfVectorizer(); self.mat=self.v.fit_transform([x['text'] for x in self.chunks])
 def retrieve(self,q,k=3):
  sims=cosine_similarity(self.v.transform([q]),self.mat)[0]; ids=sims.argsort()[::-1][:k]
  return [{'source':self.chunks[i]['source'],'score':float(sims[i]),'text':self.chunks[i]['text']} for i in ids]
 def answer(self,q):
  hits=self.retrieve(q); ctx='\n'.join(f"[{h['source']}] {h['text']}" for h in hits)
  res=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT. Answer only from supplied context; if absent say I could not find that information in the policy documents.'},{'role':'user','content':f'CONTEXT:\n{ctx}\nQUESTION:\n{q}'}])
  return res.choices[0].message.content,[h['source'] for h in hits]
rag=PolicyRAG()
def get_order(order_id):
 with sqlite3.connect(ROOT/'data/instamart.db') as con: row=con.execute('select order_id,status,order_amount,delivery_minutes from orders where order_id=?',(order_id,)).fetchone()
 return dict(zip(['order_id','status','order_amount','delivery_minutes'],row)) if row else {'error':'order not found'}
def search_policy(query):return {'results':rag.retrieve(query)}
def calculate_refund(order_amount,delivery_minutes):return {'refund_inr':round(float(order_amount),2) if float(delivery_minutes)>45 else 0.0}
def _ledger(tool,order_id,refund_inr):
 with (OUT/'refund_ledger.jsonl').open('a') as f:f.write(json.dumps({'timestamp':datetime.now(timezone.utc).isoformat(),'tool':tool,'order_id':order_id,'refund_inr':refund_inr})+'\n')
 return {'refund_inr':refund_inr}
def issue_refund(order_id,refund_inr):return _ledger('issue_refund',order_id,refund_inr)
def escalate_to_human(order_id,refund_inr):return _ledger('escalate_to_human',order_id,refund_inr)
TOOLS_SPEC=[{'name':'get_order','description':'Fetch order status and delivery details.','parameters':{'order_id':'string'}},{'name':'search_policy','description':'Retrieve policy passages relevant to the query.','parameters':{'query':'string'}},{'name':'calculate_refund','description':'Calculate full or zero refund from delivery time.','parameters':{'order_amount':'number','delivery_minutes':'number'}},{'name':'issue_refund','description':'Record an automatically approved refund.','parameters':{'order_id':'string','refund_inr':'number'}},{'name':'escalate_to_human','description':'Record a refund requiring human approval.','parameters':{'order_id':'string','refund_inr':'number'}}]
TOOL_MAP={'get_order':get_order,'search_policy':search_policy,'calculate_refund':calculate_refund,'issue_refund':issue_refund,'escalate_to_human':escalate_to_human}
def run_agent(user_message,scenario_id,max_iterations=6,token_limit=6000,time_limit_seconds=10,run_id=None,crash_after=None,messages=None,tokens_used=0,steps=None,tools_done=None,llm_calls=0):
 run_id=run_id or f'{scenario_id}_{uuid.uuid4().hex[:8]}'; started=time.monotonic(); steps=steps or []; tools_done=tools_done or []
 msgs=messages or [{'role':'system','content':'AGENT. Verify order and policy before taking refund action.'},{'role':'user','content':user_message}]
 for _ in range(max_iterations):
  if time.monotonic()-started>=time_limit_seconds:return {'status':'FALLBACK_TIMEOUT','final_answer':'FALLBACK_TIMEOUT: hand off to a human agent; time limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_done':tools_done,'run_id':run_id}
  resp=mock_llm.chat(msgs); llm_calls+=1; tokens_used+=resp.usage.total_tokens
  if tokens_used>token_limit:return {'status':'FALLBACK_TOKEN_LIMIT','final_answer':'FALLBACK_TOKEN_LIMIT: hand off to a human agent; token limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_done':tools_done,'run_id':run_id}
  msg=resp.choices[0].message
  if not msg.tool_calls:
   ans=msg.content or ''; return {'status':ans.split(':',1)[0] if ':' in ans else 'FALLBACK','final_answer':ans,'steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_done':tools_done,'run_id':run_id}
  tc=msg.tool_calls[0]; args=tc.arguments if isinstance(tc.arguments,dict) else json.loads(tc.arguments)
  msgs.append({'role':'assistant','content':None,'tool_calls':[{'id':tc.id,'name':tc.name,'arguments':args}]})
  obs=TOOL_MAP[tc.name](**args); tools_done.append(tc.name); steps.append({'tool':tc.name,'arguments':args,'observation':obs})
  msgs.append({'role':'tool','tool_call_id':tc.id,'name':tc.name,'content':json.dumps(obs)})
  STATE.mkdir(exist_ok=True); cp={'run_id':run_id,'scenario_id':scenario_id,'messages':msgs,'tokens_used':tokens_used,'step_number':len(steps),'tools_executed':tools_done,'steps':steps,'llm_calls':llm_calls,'user_message':user_message}
  (STATE/f'{run_id}.json').write_text(json.dumps(cp,indent=2))
  if crash_after and len(steps)==crash_after:raise SimulatedCrash(run_id)
 return {'status':'FALLBACK_MAX_ITERATIONS','final_answer':'FALLBACK_MAX_ITERATIONS: hand off to a human agent; iteration limit reached.','steps':steps,'llm_calls':llm_calls,'tokens_used':tokens_used,'tools_done':tools_done,'run_id':run_id}
def resume_agent(run_id):
 cp=json.loads((STATE/f'{run_id}.json').read_text())
 return run_agent(cp['user_message'],cp['scenario_id'],run_id=run_id,messages=cp['messages'],tokens_used=cp['tokens_used'],steps=cp['steps'],tools_done=cp['tools_executed'],llm_calls=cp['llm_calls'])
def run():
 OUT.mkdir(exist_ok=True); STATE.mkdir(exist_ok=True); ev=pd.read_csv(ROOT/'data/raw/policy_eval.csv'); evrows=[]
 for _,r in ev.iterrows():
  hits=rag.retrieve(r.question); ans,_=rag.answer(r.question); evrows.append({'question':r.question,'expected_doc_found':r.expected_doc=='NONE' or r.expected_doc in [h['source'] for h in hits],'fact_found':r.expected_doc=='NONE' or r.expected_fact in ans,'fallback':ans.startswith('I could not find')})
 pd.DataFrame(evrows).to_csv(OUT/'rag_evaluation.csv',index=False)
 rag0=PolicyRAG(0); zrows=[]
 for _,r in ev.iterrows():
  hits=rag0.retrieve(r.question); ctx='\n'.join(f"[{h['source']}] {h['text']}" for h in hits)
  ans=mock_llm.chat([{'role':'system','content':'POLICY ASSISTANT. Answer only from context; otherwise say I could not find.'},{'role':'user','content':f'CONTEXT:\n{ctx}\nQUESTION:\n{r.question}'}]).choices[0].message.content
  zrows.append({'question':r.question,'expected_doc_found':r.expected_doc=='NONE' or r.expected_doc in [h['source'] for h in hits],'fact_found':r.expected_doc=='NONE' or r.expected_fact in ans,'fallback':ans.startswith('I could not find')})
 pd.DataFrame(zrows).to_csv(OUT/'rag_zero_overlap.csv',index=False)
 scenarios=pd.read_csv(ROOT/'data/raw/agent_scenarios.csv'); runs=[]; (OUT/'refund_ledger.jsonl').write_text('')
 for _,r in scenarios.iterrows():runs.append(run_agent(r.message,r.scenario_id))
 with (OUT/'agent_runs.jsonl').open('w') as f:
  for (_,s),x in zip(scenarios.iterrows(),runs):f.write(json.dumps({'scenario_id':s.scenario_id,'status':x['status'],'final_answer':x['final_answer'],'llm_calls':x['llm_calls'],'tokens_used':x['tokens_used'],'steps':x['steps']})+'\n')
 s1=scenarios.iloc[0]; token_demo=run_agent(s1.message,'S1_TOKEN',token_limit=200); timeout_demo=run_agent(s1.message,'S1_TIMEOUT',time_limit_seconds=0)
 runid='S1_CRASH_RESUME'
 try:run_agent(s1.message,'S1',run_id=runid,crash_after=2)
 except SimulatedCrash:resumed=resume_agent(runid)
 meta={'chunks':len(rag.chunks),'chunk_counts':{p.name:sum(x['source']==p.name for x in rag.chunks) for p in (ROOT/'data/policies').glob('*.txt')},'token_demo_status':token_demo['status'],'timeout_demo_status':timeout_demo['status'],'crash_resume_status':resumed['status'],'crash_resume_tools':resumed['tools_done'],'scenario_results':[{'scenario_id':s.scenario_id,'status':x['status'],'final_answer':x['final_answer'],'llm_calls':x['llm_calls'],'tokens_used':x['tokens_used']} for (_,s),x in zip(scenarios.iterrows(),runs)]}
 (OUT/'part_d_summary.json').write_text(json.dumps(meta,indent=2)); print('Part D complete',meta)
if __name__=='__main__':run()

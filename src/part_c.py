from pathlib import Path
import sys,re,json,random,time
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT)); import mock_llm
OUT=ROOT/'outputs'; CATS={'billing','delivery','technical','fraud'}; BASE_DELAY=.05; MAX_INPUT_TOKENS=400
EMAIL=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+'); PHONE=re.compile(r'\b[6-9]\d{9}\b'); ORDER=re.compile(r'ORD-\d{6}')
ZERO='You are a CLASSIFY support ticket classifier. Allowed categories: billing, delivery, technical, fraud. Return JSON only in the form {"category": "...", "reason": "..."}.'
FEW=ZERO+'\nEXAMPLES:\nTicket: My card has an unauthorized transaction.\nLabel: {"category":"fraud","reason":"unauthorized transaction"}\nTicket: The app crashes on login.\nLabel: {"category":"technical","reason":"app crash"}\n'
def redact(s):
 counts=[]
 for pat,rep in [(EMAIL,'[EMAIL]'),(PHONE,'[PHONE]'),(ORDER,'[ORDER_ID]')]: s,n=pat.subn(rep,s); counts.append(n)
 assert not EMAIL.search(s) and not PHONE.search(s) and not ORDER.search(s)
 return s,counts
def call_retry(msgs,tid):
 for n in range(4):
  try:return mock_llm.chat(msgs)
  except mock_llm.TransientLLMError:
   if n==3: raise
   wait=BASE_DELAY*2**n*random.uniform(.9,1.15); print(f'retry ticket={tid} retry={n} wait={wait:.5f}s'); time.sleep(wait)
def classify(df,prompt,budget=None):
 mock_llm.reset(); random.seed(42); rows=[]; total=0
 for _,r in df.iterrows():
  text=str(r.text); status=''; cat=''; used=0
  if not text.strip(): status='skipped_empty'
  elif mock_llm.est_tokens(text)>MAX_INPUT_TOKENS: status='flagged_long'
  else:
   clean,counts=redact(text)
   if budget is not None and total>=budget: status='budget_exhausted'
   else:
    assert not EMAIL.search(clean) and not PHONE.search(clean) and not ORDER.search(clean)
    resp=call_retry([{'role':'system','content':prompt},{'role':'user','content':clean}],r.ticket_id)
    used=resp.usage.total_tokens; total+=used; answer=resp.choices[0].message.content or ''
    try:
     obj=json.loads(answer); candidate=obj.get('category')
     if candidate in CATS: cat=candidate; status='classified_json'
     else: status='flagged_invalid_output'
    except Exception:
     words=re.findall(r'[a-z]+',answer.lower()); found=CATS.intersection(words)
     if len(found)==1: cat=next(iter(found)); status='classified_keyword'
     else: status='flagged_invalid_output'
  rows.append({'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':status,'category':cat,'tokens_used':used,'true_category':r.true_category})
 return pd.DataFrame(rows),total
def run():
 df=pd.read_csv(ROOT/'data/raw/tickets.csv',keep_default_na=False)
 # Count exact redactions across all raw tickets and assert no sensitive patterns remain.
 totals=[0,0,0]; redacted=[]
 for raw in df.text.astype(str):
  clean,counts=redact(raw); totals=[a+b for a,b in zip(totals,counts)]; redacted.append(clean)
 assert all(not EMAIL.search(x) and not PHONE.search(x) and not ORDER.search(x) for x in redacted)
 (OUT/'redaction_counts.json').write_text(json.dumps({'emails':totals[0],'phones':totals[1],'order_ids':totals[2],'total':sum(totals),'remaining_matches':0},indent=2))
 z,zt=classify(df,ZERO); f,ft=classify(df,FEW)
 def stats(x,t):
  good=x[x.status.isin(['classified_json','classified_keyword'])]; return {'classified':len(good),'accuracy':float((good.category==good.true_category).mean()),'tokens':t,'tokens_per_classified':t/max(1,len(good))}
 (OUT/'triage_comparison.json').write_text(json.dumps({'zero_shot':stats(z,zt),'few_shot':stats(f,ft)},indent=2))
 pd.crosstab(z.loc[z.category!='','true_category'],z.loc[z.category!='','category']).to_csv(OUT/'triage_confusion_zero.csv')
 budget=6000; bd,_=classify(df,ZERO,budget)
 while not (bd.status=='budget_exhausted').any() and budget>1000: budget-=500; bd,_=classify(df,ZERO,budget)
 bd.to_csv(OUT/'triage_budget_demo.csv',index=False)
 f,_=classify(df,FEW,100000); seg=pd.read_csv(OUT/'customer_segments.csv').set_index('customer_id'); result=[]
 for _,r in f.iterrows():
  s=seg.loc[r.customer_id] if r.customer_id in seg.index else None; sn=s.segment_name if s is not None else ''; premium=bool(s.is_premium) if s is not None else False
  if r.status in ['skipped_empty','flagged_long','flagged_invalid_output','budget_exhausted']: route,priority='human_review','NORMAL'
  elif r.category=='fraud': route,priority='security_team','HIGH'
  elif premium: route,priority='premium_desk','HIGH'
  else: route,priority='support_queue','NORMAL'
  result.append({'ticket_id':r.ticket_id,'customer_id':r.customer_id,'status':r.status,'category':r.category,'segment_name':sn,'priority':priority,'route':route,'tokens_used':r.tokens_used})
 pd.DataFrame(result).to_csv(OUT/'triage_results.csv',index=False)
 print('Part C complete',json.loads((OUT/'triage_comparison.json').read_text()),'budget',budget,'exhausted',int((bd.status=='budget_exhausted').sum()))
if __name__=='__main__':run()

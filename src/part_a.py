from pathlib import Path
import sqlite3, pandas as pd, numpy as np
ROOT=Path(__file__).resolve().parents[1]; RAW=ROOT/'data/raw'; CLEAN=ROOT/'data/clean'; CLEAN.mkdir(exist_ok=True)
def run():
    c=pd.read_csv(RAW/'customers.csv',keep_default_na=False); p=pd.read_csv(RAW/'products.csv'); o=pd.read_csv(RAW/'orders.csv',keep_default_na=False)
    # Audit is retained as a machine-readable report for reproducibility.
    audit=[]
    for name,df in [('customers',c),('products',p),('orders',o)]:
        for col in df:
            miss=(df[col].astype(str).str.strip()=='').sum()
            audit.append({'table':name,'column':col,'dtype':str(df[col].dtype),'missing_count':int(miss),'missing_pct':round(100*miss/len(df),2)})
    pd.DataFrame(audit).to_csv(ROOT/'outputs/audit_missingness.csv',index=False)
    c['city']=c.city.astype(str).str.strip().str.title()
    c['gender']=c.gender.astype(str).str.strip().str.lower().map({'m':'male','male':'male','f':'female','female':'female'}).fillna('unknown')
    c['age']=pd.to_numeric(c.age,errors='coerce'); c.loc[~c.age.between(10,100),'age']=np.nan
    c['age']=c.age.fillna(c.groupby('city').age.transform('median')).round().astype(int)
    c['signup_date']=pd.to_datetime(c.signup_date)
    p['price_inr']=pd.to_numeric(p.price_inr)
    o['order_ts']=pd.to_datetime(o.order_ts)
    o=o.drop_duplicates().copy(); o=o.sort_values(['order_ts']).drop_duplicates('order_id',keep='last').copy()
    for col in ['quantity','discount_pct','order_amount','delivery_minutes']: o[col]=pd.to_numeric(o[col],errors='coerce')
    o=o.merge(p[['product_id','price_inr','category']],on='product_id',how='left')
    miss=o.order_amount.isna(); o.loc[miss,'order_amount']=(o.loc[miss,'price_inr']*o.loc[miss,'quantity']*(1-o.loc[miss,'discount_pct']/100)).round(2)
    o.loc[o.delivery_minutes<0,'delivery_minutes']=np.nan
    completed=o.status.eq('completed'); med=o.loc[completed].groupby('category').delivery_minutes.transform('median')
    o.loc[completed,'delivery_minutes']=med.groupby(o.loc[completed,'category']).transform(lambda x:x.fillna(x.median())) if False else o.loc[completed,'delivery_minutes']
    # Category median imputation computed across completed rows, aligned by category.
    catmed=o.loc[completed].groupby('category').delivery_minutes.median()
    o.loc[completed,'delivery_minutes']=o.loc[completed,'delivery_minutes'].fillna(o.loc[completed,'category'].map(catmed))
    o=o.drop(columns=['price_inr','category'])
    for name,df in [('customers',c),('products',p),('orders',o)]:
        if name=='orders': df['order_ts']=df.order_ts.dt.strftime('%Y-%m-%d %H:%M:%S')
        if name=='customers': df['signup_date']=df.signup_date.dt.strftime('%Y-%m-%d')
        df.to_csv(CLEAN/f'{name}.csv',index=False)
    db=ROOT/'data/instamart.db'
    with sqlite3.connect(db) as con:
        for name in ['customers','products','orders']: pd.read_csv(CLEAN/f'{name}.csv').to_sql(name,con,if_exists='replace',index=False)
    assert len(c)==1200 and c.city.nunique()==6 and c.age.isna().sum()==0
    assert len(o)==12309 and not o.status.eq('pending').any() and o.order_amount.isna().sum()==0
    assert o.loc[o.status.eq('completed'),'delivery_minutes'].isna().sum()==0
    assert o.loc[o.status.eq('cancelled'),'delivery_minutes'].isna().sum()==1199
    assert round(o.loc[o.status.eq('completed'),'order_amount'].sum(),2)==2998668.80
    print('ALL CHECKPOINTS PASSED')
    # Run all saved SQL queries and persist results for inspection.
    sql=(ROOT/'sql/queries.sql').read_text(); statements=[s.strip() for s in sql.split(';') if s.strip()]
    with sqlite3.connect(db) as con:
        for i,q in enumerate(statements,1): pd.read_sql_query(q,con).to_csv(ROOT/f'outputs/sql_q{i}.csv',index=False)
if __name__=='__main__': run()

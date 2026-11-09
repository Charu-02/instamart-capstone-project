from pathlib import Path
import pandas as pd,numpy as np,joblib
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score,confusion_matrix,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
def run():
 c=pd.read_csv(ROOT/'data/clean/customers.csv'); o=pd.read_csv(ROOT/'data/clean/orders.csv',parse_dates=['order_ts']); o['order_ts']=pd.to_datetime(o.order_ts)
 done=o[o.status=='completed']; snap=pd.Timestamp('2025-10-01')
 rfm=done.groupby('customer_id').agg(last=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index(); rfm['recency']=(snap-rfm.pop('last')).dt.days
 scaler=StandardScaler(); X=scaler.fit_transform(rfm[['recency','frequency','monetary']]); metrics=[]
 for k in range(2,9):
  km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X); metrics.append({'k':k,'inertia':km.inertia_,'silhouette':silhouette_score(X,km.labels_)})
 pd.DataFrame(metrics).to_csv(OUT/'kmeans_metrics.csv',index=False)
 k=4; km=KMeans(n_clusters=k,n_init=10,random_state=42).fit(X); rfm['segment_id']=km.labels_
 prof=rfm.groupby('segment_id').agg(size=('customer_id','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean')).reset_index()
 names={}
 premium=int(prof.loc[prof.monetary.idxmax(),'segment_id'])
 for _,z in prof.iterrows():
  sid=int(z.segment_id); names[sid]=('Premium Champions' if sid==premium else 'Frequent Essentials' if z.frequency>14 else 'At-Risk High Value' if z.monetary>4000 else 'Occasional Shoppers')
 rfm['segment_name']=rfm.segment_id.map(names); rfm['is_premium']=rfm.segment_id.eq(premium); rfm.to_csv(OUT/'customer_segments.csv',index=False)
 pd.DataFrame({'component':['PC1','PC2'],'explained_variance_ratio':PCA(n_components=2).fit(StandardScaler().fit_transform(rfm[['recency','frequency','monetary']])).explained_variance_ratio_}).to_csv(OUT/'pca_variance.csv',index=False)
 # Cohort and pre-cutoff features only.
 cutoff=pd.Timestamp('2025-09-01'); pre=o[(o.order_ts<cutoff)&o.status.isin(['completed','cancelled'])].copy(); pre_done=pre[pre.status=='completed']
 cohort=pre_done.groupby('customer_id').agg(last=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount=('discount_pct','mean'),avg_delivery=('delivery_minutes','mean')).reset_index()
 cohort['recency']=(cutoff-cohort.pop('last')).dt.days
 cancel=pre.groupby('customer_id').status.apply(lambda s:(s=='cancelled').sum()/len(s)).rename('cancellation_rate')
 fav=pre_done.merge(pd.read_csv(ROOT/'data/clean/products.csv')[['product_id','category']],on='product_id').groupby('customer_id').category.agg(lambda x:x.mode().iloc[0]).rename('favourite_category')
 cohort=cohort.merge(cancel,on='customer_id').merge(fav,on='customer_id').merge(c,on='customer_id'); cohort['tenure_days']=(cutoff-pd.to_datetime(cohort.signup_date)).dt.days
 after=done[(done.order_ts>=cutoff)&(done.order_ts<pd.Timestamp('2025-10-01'))].groupby('customer_id').size(); cohort['lapsed']=(~cohort.customer_id.isin(after.index)).astype(int)
 num=['recency','frequency','monetary','avg_discount','cancellation_rate','avg_delivery','age','tenure_days']; cat=['city','gender','favourite_category']; cohort[num]=cohort[num].replace([np.inf,-np.inf],np.nan); cohort[num]=cohort[num].fillna(cohort[num].median()); cohort[cat]=cohort[cat].fillna('unknown'); assert not cohort[num+cat].isna().any().any()
 X=cohort[num+cat]; y=cohort.lapsed; Xtr,Xte,ytr,yte,idxtr,idxte=train_test_split(X,y,cohort.index,test_size=.2,stratify=y,random_state=42)
 prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),cat)])
 lp=Pipeline([('prep',prep),('model',LogisticRegression(max_iter=1000))]); lg=GridSearchCV(lp,{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1').fit(Xtr,ytr)
 rf=Pipeline([('prep',prep),('model',RandomForestClassifier(n_estimators=100,random_state=42))]); rg=GridSearchCV(rf,{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1').fit(Xtr,ytr)
 chosen=rg.best_estimator_ if f1_score(yte,rg.predict(Xte),zero_division=0)>=f1_score(yte,lg.predict(Xte),zero_division=0) else lg.best_estimator_
 model=chosen; pred=model.predict(Xte); joblib.dump(model,OUT/'lapse_model.joblib'); assert np.array_equal(pred,joblib.load(OUT/'lapse_model.joblib').predict(Xte))
 # Baseline and model-specific test metrics (positive class = lapsed).
 baseline=np.zeros(len(yte),dtype=int)
 metrics={}
 for name,est in [('logistic',lg.best_estimator_),('random_forest',rg.best_estimator_)]:
  pp=est.predict(Xte)
  metrics[name]={'confusion_matrix':confusion_matrix(yte,pp).tolist(),'precision':float(precision_score(yte,pp,zero_division=0)),'recall':float(recall_score(yte,pp,zero_division=0)),'f1':float(f1_score(yte,pp,zero_division=0))}
 metrics['baseline']={'accuracy':float((baseline==yte).mean()),'lapsed_f1':float(f1_score(yte,baseline,zero_division=0))}
 # Training-only random oversampling for the best logistic model.
 train=pd.concat([Xtr.reset_index(drop=True),ytr.reset_index(drop=True).rename('lapsed')],axis=1)
 counts=train.lapsed.value_counts(); majority=int(counts.idxmax()); minority=int(counts.idxmin()); needed=int(counts.max()-counts.min())
 extras=train[train.lapsed==minority].sample(n=needed,replace=True,random_state=42)
 balanced=pd.concat([train,extras],ignore_index=True); balanced_log=clone(lg.best_estimator_); balanced_log.fit(balanced[num+cat],balanced.lapsed)
 bp=balanced_log.predict(Xte)
 metrics['balanced_logistic']={'precision':float(precision_score(yte,bp,zero_division=0)),'recall':float(recall_score(yte,bp,zero_division=0)),'f1':float(f1_score(yte,bp,zero_division=0)),'train_class_counts':balanced.lapsed.value_counts().to_dict()}
 # Extract transformed feature names and model interpretation.
 rfprep=rg.best_estimator_.named_steps['prep']; feat=rfprep.get_feature_names_out(); importances=rg.best_estimator_.named_steps['model'].feature_importances_
 top=sorted(zip(feat,importances),key=lambda x:x[1],reverse=True)[:5]
 lprep=lg.best_estimator_.named_steps['prep']; lnames=lprep.get_feature_names_out(); coefs=lg.best_estimator_.named_steps['model'].coef_[0]
 coef_top=sorted(zip(lnames,coefs),key=lambda x:abs(x[1]),reverse=True)[:3]
 # training-fitted numeric scaler frequency mean versus full cohort mean
 freq_idx=num.index('frequency'); train_freq_mean=float(lprep.named_transformers_['num'].named_steps['scale'].mean_[freq_idx])
 metrics['interpretation']={'rf_top5':[(str(a),float(b)) for a,b in top],'logistic_top3_coefficients':[(str(a),float(b)) for a,b in coef_top],'training_scaler_frequency_mean':train_freq_mean,'full_cohort_frequency_mean':float(cohort.frequency.mean())}
 (OUT/'part_b_metrics.json').write_text(__import__('json').dumps(metrics,indent=2))
 # Premium list is explicitly post-hoc and illustrative as specified.
 testids=cohort.loc[idxte,'customer_id']; premium_ids=set(rfm.loc[rfm.is_premium,'customer_id']); probs=model.predict_proba(Xte)[:,1]; ret=pd.DataFrame({'customer_id':testids.values,'lapse_probability':probs,'actually_lapsed':yte.values}); ret=ret[ret.customer_id.isin(premium_ids)].sort_values('lapse_probability',ascending=False).head(10); ret.to_csv(OUT/'retention_list.csv',index=False)
 # persist metrics, feature importances, and profiles for README evidence
 summary={'cohort_n':len(cohort),'lapsed_n':int(y.sum()),'train_n':len(Xtr),'test_n':len(Xte),'logistic_best':str(lg.best_params_),'logistic_cv_f1':float(lg.best_score_),'forest_best':str(rg.best_params_),'forest_cv_f1':float(rg.best_score_),'selected_test_f1':float(f1_score(yte,pred,zero_division=0)),'selected_model':'random_forest' if model is rg.best_estimator_ else 'logistic_regression','retention_true_lapsed':int(ret.actually_lapsed.sum())}
 import json;(OUT/'part_b_summary.json').write_text(json.dumps(summary,indent=2))
 prof.to_csv(OUT/'segment_profiles.csv',index=False)
 print('Part B complete',summary)
if __name__=='__main__':run()

import numpy as np
from scipy.sparse import csr_matrix

class ItemBasedCF:
    def __init__(self,n_neighbors=50,min_common=2,shrinkage=10.):
        self.n_neighbors,self.min_common,self.shrinkage=n_neighbors,min_common,shrinkage

    def fit(self,df):
        users=sorted(df.user_id.unique());items=sorted(df.business_id.unique())
        self.users=np.array(users);self.items=np.array(items)
        self.user_to_idx={x:i for i,x in enumerate(users)}
        self.item_to_idx={x:i for i,x in enumerate(items)}
        row=df.user_id.map(self.user_to_idx).to_numpy()
        col=df.business_id.map(self.item_to_idx).to_numpy()
        value=df.stars.to_numpy(dtype=np.float32)
        R=csr_matrix((value,(row,col)),shape=(len(users),len(items)))
        self.R=R
        count=np.diff(R.indptr)
        self.mean=float(value.mean())
        self.user_mean=np.asarray(R.sum(1)).ravel()/np.maximum(count,1)
        num=np.asarray(R.sum(0)).ravel()
        cnt=np.asarray((R!=0).sum(0)).ravel()
        self.item_mean=(num+5*self.mean)/(cnt+5)
        C=R.copy()
        C.data-=np.repeat(self.user_mean,count).astype(np.float32)
        norms=np.sqrt(np.asarray(C.power(2).sum(0)).ravel())
        sim=(C.T@C).tocoo()
        denom=norms[sim.row]*norms[sim.col]
        score=np.divide(sim.data,denom,out=np.zeros_like(sim.data),where=denom>0)
        B=R.copy();B.data[:]=1
        common=(B.T@B).tocsr()
        overlap=np.asarray(common[sim.row,sim.col]).ravel()
        keep=(sim.row!=sim.col)&(overlap>=self.min_common)&(score>0)
        score=score[keep]*(overlap[keep]/(overlap[keep]+self.shrinkage))
        S=csr_matrix((score,(sim.row[keep],sim.col[keep])),shape=(len(items),len(items)))
        rows=[];cols=[];vals=[]
        for target in range(len(items)):
            a,b=S.indptr[target:target+2]
            js=S.indices[a:b];vv=S.data[a:b]
            if len(js)>self.n_neighbors:
                top=np.argpartition(vv,-self.n_neighbors)[-self.n_neighbors:]
                js=js[top];vv=vv[top]
            rows.extend([target]*len(js));cols.extend(js.tolist());vals.extend(vv.tolist())
        self.similarity=csr_matrix((np.asarray(vals,dtype=np.float32),(rows,cols)),shape=S.shape)
        self.abs_similarity=self.similarity.copy();self.abs_similarity.data=np.abs(self.abs_similarity.data)
        return self

    def recommend(self,user_id,candidates,k=10):
        if user_id not in self.user_to_idx:return []
        user=self.R.getrow(self.user_to_idx[user_id])
        centered=np.zeros(self.R.shape[1],dtype=np.float32)
        centered[user.indices]=user.data-self.user_mean[self.user_to_idx[user_id]]
        mask=np.zeros(self.R.shape[1],dtype=np.float32);mask[user.indices]=1
        numerator=self.similarity@centered
        denominator=self.abs_similarity@mask
        scores=self.item_mean.copy()
        present=denominator>0
        scores[present]=self.user_mean[self.user_to_idx[user_id]]+numerator[present]/denominator[present]
        scores=np.clip(scores,1,5)
        valid=sorted([v for v in candidates if v in self.item_to_idx])
        indices=np.array([self.item_to_idx[x] for x in valid],dtype=int)
        if not len(indices):return []
        order=np.lexsort((indices,-scores[indices]))[:k]
        return [valid[j] for j in order]


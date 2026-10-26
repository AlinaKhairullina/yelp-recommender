import ast
import numpy as np
import pandas as pd
import torch
from torch import nn

class FMNet(nn.Module):
    def __init__(self,n_features,factors):
        super().__init__()
        self.linear=nn.Embedding(n_features+1,1,padding_idx=0)
        self.emb=nn.Embedding(n_features+1,factors,padding_idx=0)
        self.bias=nn.Parameter(torch.zeros(1))
        nn.init.normal_(self.emb.weight,std=.02)
        nn.init.zeros_(self.linear.weight)
    def forward(self,x):
        w=self.linear(x).sum((1,2))
        v=self.emb(x)
        pair=.5*((v.sum(1)**2)-(v**2).sum(1)).sum(1)
        return self.bias+w+pair

class FactorizationMachine:
    def __init__(self,meta_business,meta_users,factors=16,epochs=5,lr=.005,reg=1e-5,batch_size=1024,seed=42):
        self.meta_business=meta_business
        self.meta_users=meta_users
        self.factors,self.epochs,self.lr,self.reg,self.batch_size,self.seed=factors,epochs,lr,reg,batch_size,seed

    @staticmethod
    def clean(v):
        if v is None or str(v).strip() in ('','None','nan'):return 'unknown'
        s=str(v).strip().lower()
        if len(s)>2 and s[0]=='u' and s[1] in "\"'":s=s[1:]
        return s.strip("'\"")

    def fit(self,df):
        torch.manual_seed(self.seed)
        self.business={}
        for r in self.meta_business.to_dict('records'):
            attrs=r.get('attributes') if isinstance(r.get('attributes'),dict) else {}
            tokens=[]
            cats=[c.strip().lower() for c in str(r.get('categories') or '').split(',') if c.strip()]
            tokens.extend('cat='+c for c in cats[:8])
            for a in ('RestaurantsPriceRange2','WiFi','OutdoorSeating','RestaurantsDelivery','RestaurantsTakeOut','Alcohol'):
                tokens.append(a+'='+self.clean(attrs.get(a)))
            self.business[r['business_id']]=tokens
        um=self.meta_users[['user_id','yelping_since']].copy()
        um['yelping_since']=pd.to_datetime(um['yelping_since'],errors='coerce')
        self.since=dict(zip(um.user_id,um.yelping_since))
        self.cutoff=pd.to_datetime(df.date).max()
        self.vocab={}
        for user in sorted(df.user_id.unique()):self.vocab['user='+user]=len(self.vocab)+1
        for item in sorted(df.business_id.unique()):
            self.vocab['item='+item]=len(self.vocab)+1
            for token in self.business.get(item,[]):
                if token not in self.vocab:self.vocab[token]=len(self.vocab)+1
        for t in range(0,21):self.vocab['tenure='+str(t)]=len(self.vocab)+1
        self.vocab['tenure=unknown']=len(self.vocab)+1
        self.max_tokens=2+8+6+1
        self.net=FMNet(len(self.vocab),self.factors)
        opt=torch.optim.Adam(self.net.parameters(),lr=self.lr,weight_decay=self.reg)
        x=self._features(df.user_id.to_list(),df.business_id.to_list(),pd.to_datetime(df.date).to_list())
        y=torch.tensor((df.stars.to_numpy()>=4).astype(np.float32))
        for _ in range(self.epochs):
            for ids in torch.randperm(len(y)).split(self.batch_size):
                opt.zero_grad()
                loss=nn.functional.binary_cross_entropy_with_logits(self.net(x[ids]),y[ids])
                loss.backward();opt.step()
        self.net.eval()
        return self

    def _features(self,users,items,dates):
        result=np.zeros((len(users),self.max_tokens),dtype=np.int64)
        for ix,(u,i,date) in enumerate(zip(users,items,dates)):
            tokens=['user='+u,'item='+i]+self.business.get(i,[])
            start=self.since.get(u)
            if start is None or pd.isna(start):age='unknown'
            else:age=str(min(20,max(0,int((pd.Timestamp(date)-start).days/365.25))))
            tokens.append('tenure='+age)
            result[ix,:min(len(tokens),self.max_tokens)]=[self.vocab.get(t,0) for t in tokens[:self.max_tokens]]
        return torch.from_numpy(result)

    @torch.no_grad()
    def recommend(self,user_id,candidates,k=10):
        if 'user='+user_id not in self.vocab:return []
        valid=sorted([x for x in candidates if 'item='+x in self.vocab])
        if not valid:return []
        scores=[]
        for st in range(0,len(valid),2048):
            batch=valid[st:st+2048]
            x=self._features([user_id]*len(batch),batch,[self.cutoff]*len(batch))
            scores.extend(self.net(x).numpy().tolist())
        indices=np.argsort(-np.asarray(scores),kind='stable')[:k]
        return [valid[j] for j in indices]

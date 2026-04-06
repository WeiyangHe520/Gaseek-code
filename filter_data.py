import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

#src_dir=r"data"
#mask=np.ones(78,dtype=np.int32)
#for f in os.listdir(src_dir):
#    if not f.endswith(".csv"):
#        continue
#    with open(os.path.join(src_dir,f),"r") as fid:
#        df=pd.read_csv(os.path.join(src_dir,f))
#        cnt=df.isnull().sum().values
#        null_ratio=cnt/df.shape[0]
#        mask[null_ratio>0.2]=0
#        
#for f in os.listdir(src_dir):
#    if not f.endswith(".csv"):
#        continue
#    with open(os.path.join(src_dir,f),"r") as fid:
#        df=pd.read_csv(os.path.join(src_dir,f))
#        col_names=df.columns
#        values=df.values
#        col_names=col_names[mask==1]
#        values=values[:,mask==1]
#        df1=pd.DataFrame(data=values,columns=col_names)
#        df1.to_csv(f.replace(".csv","_valid"+".csv"),index=False,encoding="gbk")

src_dir=r"D:\何维阳\20250927\20250927\data\csv"
mask=np.ones(18,dtype=np.int32)
for f in os.listdir(src_dir):
    if not f.endswith(".csv"):
        continue
    with open(os.path.join(src_dir,f),"r") as fid:
        df=pd.read_csv(os.path.join(src_dir,f))
        cnt=df.isnull().sum().values
        null_ratio=cnt/df.shape[0]
        mask[null_ratio>0.9]=0

imp = IterativeImputer(max_iter=20, random_state=0)
for f in os.listdir(src_dir):
    if not f.endswith(".csv"):
        continue
    with open(os.path.join(src_dir,f),"r") as fid:
        df=pd.read_csv(os.path.join(src_dir,f))
        col_names=df.columns[mask==1]
        data=df.values[:,mask==1]
        data=imp.fit_transform(data)
        df1=pd.DataFrame(data=data,columns=col_names)
        df1.to_csv(f.replace(".csv","_mice"+".csv"),index=False,encoding="gbk")
        
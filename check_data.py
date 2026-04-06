import os
import numpy as np
import pandas as pd

src_dir=r"data\csv"
for f in os.listdir(src_dir):
    if not f.endswith(".csv"):
        continue
    with open(os.path.join(src_dir,f),"r") as fid:
        df=pd.read_csv(os.path.join(src_dir,f))
        col_names=df.columns
        data=df.to_numpy()
        for k in range(data.shape[1]):
            try:
                col=data[:,k].astype(np.float32)
            except:
                print("file:{},col:{}".format(f,col_names[k]))
            
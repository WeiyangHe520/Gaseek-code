import os
import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.model_selection import train_test_split

def HosmerLemeshowTest(y,yp,num_box=10):
    num=y.shape[0]
    box_idx=np.round(num*np.arange(num_box+1)/num_box).astype(np.int32)
    
    obs0=np.zeros(num_box,dtype=np.float32)
    exp0=np.zeros(num_box,dtype=np.float32)
    obs1=np.zeros(num_box,dtype=np.float32)
    exp1=np.zeros(num_box,dtype=np.float32)
    
    idx_sort=np.argsort(yp[:,1])
    for k in range(num_box):
        idx=idx_sort[box_idx[k]:box_idx[k+1]]
        obs0[k]=np.sum(1-y[idx])
        exp0[k]=np.sum(1-yp[idx,1])
        obs1[k]=np.sum(y[idx])
        exp1[k]=np.sum(yp[idx,1])    
        
    hltest=np.sum((obs0-exp0)**2/exp0+(obs1-exp1)**2/exp1)
    pval = 1 - chi2.cdf(hltest, num_box-2)
    box_num=box_idx[1:]-box_idx[:-1]
    cali_curve=np.vstack((exp1/box_num,obs1/box_num))
#    brier=np.mean((cali_curve[0,:]-cali_curve[1,:])**2)
    return cali_curve,pval

def dca_curve(y_label,pred_p):
    num_thr=50
    dca=np.zeros((4,num_thr),dtype=np.float32)
    n=y_label.shape[0]
    for k,thr in enumerate(np.linspace(0,0.99999,50)):
        pred_y=(pred_p>thr).astype(np.int32)
        tn=np.count_nonzero(np.logical_and(pred_y==0,y_label==0))
        tp=np.count_nonzero(np.logical_and(pred_y==1,y_label==1))
        fp=np.count_nonzero(np.logical_and(pred_y==1,y_label==0))
        fn=np.count_nonzero(np.logical_and(pred_y==0,y_label==1))
        benifit_model=tp/n-fp/n*(thr/(1-thr))
        benifit_all=(tp+fn)/n-(tn+fp)/n*(thr/(1-thr))
        dca[1,k]=benifit_all
        dca[2,k]=benifit_model
        dca[3,k]=thr
        
    return dca

    
def eval_clf(clf,x,y,thr=None):
    pred_p=clf.predict_proba(x)
    auc=roc_auc_score(y,pred_p[:,1])    
    roc=roc_curve(y,pred_p[:,1])
    dca=dca_curve(y,pred_p[:,1])
    
    if thr is None:
        idx=np.argmax(roc[1]-roc[0]) # 临界值索引
        thr=roc[2][idx]
    pred_y=(pred_p[:,1]>thr).astype(np.int32)
    
    tn=np.count_nonzero(np.logical_and(pred_y==0,y==0))
    tp=np.count_nonzero(np.logical_and(pred_y==1,y==1))
    fp=np.count_nonzero(np.logical_and(pred_y==1,y==0))
    fn=np.count_nonzero(np.logical_and(pred_y==0,y==1))
    
    acc=(tn+tp)/(x.shape[0])
    sens=tp/(tp+fn) # recall
    spec=tn/(tn+fp) 
    ppv=tp/(tp+fp) #precision
    npv=tn/(tn+fn)
    ydi=sens+spec-1
    f1=2*sens*ppv/(sens+ppv)
    cali_curve,hl=HosmerLemeshowTest(y,pred_p)
    brier=np.mean((y-pred_p[:,1])**2)
    
    res={"roc":roc,
         "auc":auc,
         "dca":dca,
         "acc":acc,
         "sens":sens,
         "spec":spec,
         "ppv":ppv,
         "npv":npv,
         "ydi":ydi,
         "f1":f1,
         "hl":hl,
         "brier":brier,
         "cali_curve":cali_curve}
    return res,thr

def normalize_data(x,mode=0,axis=0):
    if mode ==0:
        min_val=np.min(x,axis=axis,keepdims=True)
        max_val=np.max(x,axis=axis,keepdims=True)
        x=(x-min_val)/(max_val-min_val)
    else:
        mu=np.mean(x,axis=axis,keepdims=True)
        std=np.std(x,axis=axis,keepdims=True)
        x=(x-mu)/std
    return x

def load_data():
    sccc_nr=pd.read_csv("data/sccc_nr_mice.csv",encoding="gbk")
    sccc_nr=sccc_nr.fillna(value=sccc_nr.mean(axis=0))
    xn_sccc=sccc_nr.values.astype(np.float32)
    yn_sccc=np.zeros(xn_sccc.shape[0],dtype=np.int32)
    
    sccc_gc=pd.read_csv("data/sccc_gc_mice.csv",encoding="gbk")
    sccc_cc=pd.read_csv("data/sccc_cc_mice.csv",encoding="gbk")
    sccc_gc=sccc_gc.fillna(value=sccc_gc.mean(axis=0))
    sccc_cc=sccc_cc.fillna(value=sccc_cc.mean(axis=0))
    xp_sccc=np.vstack((sccc_gc.values.astype(np.float32),
                       sccc_cc.values.astype(np.float32)))
    yp_sccc=np.ones(xp_sccc.shape[0],dtype=np.int32)
    
    
    zj_nr=pd.read_csv("data/zj_nr_mice.csv",encoding="gbk")
    zj_nr=zj_nr.fillna(value=zj_nr.mean(axis=0))
    xn_zj=zj_nr.values.astype(np.float32)
    yn_zj=np.zeros(xn_zj.shape[0],dtype=np.int32)    
    
    zj_gc=pd.read_csv("data/zj_gc_mice.csv",encoding="gbk")
    zj_cc=pd.read_csv("data/zj_cc_mice.csv",encoding="gbk") 
    zj_gc=zj_gc.fillna(value=zj_gc.mean(axis=0))
    zj_cc=zj_cc.fillna(value=zj_cc.mean(axis=0))
    xp_zj=np.vstack((zj_gc.values.astype(np.float32),
                     zj_cc.values.astype(np.float32)))
    yp_zj=np.ones(xp_zj.shape[0],dtype=np.int32)
    
    
    wh_nr=pd.read_csv("data/wh_nr_mice.csv",encoding="gbk")
    wh_nr=wh_nr.fillna(value=wh_nr.mean(axis=0))
    xn_wh=wh_nr.values.astype(np.float32)
    yn_wh=np.zeros(xn_wh.shape[0],dtype=np.int32)
    wh_gc=pd.read_csv("data/wh_gc_mice.csv",encoding="gbk")
    wh_cc=pd.read_csv("data/wh_cc_mice.csv",encoding="gbk")
    wh_gc=wh_gc.fillna(value=wh_gc.mean(axis=0))
    wh_cc=wh_cc.fillna(value=wh_cc.mean(axis=0))
    xp_wh=np.vstack((wh_gc.values.astype(np.float32),
                     wh_cc.values.astype(np.float32)))
    yp_wh=np.ones(xp_wh.shape[0],dtype=np.int32)
    
    cdfh_nr=pd.read_csv("data/cdfh_nr_mice.csv",encoding="gbk")
    cdfh_nr=cdfh_nr.fillna(value=cdfh_nr.mean(axis=0))
    xn_cdfh=cdfh_nr.values.astype(np.float32)
    yn_cdfh=np.zeros(xn_cdfh.shape[0],dtype=np.int32)
    
    cdfh_gc=pd.read_csv("data/cdfh_gc_mice.csv",encoding="gbk")
    cdfh_cc=pd.read_csv("data/cdfh_cc_mice.csv",encoding="gbk")
    cdfh_gc=cdfh_gc.fillna(value=cdfh_gc.mean(axis=0))
    cdfh_cc=cdfh_cc.fillna(value=cdfh_cc.mean(axis=0))
    xp_cdfh=np.vstack((cdfh_gc.values.astype(np.float32),
                       cdfh_cc.values.astype(np.float32)))
    yp_cdfh=np.ones(xp_cdfh.shape[0],dtype=np.int32)   
    
    col_names=zj_nr.columns
    
    x_train=np.vstack([xn_sccc,xp_sccc])
    y_train=np.hstack([yn_sccc,yp_sccc])
    
    x_test1=np.vstack([xn_zj,xp_zj])
    y_test1=np.hstack([yn_zj,yp_zj])
    
    x_test2=np.vstack([xn_wh,xp_wh])
    y_test2=np.hstack([yn_wh,yp_wh])
    
    x_test3=np.vstack([xn_cdfh,xp_cdfh])
    y_test3=np.hstack([yn_cdfh,yp_cdfh])
        
    mu=np.mean(x_train,axis=0,keepdims=True)
    std=np.std(x_train,axis=0,keepdims=True)    
    x_train=(x_train-mu)/std
    
    mu=np.mean(x_test1,axis=0,keepdims=True)
    std=np.std(x_test1,axis=0,keepdims=True)
    x_test1=(x_test1-mu)/std
    
    mu=np.mean(x_test2,axis=0,keepdims=True)
    std=np.std(x_test2,axis=0,keepdims=True)
    x_test2=(x_test2-mu)/std
    
    mu=np.mean(x_test3,axis=0,keepdims=True)
    std=np.std(x_test3,axis=0,keepdims=True)
    x_test3=(x_test3-mu)/std
    
    x_test=np.vstack([x_test1,x_test2,x_test3])
    y_test=np.hstack([y_test1,y_test2,y_test3])
    
    x_train,x_val,y_train,y_val=train_test_split(x_train,
                                                 y_train,
                                                 test_size=0.2,
                                                 random_state=1024)
    feat_mask=None
    if os.path.exists("data/feat_mask.csv"):
        feature_mask=np.loadtxt("data/feat_mask.csv",dtype=np.int32)
        feat_mask=feature_mask==1
        
    data={"factors":col_names,
          "x_train":x_train,
          "y_train":y_train,
          "x_val":x_val,
          "y_val":y_val,
          "x_test":x_test,
          "y_test":y_test,
          "x_test1":x_test1,
          "y_test1":y_test1,
          "x_test2":x_test2,
          "y_test2":y_test2,
          "x_test3":x_test3,
          "y_test3":y_test3,
          "feat_mask":feat_mask}
    
    return data
    
    
if __name__=="__main__":
#    x=np.random.randint(0,2,100)
#    xp=np.random.rand(100,2)
#    p,cali_curve=HosmerLemeshowTest(x,xp)
    data=load_data()
    
    
    
    
    
    








    
    
    
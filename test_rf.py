import os
import numpy as np
import pandas as pds
import pickle

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

import shap
from matplotlib import pyplot as plt
from utils import eval_clf,load_data

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus']=False

def train_model(data,save_dir):
    x_train=data["x_train"]
    y_train=data["y_train"]
    factors=data["factors"]
    
    if data["feat_mask"] is not None:
        mask=data["feat_mask"]
        x_train=x_train[:,mask]
        factors=factors[mask]
        
    clf=RandomForestClassifier(n_estimators=200,
                               criterion="gini",
                               max_depth=10,
                               bootstrap=True,
                               max_samples=1000,
                               class_weight="balanced")
    clf.fit(x_train,y_train)
    
    res,_=eval_clf(clf,x_train,y_train)
    print("auc={:.3f},acc={:.3f}".format(res["auc"],res["acc"]))    
    
    res_import = permutation_importance(clf, x_train, y_train)    
    plt.figure(21,figsize=(8,3))
    plt.bar(np.arange(len(res_import.importances_mean)),res_import.importances_mean) 
    plt.xticks(np.arange(len(res_import.importances_mean)),labels=factors,rotation=45)
    plt.show()
    
    with open(os.path.join(save_dir,'permutation_importance.csv'), 'w',encoding="gbk") as fid:
        for k in range(x_train.shape[1]):
            fid.writelines("{},{:.5f},{:.5f}\n".format(
                    factors[k],res_import.importances_mean[k],res_import.importances_std[k]))
#    SHAP    
#    explainer = shap.TreeExplainer(clf)
#    shap_values = explainer.shap_values(x_train)
#    shap_val=shap_values[1]
    
#    fig=plt.figure(22)  
#    fig.set_size_inches(12, 8)
#    shap.summary_plot(shap_val,x_train,feature_names=factors)
#    fig.savefig(os.path.join(save_dir,"shap_summary.png"),dpi=128,bbox_inches='tight')
    
#    fig=plt.figure(23) 
#    fig.set_size_inches(12, 8)
#    shap.summary_plot(shap_val,x_train,feature_names=factors,plot_type="bar")
#    fig.savefig(os.path.join(save_dir,"shap_importance.png"),dpi=128,bbox_inches='tight')
    
    with open(os.path.join(save_dir,'clf.pkl'), 'wb') as fid:
        pickle.dump(clf, fid)
    
    return 

def eval_model(data,save_dir,test_split=0):
    
    x_train=data["x_train"]
    y_train=data["y_train"]
    x_val=data["x_val"]
    y_val=data["y_val"]
    if test_split==0:
        x_test=data["x_test"]
        y_test=data["y_test"]
        save_dir1=save_dir
    elif test_split==1:
        x_test=data["x_test1"]
        y_test=data["y_test1"]
        save_dir1=os.path.join(save_dir,"zhejiang")
    elif test_split==2:
        x_test=data["x_test2"]
        y_test=data["y_test2"]
        save_dir1=os.path.join(save_dir,"wuhan")
    elif test_split==3:
        x_test=data["x_test3"]
        y_test=data["y_test3"]
        save_dir1=os.path.join(save_dir,"chengdu")
    if not os.path.exists(save_dir1):
        os.makedirs(save_dir1)
    
    factors=data["factors"]
    
    if data["feat_mask"] is not None:
        mask=data["feat_mask"]
        x_train=x_train[:,mask]
        x_val=x_val[:,mask]
        x_test=x_test[:,mask]
        factors=factors[mask]
        
    with open(os.path.join(save_dir,'clf.pkl'), 'rb') as fid:
        clf=pickle.load(fid) 
        
    clf=CalibratedClassifierCV(clf,cv="prefit").fit(x_train[::2,:],y_train[::2])
    res1,thr=eval_clf(clf,x_train,y_train)
    res2,_=eval_clf(clf,x_val,y_val,thr)
    res3,_=eval_clf(clf,x_test,y_test,thr)
    
    cali_curve=np.vstack((res1["cali_curve"],res2["cali_curve"],res3["cali_curve"]))
    np.savetxt(os.path.join(save_dir1,"cali_curve.csv"),cali_curve,fmt="%.5f",delimiter=",")
    
    roc_train=np.vstack((res1["roc"][0],res1["roc"][1]))
    np.savetxt(os.path.join(save_dir1,"roc_train.csv"),roc_train,fmt="%.5f",delimiter=",")
    
    roc_val=np.vstack((res2["roc"][0],res2["roc"][1]))
    np.savetxt(os.path.join(save_dir1,"roc_val.csv"),roc_val,fmt="%.5f",delimiter=",")   
    
    roc_test=np.vstack((res3["roc"][0],res3["roc"][1]))
    np.savetxt(os.path.join(save_dir1,"roc_test.csv"),roc_test,fmt="%.5f",delimiter=",")  
        
    with open(os.path.join(save_dir1,"clf_stats.csv"),"w",encoding="gbk") as fid:
        fid.writelines("Random Forest\n")
        fid.writelines("训练样本数,{}\n".format(x_train.shape[0]))
        fid.writelines("内部验证样本数,{}\n".format(x_val.shape[0]))
        fid.writelines("外部验证样本数,{}\n".format(x_test.shape[0]))
        fid.writelines("\n")
        
        fid.writelines(",AUC,ACC,SENS,SPEC,PPV,NPV,YDI,F1,HL,BRIER\n")   
        fid.writelines("{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f}\n".format(
                   "训练集",res1["auc"],res1["acc"],res1["sens"],res1["spec"],
                   res1["ppv"],res1["npv"],res1["ydi"],res1["f1"],res1["hl"],res1["brier"]))
        fid.writelines("{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f}\n".format(
                   "内部验证集",res2["auc"],res2["acc"],res2["sens"],res2["spec"],
                   res2["ppv"],res2["npv"],res2["ydi"],res2["f1"],res2["hl"],res2["brier"]))
        fid.writelines("{},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f},{:4f}\n".format(
                   "外部验证集",res3["auc"],res3["acc"],res3["sens"],res3["spec"],
                   res3["ppv"],res3["npv"],res3["ydi"],res3["f1"],res3["hl"],res3["brier"]))
    
    plt.figure(31,figsize=(15,4))
    plt.subplot(1,3,1)
    plt.plot(roc_train[0],roc_train[1],c="xkcd:blue")
    plt.title("auc={:.4f},acc={:.4f}".format(res1["auc"],res1["acc"]))
    plt.subplot(1,3,2)
    plt.plot(roc_val[0],roc_val[1],c="xkcd:blue")
    plt.title("auc={:.4f},acc={:.4f}".format(res2["auc"],res2["acc"]))
    plt.subplot(1,3,3)
    plt.plot(roc_test[0],roc_test[1],c="xkcd:blue")
    plt.title("auc={:.4f},acc={:.4f}".format(res3["auc"],res3["acc"]))
    plt.show()

    
    plt.figure(32,figsize=(15,4))
    plt.subplot(1,3,1)
    plt.plot(np.linspace(0,1,50),np.linspace(0,1,50),"k--")
    plt.plot(cali_curve[0,:],cali_curve[1,:],"-s",c="xkcd:blue",mfc="xkcd:blue")
    plt.title("hl={:.4f},brier={:.4f}".format(res1["hl"],res1["brier"]))
    plt.subplot(1,3,2)
    plt.plot(np.linspace(0,1,50),np.linspace(0,1,50),"k--")
    plt.plot(cali_curve[2,:],cali_curve[3,:],"-s",c="xkcd:blue",mfc="xkcd:blue")
    plt.title("hl={:.4f},brier={:.4f}".format(res2["hl"],res2["brier"]))
    plt.subplot(1,3,3)
    plt.plot(np.linspace(0,1,50),np.linspace(0,1,50),"k--")
    plt.plot(cali_curve[4,:],cali_curve[5,:],"-s",c="xkcd:blue",mfc="xkcd:blue")
    plt.title("hl={:.4f},brier={:.4f}".format(res2["hl"],res2["brier"]))
    plt.show()
    
    dca=res2["dca"]
    plt.figure(33,figsize=(8,6))
    plt.plot(dca[3,:],dca[0,:],"k--",label="Treat None")
    plt.plot(dca[3,:],dca[1,:],"k-",label="Treat All")
    plt.plot(dca[3,:],dca[2,:],"-",c="xkcd:blue",label="Random Forest")
    plt.ylim(-0.05,0.8)
    plt.legend(fontsize=12)
    plt.xlabel("Threshold",fontsize=12)
    plt.ylabel("Net Benifit",fontsize=12)
    plt.savefig(os.path.join(save_dir1,"dca_curve_val.png"),dpi=96)
    plt.show()
    
    dca=res3["dca"]
    plt.figure(34,figsize=(8,6))
    plt.plot(dca[3,:],dca[0,:],"k--",label="Treat None")
    plt.plot(dca[3,:],dca[1,:],"k-",label="Treat All")
    plt.plot(dca[3,:],dca[2,:],"-",c="xkcd:blue",label="Random Forest")
    plt.ylim(-0.05,0.8)
    plt.legend(fontsize=12)
    plt.xlabel("Threshold",fontsize=12)
    plt.ylabel("Net Benifit",fontsize=12)
    plt.savefig(os.path.join(save_dir1,"dca_curve_test.png"),dpi=96)
    plt.show()
    
    
    print("train: auc={:.3f},acc={:.3f}".format(res1["auc"],res1["acc"]))   
    print("val: auc={:.3f},acc={:.3f}".format(res2["auc"],res2["acc"])) 
    print("test: auc={:.3f},acc={:.3f}".format(res3["auc"],res3["acc"])) 
    return
    
 

if __name__=="__main__":
    ## Load data
    save_dir=r"results\rf"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    data=load_data()
    train_model(data,save_dir)
    eval_model(data,save_dir)
    eval_model(data,save_dir,1)
    eval_model(data,save_dir,2)
    eval_model(data,save_dir,3)
















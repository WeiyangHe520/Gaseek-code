import os
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier

import shap
from matplotlib import pyplot as plt
from utils import eval_clf,load_data

data=load_data()
x_train=data["x_train"]
x_val=data["x_val"]
x_test1=data["x_test1"]
x_test2=data["x_test2"]
x_test3=data["x_test3"]

x_all=np.vstack([x_train,x_val,x_test1,x_test2,x_test3])
factors=data["factors"]

for ik in range(x_train.shape[1]):
    min_val,max_val=np.percentile(x_all[:,ik],1),np.percentile(x_all[:,ik],99)
    bins=np.linspace(min_val,max_val,50)
    h1=np.histogram(x_train[:,ik],bins=bins)[0]
    h2=np.histogram(x_val[:,ik],bins=bins)[0]
    h3=np.histogram(x_test1[:,ik],bins=bins)[0]
    h4=np.histogram(x_test2[:,ik],bins=bins)[0]
    h5=np.histogram(x_test3[:,ik],bins=bins)[0]
    h1=h1/np.sum(h1)
    h2=h2/np.sum(h2)
    h3=h3/np.sum(h3)
    h4=h4/np.sum(h4)
    h5=h5/np.sum(h5)
    plt.figure(1,figsize=(10,8))
    plt.plot(h1,"k-",linewidth=2)
    plt.plot(h2,"k--",linewidth=2)
    plt.plot(h3,"r-",linewidth=2)
    plt.plot(h4,"g-",linewidth=2)
    plt.plot(h5,"b-",linewidth=2)
    plt.title(factors[ik])
    plt.savefig(f"FeatDistr_{ik}.png".replace("/","-"),dpi=96)
    plt.clf()
    plt.close()
    
    
    
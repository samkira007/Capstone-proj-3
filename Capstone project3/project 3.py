# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 03:22:49 2021

@author: sampa
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv ("C:\\Users\\sampa\\OneDrive\\Desktop\\project3\\test.csv")



df1=df.iloc[:,9:23]
print(df1.head())
x=df1.values
x_mean=np.mean(x,axis=0)
x_n=x-np.matrix(x_mean)
x_n=x_n.T
print(x_n.shape)

c1=np.cov(x_n)
c2=np.corrcoef(x_n)
ax=sns.heatmap(c2,cmap='Blues')

eig_val,eig_vec=np.linalg.eig(c1)
eig_sorted=np.sort(eig_val)[::-1]
arg_sort=np.argsort(eig_val)[::-1]

eig_vec_ls=[]
eig_val_ls=[]
imp_vec= arg_sort[:2]
for i in imp_vec:
    eig_vec_ls.append(eig_vec[:,i])
    eig_val_ls.append(eig_val[i])
print(eig_vec_ls)
print(eig_val_ls)

eig_val_arr= np.array(eig_val_ls)
lamda_1=np.diag(eig_val_arr)
print(lamda_1)
eig_vec_mat=np.matrix(eig_vec_ls).T
V=eig_vec_mat@np.sqrt(lamda_1)
print(V)

var_ls=[]
x_var = np.var(x_n,axis=1)
x_var=np.ravel(x_var)
print(x_var.shape)
print(x_var)
for i in range(V.shape[0]):
    s=np.sum(np.square(np.ravel(V[i,:])))
    sig_2=x_var[i]-s
    var_ls.append(sig_2)
    
var_ls=np.array(var_ls)
S=np.diag(var_ls)
print(S.shape)
print(S)

c1_inv=np.linalg.inv(c1)
W=V.T@c1_inv
print(W.shape)
print(W)

z=W@x_n
z1=z.T
plt.scatter(np.ravel(z1[:,0]),np.ravel(z1[:,1]))
plt.show

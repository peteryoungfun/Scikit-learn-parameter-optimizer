import Neighbor_points_optimization

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
  
#-----------------------------------------simple function test-----------------------------------------------
#target function
def cal_sum_r(a,b1,b2,c1,c2,d1,d2):
    try:
        C1=int(c1)
    except:
        print('C1 IS NOT AN INTEGER!')
    try:
        C2=int(c2)
    except:
        print('C2 IS NOT AN INTEGER!')
    if d1=="a":
        D=0
    elif d1=='b':
        D=5
    elif d1=='c':
        D=2
    elif d1=='d':
        D=-1
    elif d1=='e':
        D=10
    if d2 =='x':
        E=5
    elif d2 =='y':
        E=10
    elif d2== 'z':
        E=0
    sum=5*a**3-2*b1**4+100*b2*D-5*C1+2*C1**2*D-2*C2*E*2+C2**2
    return sum
#define feasible space of parameters:
continuous_list=[['a',[2,8]],['b1',[1,6]],['b2',[1,7]]]
int_list=[['c1',[1,2,3,4,5,6,7,8,9]],['c2',[3,5,7,9,11,13]]]
label_list=[['d1',['a','b','c','d','e']],['d2',['x','y','z']]]
#run
Neighbor_points_optimization.opt_neighbor(func=cal_sum_r,continuous_list=continuous_list,int_list=int_list,label_list=label_list,alpha=0.001,max_iter=100,min_improve=0.01,random_test_num=2)

#---------------------------------------test empty conditions------------------------------------------------------
#test empty conditions1:
def small_fun(d1,d2):
    if d1=="a":
        D=0
    elif d1=='b':
        D=5
    elif d1=='c':
        D=2
    elif d1=='d':
        D=-1
    elif d1=='e':
        D=10
    if d2 =='x':
        E=5
    elif d2 =='y':
        E=10
    elif d2== 'z':
        E=0
    return random.uniform(0,1)+D+E
con=[]
intl=[]
lab=[['d1',['a','b','c','d','e']],['d2',['x','y','z']]]
Neighbor_points_optimization.opt_neighbor(func=small_fun,continuous_list=con,int_list=intl,label_list=lab,alpha=0.001,max_iter=100,min_improve=0.01,random_test_num=3)
  
#test empty conditions2:
def small_fun(c1,c2):
    return c1**3+c2-3*c1**2
con=[]
intl=[['c1',range(0,20,3)],['c2',range(0,10,3)]]
lab=[]
Neighbor_points_optimization.opt_neighbor(func=small_fun,continuous_list=con,int_list=intl,label_list=lab,alpha=0.001,max_iter=100,min_improve=0.01,random_test_num=3)  
  
#test empty conditions3:
def small_fun(a1,a2):
    return a1**3+a2-3*a1**2
con=[['a1',[1,10]],['a2',[1,20]]]
intl=[]
lab=[]
Neighbor_points_optimization.opt_neighbor(func=small_fun,continuous_list=con,int_list=intl,label_list=lab,alpha=0.001,max_iter=100,min_improve=0.01,random_test_num=3)  

#test empty conditions4:
def small_fun():
    return random.uniform(0,1)
con=[]
intl=[]
lab=[]
Neighbor_points_optimization.opt_neighbor(func=small_fun,continuous_list=con,int_list=intl,label_list=lab,alpha=0.001,max_iter=100,min_improve=0.01,random_test_num=3) 


#----------------------------------------Apply in Sklearn----------------------------------------------------- 

x, y = make_classification(n_samples=2000,n_features=10,n_classes=2)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
y_train=np.array(y_train)
y_test=np.array(y_test)

#define feasible space
cl=[]
intl=[['min_samples_split',[2,3,4]],['n_estimators',[100,200,300,400,500]],["min_samples_split",[2,3,4]]]
labl=[['criterion',['gini','entropy']],['max_features',['sqrt','log2']]]

#define target function
def rf_f1(**para):
    return f1_score(y_test,RandomForestClassifier(**para).fit(X_train,y_train).predict(X_test),average='micro')
#run  
Neighbor_points_optimization.opt_neighbor(func=rf_f1,continuous_list=cl,int_list=intl,label_list=labl,alpha=0.001,max_iter=100,min_improve=0.001,random_test_num=1)

import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog
import os, glob
from math import floor

directory=filedialog.askdirectory()
NFILE=len(glob.glob1(directory,"*.csv"))

LOWERBOUND=int(NFILE*10/100)
UPPERBOUND=NFILE-LOWERBOUND

mean_matrix=np.zeros((301,14))
mean_trimmed=np.zeros((301,14))
median_matrix=np.zeros((301,14,NFILE))
batch_matrix=np.zeros((301,14,5))
mean_of_batch=np.zeros((301,14))
median_of_batch=np.zeros((301,14,int(NFILE/5)+1))
os.makedirs(directory+"\\media", exist_ok=True)

##################### mean e raccolta dati
for i in range(NFILE):
    data = np.loadtxt(directory+"\\data"+str(i%NFILE)+".csv", delimiter=",")
    for c in range(1,15):
        for r in range(301):
            mean_matrix[r,c-1]+=data[r,c]
            median_matrix[r,c-1,i]=data[r,c]
            batch_matrix[r,c-1,i%5]=data[r,c]    
            if i%5==4:
                mean_of_batch[r,c-1]+=batch_matrix[r,c-1,batch_matrix[r, 0,:].argsort()[0]]
                median_of_batch[r,c-1,floor(i/5)]=batch_matrix[r,c-1,batch_matrix[r, 0,:].argsort()[0]]

for c in range(14):
        for r in range(301):
            mean_matrix[r,c]=mean_matrix[r,c]/NFILE
mean_matrix=np.c_[data[:,0],mean_matrix]
np.savetxt(
            directory+"\\media\\data0.csv",
            mean_matrix,
            delimiter=",", 
            fmt=['%d','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f'], 
            header="iter,cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias")

       
# ordina i dati in base ad una delle colonne 1=cost
median_matrix=median_matrix[:,:,median_matrix[r,0,:].argsort()]
median_of_batch=median_of_batch[:,:,median_of_batch[r,0,:].argsort()]

for i in range(NFILE):
    for c in range(14):
        for r in range(301):
            if(i>LOWERBOUND and i<UPPERBOUND):
                mean_trimmed[r,c]+=median_matrix[r,c,i]

variance_matrix=np.zeros((301,14))
for i in range(NFILE):
    data = np.loadtxt(directory+"\\data"+str(i%NFILE)+".csv", delimiter=",")
    for c in range(1,15):
        for r in range(301):
            variance_matrix[r,c-1]+=pow(data[r,c]-mean_matrix[r,c],2)

for c in range(14):
        for r in range(301):
            variance_matrix[r,c]=variance_matrix[r,c]/NFILE
variance_matrix=np.c_[data[:,0],variance_matrix]
np.savetxt(
            directory+"\\media\\variance.csv",
            variance_matrix,
            delimiter=",", 
            fmt=['%d','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f'], 
            header="iter,cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias")




for c in range(14):
        for r in range(301):
            mean_trimmed[r,c]=mean_trimmed[r,c]/(UPPERBOUND-LOWERBOUND)
mean_trimmed=np.c_[data[:,0],mean_trimmed]
np.savetxt(
            directory+"\\media\\data1.csv",
            mean_trimmed,
            delimiter=",", 
            fmt=['%d','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f'], 
            header="iter,cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias")




median_matrix=np.c_[data[:,0],median_matrix[:,:,int(NFILE/2)]]
np.savetxt(
            directory+"\\media\\data2.csv",
            median_matrix,
            delimiter=",", 
            fmt=['%d','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f'], 
            header="iter,cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias")



for c in range(14):
        for r in range(301):
            mean_of_batch[r,c]=mean_of_batch[r,c]/(NFILE/5)
mean_of_batch=np.c_[data[:,0],mean_of_batch]

np.savetxt(
            directory+"\\media\\data3.csv",
            mean_of_batch,
            delimiter=",", 
            fmt=['%d','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f'], 
            header="iter,cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias")

median_of_batch=np.c_[data[:,0],median_of_batch[:,:,int(NFILE/10)]]
np.savetxt(
            directory+"\\media\\data4.csv",
            median_of_batch,
            delimiter=",", 
            fmt=['%d','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f'], 
            header="iter,cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias")

# cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias

# iter,cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias

from matplotlib.axes import Axes
from tkinter import filedialog
import os, os.path,glob, numpy as np, matplotlib.pyplot as plt

directory=filedialog.askdirectory()

MAX_LINE_WIDTH=0.5
DATA_LINE_WIDTH=0.8

#iter,cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias
def plot(directory):
    data = np.loadtxt(directory+"\\data0.csv", delimiter=",")
    var = np.loadtxt(directory+"\\variance.csv", delimiter=",")
    fig.suptitle("media")

    X=data[:,0]
    cost=data[:,1]
    bias=data[:,-1]
    for c in range(COLS):
        for r in range(ROWS):
            ax[r][c].clear()

    position=2
    for c in range(COLS):
        for r in range(ROWS):
            ax[r,c].set_title(ax[r][c].get_label())
            ax[r,c].plot(X,cost,c="yellow",linewidth=DATA_LINE_WIDTH)
            ax[r,c].fill_between(X,cost-var[:,1],cost+var[:,1],linewidth=0,color="#ffffe0")
            
            ax[r,c].plot(X,bias,c="orange",linewidth=DATA_LINE_WIDTH)
            ax[r,c].fill_between(X,bias-var[:,-1],bias+var[:,-1],linewidth=0,color="#FFD580")

            ax[r,c].plot(X,data[:,position],color="blue",linewidth=DATA_LINE_WIDTH)
            ax[r,c].plot(X,data[:,position+1],color="darkgreen", linewidth=DATA_LINE_WIDTH)
            ax[r,c].fill_between(X,data[:,position]-var[:,position],data[:,position]+var[:,position],linewidth=0,color="lightblue")
            ax[r,c].set_ylim(0,1)
            ax[r,c].set_ybound(-0.1,1.1)
            position+=2

    
fig=plt.figure(layout="compressed")


# inp_c=input("scegliere cost function (c: cross entropy, s: sigmoid)")
# inp_e=int(input("scegliere esecuzione"))
inp_s=input("salvare immagini (y: si n: no)")

ROWS=3
COLS=2

ax=np.empty((3,2), dtype=Axes)

labels=["Accuracy","auc ROC","F1","auc PR","Precision","Recall"]
for c in range(COLS):
    for r in range(ROWS):
        x_pos=0.1+0.06*(c%COLS)+0.37*c
        y_pos=1-(0.07+0.25*(r+1)+0.06*(r%ROWS))
        ax[r,c]=fig.add_axes(rect=(x_pos,y_pos,0.37, 0.25),fc="gray",label=labels[r+c*ROWS])
        ax[r,c].set_autoscaley_on(False)
        
NFILE=len(glob.glob1(directory,"*.csv"))


if(inp_s=='y'):
    for l in range(NFILE):
        plot(directory,l)
        d=directory+"\\imgs\\data"+str(l)+".pdf"
        fig.set_figwidth(16)
        fig.set_figheight(9)
        os.makedirs(directory+"\\imgs", exist_ok=True)
        fig.savefig(d,format="pdf")

plot(directory)
fig.canvas.draw()

plt.show()
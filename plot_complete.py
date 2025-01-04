from matplotlib.axes import Axes
from tkinter import filedialog
import os, os.path,glob, numpy as np, matplotlib.pyplot as plt

directory=filedialog.askdirectory()

MAX_LINE_WIDTH=0.5
DATA_LINE_WIDTH=0.8
i=0
def event(event):
    global i
    #print(event.key)
    if event.key=='left':
        i= i-1
    elif event.key=="right":
        i= i+1
    else: 
        print(event.key)
        return

    plot(directory,i)
    fig.canvas.draw()
#iter,cost,accuracy_train,accuracy_val,auc_train,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val,precision_train,precision_val,recall_train,recall_val,bias
def plot(directory,i):
    data = np.loadtxt(directory+"\\data"+str(i%NFILE)+".csv", delimiter=",")

    fig.suptitle("iterazione "+str(i%NFILE)+" esecuzione")
    X=data[:,0]
    cost=data[:,1]
    bias=data[:,-1]
    for c in range(COLS):
        for r in range(ROWS):
            ax[r][c].clear()

    position=2
    legend_lables=["iter","cost","accuracy_train","accuracy_val","auc_train","auc_val","f1_train","f1_val","auc_pr_train","auc_pr_val","precision_train","precision_val","recall_train","recall_val","bias"]
    for c in range(COLS):
        for r in range(ROWS):
            ax[r,c].set_title(ax[r][c].get_label())
            cost_artist, = ax[r,c].plot(X,cost,c="yellow",linewidth=DATA_LINE_WIDTH)
            bias_artist, = ax[r,c].plot(X,bias,c="orange",linewidth=DATA_LINE_WIDTH)
            train_artist, = ax[r,c].plot(X,data[:,position],color="blue",linewidth=DATA_LINE_WIDTH)
            val_artist, = ax[r,c].plot(X,data[:,position+1],color="darkgreen", linewidth=DATA_LINE_WIDTH)
            ax[r,c].fill_between(X,data[:,position],data[:,position+1],linewidth=0,color="limegreen")
            ax[r,c].set_ylim(0,1)
            ax[r,c].set_ybound(-0.1,1.1)
            if(c<COLS/2):
                ax[r,c].legend(
                    handles=[train_artist,val_artist,cost_artist,bias_artist],
                    labels=["train", "valuation", "cost", "bias"],
                    loc='upper left',
                    bbox_to_anchor=(1.05, 1),
                    )
            position+=2

    
fig=plt.figure(layout="compressed")

fig.canvas.manager.set_window_title(directory)
# inp_c=input("scegliere cost function (c: cross entropy, s: sigmoid)")
# inp_e=int(input("scegliere esecuzione"))
inp_s=input("salvare immagini (y: si n: no)")

ROWS=3
COLS=2

ax=np.empty((3,2), dtype=Axes)

labels=["Accuracy","auc ROC","F1","auc PR","Precision","Recall"]
for c in range(COLS):
    for r in range(ROWS):
        x_pos=0.06+0.16*(c%COLS)+0.37*c
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

plot(directory,i)
fig.canvas.draw()

fig.canvas.mpl_connect('key_press_event', event)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from screeninfo import get_monitors
import os, os.path
from sys import exit

parent_dir = os.path.dirname(os.getcwd())+"\dati\\"

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

    if inp_c=='s' and inp_e==1:
        plot_exec1(directory,i)
    else:
        plot_exec2(directory,i)
    fig.canvas.draw()

def plot_exec1(directory,i):
    data = np.loadtxt(directory+"\data"+str(i%NFILE)+".csv", delimiter=",")
    X=data[:,0]

    cost=data[:,1]
    train_acc=data[:,2]
    val_acc=data[:,3]
    
    # ax=fig.add_axes(rect=(1/8,1/8, 6/8,3/8),fc="gray")
    # ax2=fig.add_axes(rect=(1/8,4/8, 6/8,3/8),fc="gray")
    
    ax.clear()
    ax2.clear()
    ax2.set_title("iterazione "+str(i%NFILE))
    ax.plot(X,cost,c="yellow",linewidth=DATA_LINE_WIDTH)
    ax2.plot(train_acc, color="blue",linewidth=DATA_LINE_WIDTH)
    ax2.fill_between(X,train_acc,val_acc,linewidth=0,color="limegreen")
    ax2.plot(val_acc, color="forestgreen", linewidth=DATA_LINE_WIDTH)

    ax2.vlines(train_acc.argmax(),0,1,colors="black",linestyles="dashed",linewidth=MAX_LINE_WIDTH)
    ax2.hlines(max(train_acc),0,300,colors="black",linestyles="dashed",linewidth=MAX_LINE_WIDTH)
    ax.vlines(train_acc.argmax(),0,1,colors="black",linestyles="dashed",linewidth=MAX_LINE_WIDTH)
    
    ax2.vlines(val_acc.argmax(),0,1,colors="black",linestyles="dotted",linewidth=MAX_LINE_WIDTH)
    ax2.hlines(max(val_acc),0,300,colors="black",linestyles="dotted",linewidth=MAX_LINE_WIDTH)
    ax.vlines(val_acc.argmax(),0,1,colors="black",linestyles="dotted",linewidth=MAX_LINE_WIDTH)

    ax.set_xticks([train_acc.argmax(),val_acc.argmax()])
    ax2.set_yticks([max(train_acc),max(val_acc)])

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

def plot_exec2(directory,i):
    data = np.loadtxt(directory+"\data"+str(i%NFILE)+".csv", delimiter=",")
    #iter,cost,accuracy_train,auc_train,accuracy_val,auc_val
    X=data[:,0]
    cost=data[:,1]
    train_acc=data[:,2]
    auc_train=data[:,3]
    val_acc=data[:,4]
    auc_val=data[:,5]

    ax.clear()
    ax2.clear()
    ax3.clear()

    ax2.set_title("iterazione "+str(i%NFILE)+" esecuzione "+str(inp_e)+" "+costf)

    ax.plot(X,cost,c="yellow",linewidth=DATA_LINE_WIDTH)

    training_acc_handler = ax2.plot(train_acc, color="blue",linewidth=DATA_LINE_WIDTH)
    val_acc_handler = ax2.plot(val_acc, color="darkgreen", linewidth=DATA_LINE_WIDTH)
    ax2.fill_between(X,train_acc,val_acc,linewidth=0,color="limegreen")

    ax3.plot(auc_train, color="blue",linewidth=DATA_LINE_WIDTH)
    ax3.plot(auc_val, color="darkgreen", linewidth=DATA_LINE_WIDTH)
    ax3.fill_between(X,auc_train,auc_val,linewidth=0,color="limegreen")


    ax2.vlines(train_acc.argmax(),0,1,colors="black",linestyles="dashed",linewidth=MAX_LINE_WIDTH)
    ax2.hlines(max(train_acc),0,300,colors="black",linestyles="dashed",linewidth=MAX_LINE_WIDTH)
    ax.vlines(train_acc.argmax(),0,1,colors="black",linestyles="dashed",linewidth=MAX_LINE_WIDTH)
    
    ax2.vlines(val_acc.argmax(),0,1,colors="black",linestyles="dotted",linewidth=MAX_LINE_WIDTH)
    ax2.hlines(max(val_acc),0,300,colors="black",linestyles="dotted",linewidth=MAX_LINE_WIDTH)
    ax.vlines(val_acc.argmax(),0,1,colors="black",linestyles="dotted",linewidth=MAX_LINE_WIDTH)


    legend_acc= ax2.legend([training_acc_handler,val_acc_handler],labels=["training accuracy","valuation accuracy"])
    legend_acc.set_loc('lower right')

    legend_cost = ax.legend(labels=["cost function"])
    legend_cost.set_loc('lower right')
    legend_auc= ax3.legend([training_acc_handler,val_acc_handler],labels=["training ROC auc","valuation ROC auc"])
    legend_auc.set_loc('lower right')

    ax.set_xticks([train_acc.argmax(),val_acc.argmax()])
    ax2.set_yticks([max(train_acc),max(val_acc)])

    # ax.set_xticks([train_acc.argmax(),val_acc.argmax()])
    # ax.set_yticks([max(train_acc),max(val_acc)])
    ax2.text(train_acc.argmax(), max(train_acc), max(train_acc), fontsize=7, horizontalalignment='right', verticalalignment='bottom')
    ax2.text(val_acc.argmax(), max(val_acc), max(val_acc), fontsize=7, horizontalalignment='right', verticalalignment='bottom')


fig=plt.figure(layout="compressed")

inp_c=input("scegliere cost function (c: cross entropy, s: sigmoid)")
inp_e=int(input("scegliere esecuzione"))
inp_s=input("salvare immagini (y: si n: no)")


if inp_c=='s' and inp_e==1:
    directory=parent_dir+"sigLoss\exec1\data"
    ax=fig.add_axes(rect=(1/8,1/8, 6/8,3/8),fc="gray")
    ax2=fig.add_axes(rect=(1/8,4/8, 6/8,3/8),fc="gray")

    NFILE=len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])-1
    plot_exec1(directory,i)
    fig.canvas.draw()
elif type(inp_e) is int:
    if inp_c=='s':
        costf="sigLoss"
    elif inp_c=='c':
        costf="crossEntropy"
    directory=parent_dir+costf+"\exec"+str(inp_e)
    ax3=fig.add_axes(rect=(1/8,1/11, 6/8,3/11),fc="gray")
    ax=fig.add_axes(rect=(1/8,4/11, 6/8,3/11),fc="gray")
    ax2=fig.add_axes(rect=(1/8,7/11, 6/8,3/11),fc="gray")

    NFILE=len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])-1
    
    if(inp_s=='y'):
        for l in range(NFILE):
            plot_exec2(directory,l)
            d=directory+"\imgs\data"+str(l)+".pdf"
            fig.set_figwidth(16)
            fig.set_figheight(9)
            fig.savefig(d,format="pdf")
    
    plot_exec2(directory,i)
    fig.canvas.draw()
else: exit()


fig.canvas.mpl_connect('key_press_event', event)
plt.show()


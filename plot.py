import numpy as np
import matplotlib.pyplot as plt
from screeninfo import get_monitors

from sys import exit

MAX_LINE_WIDTH=0.5
DATA_LINE_WIDTH=0.8
i=0
def event(event):
    global i
    #print(event.key)
    if event.key=='left':
        i= i-1
    if event.key=="right":
        i= i+1
    else: 
        print(event.key)
        return

    if inp==1:
        plot_exec1(i)
    elif inp==2:
        plot_exec2(i)

def plot_exec1(i):
    data = np.loadtxt("C:\AAAAvscode\TesiQSVM\dati\exec1\\data"+str(i%10)+".csv", delimiter=",")
    X=data[:,0]

    cost=data[:,1]
    train_acc=data[:,2]
    val_acc=data[:,3]
    
    # ax=fig.add_axes(rect=(1/8,1/8, 6/8,3/8),fc="gray")
    # ax2=fig.add_axes(rect=(1/8,4/8, 6/8,3/8),fc="gray")
    
    ax.clear()
    ax2.clear()
    ax2.set_title("iterazione "+str(i%10))
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

    fig.canvas.draw()

    # figManager = plt.get_current_fig_manager()
    # figManager.window.showMaximized()

def plot_exec2(i):
    data = np.loadtxt("C:\AAAAvscode\TesiQSVM\dati\exec2\\data"+str(i%10)+".csv", delimiter=",")
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

    ax2.set_title("iterazione "+str(i%10))

    ax.plot(X,cost,c="yellow",linewidth=DATA_LINE_WIDTH)

    ax2.plot(train_acc, color="blue",linewidth=DATA_LINE_WIDTH)
    ax2.fill_between(X,train_acc,val_acc,linewidth=0,color="limegreen")
    ax2.plot(val_acc, color="forestgreen", linewidth=DATA_LINE_WIDTH)

    ax3.plot(auc_train, color="blue",linewidth=DATA_LINE_WIDTH)
    ax3.fill_between(X,auc_train,auc_val,linewidth=0,color="limegreen")
    ax3.plot(auc_val, color="forestgreen", linewidth=DATA_LINE_WIDTH)


    ax2.vlines(train_acc.argmax(),0,1,colors="black",linestyles="dashed",linewidth=MAX_LINE_WIDTH)
    ax2.hlines(max(train_acc),0,300,colors="black",linestyles="dashed",linewidth=MAX_LINE_WIDTH)
    ax.vlines(train_acc.argmax(),0,1,colors="black",linestyles="dashed",linewidth=MAX_LINE_WIDTH)
    
    ax2.vlines(val_acc.argmax(),0,1,colors="black",linestyles="dotted",linewidth=MAX_LINE_WIDTH)
    ax2.hlines(max(val_acc),0,300,colors="black",linestyles="dotted",linewidth=MAX_LINE_WIDTH)
    ax.vlines(val_acc.argmax(),0,1,colors="black",linestyles="dotted",linewidth=MAX_LINE_WIDTH)

    ax.set_xticks([train_acc.argmax(),val_acc.argmax()])
    ax2.set_yticks([max(train_acc),max(val_acc)])

    fig.canvas.draw()

fig=plt.figure(layout="compressed")
inp=int(input("sceglie esecuzione"))
if inp==1:
    ax=fig.add_axes(rect=(1/8,1/8, 6/8,3/8),fc="gray")
    ax2=fig.add_axes(rect=(1/8,4/8, 6/8,3/8),fc="gray")

    plot_exec1(i)
elif inp==2:
    ax3=fig.add_axes(rect=(1/8,1/11, 6/8,3/11),fc="gray")
    ax=fig.add_axes(rect=(1/8,4/11, 6/8,3/11),fc="gray")
    ax2=fig.add_axes(rect=(1/8,7/11, 6/8,3/11),fc="gray")
    plot_exec2(i)
else: exit()


    

fig.canvas.mpl_connect('key_press_event', event)
plt.show()
input()

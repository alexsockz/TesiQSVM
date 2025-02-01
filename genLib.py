import pennylane as qml
from pennylane import numpy as np
from sklearn.preprocessing import normalize
from numpy import array, genfromtxt
from sklearn.utils import resample
#da modificare in base al tipo di entanglement che vogliamo, questo è ciclico
def ansatz(weights, iterations, dimensions):
    for wire in range(dimensions):
        qml.RZ(weights[0*dimensions+wire*2+0], wires=wire)
        qml.RY(weights[0*dimensions+wire*2+1], wires=wire)

    for layer in range(1,iterations):
        ansatz_ent("full_swap",dimensions)
        for wire in range(dimensions):
            qml.RZ(weights[layer*dimensions+wire*2+0], wires=wire)
            qml.RY(weights[layer*dimensions+wire*2+1], wires=wire)
        qml.Barrier(range(dimensions))

    #da modificare in base al tipo di entanglement che vogliamo, questo è ciclico
    
def ansatz_ent(ent,dimensions):
    if ent=="circular":
        for wires in range(dimensions):
            qml.CNOT([(wires-1)%dimensions,wires]) 
    if ent=="linear":
        for wires in range(dimensions-1):
            qml.CNOT([wires,wires+1]) 
    if ent=="full_swap":
         for i in range(dimensions-1):
             for j in range(i,dimensions-1):
                 qml.SWAP([i,j+1])

# i dati devono essere (0,2pi]^dim
# noi vogliamo che la feature map mappi sull'intera base di pauli Z, 

# questo come possiamo notare theta viene dimezzato e quindi per poter mantenere la rotazione su 2pi è necessario compensare
def feature_map_layer(record,dimensions):
    for wires in range(dimensions):
        qml.Hadamard(wires)#applica hadamard su tutti i bit
    for wires in range(dimensions):
        phi=(record[wires])
        qml.PhaseShift(phi, wires=wires)
    feature_map_ent("full",record,dimensions)

def feature_map_ent(ent, record,dimensions):
    if ent=="circular":
        for wires in range(dimensions):
            prv=(wires-1)%dimensions
            phi=(np.pi-record[prv])*(np.pi-record[wires])
            qml.CNOT([prv, wires])
            qml.PhaseShift(phi, wires=wires)
            qml.CNOT([prv, wires])
    if ent=="linear":
        for wires in range(dimensions-1):
            nxt=(wires+1)%dimensions
            phi=(np.pi-record[nxt])*(np.pi-record[wires])
            qml.CNOT([wires, nxt])
            qml.PhaseShift(phi, wires=nxt)
            qml.CNOT([wires, nxt])
    if ent=="full":
        for target in range(1,dimensions):
            for ctrl in range(target):
                phi=(np.pi-record[ctrl])*(np.pi-record[target])
                qml.CNOT([ctrl, target])
                qml.PhaseShift(phi, wires=target)
                qml.CNOT([ctrl, target])

def print_circuit(weights,x,circuit):
    drawer = qml.draw(circuit)
    print(drawer(weights, x))

#assumo data sia un multi dimensional dataset
def normalize2pi(data):
    norm= (normalize(data, norm="l2",axis=1)+1)*np.pi
    return norm

#da sistemare, non ha senso?
def accuracy(correct, predictions):
    acc=0
    for l, p in zip(correct, predictions):
        if l==p:  
            acc +=1
    acc = acc / len(correct)
    return acc

#NOTE: è SOLO PER IL CASO BINARIO
def precision_recall(correct,predictions):
    tp=0
    fp=0
    fn=0
    tn=0
    for c,p in zip(correct, predictions):
        if c==p and c==1:
            tp=tp+1
        elif c!=p and c==1:
            fp=fp+1
        elif c!=p and c!=1:
            fn=fn+1
        elif c==p and c!=1:
            tn=tn+1
    return [tp/(tp+fp), tn/(tn+fn)], [tp/(tp+fn), tn/(tn+fp)]

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """

    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

def get_iris():
    data = genfromtxt("iris3.csv",delimiter=",", dtype=str)
    X = np.array(data[:, :-1], dtype=float)
    #REMEMBER: devo indicare il tipo della label, cerco di essere il piu generico possibile in modo da classificare anche stringhe
    Y = array(data[:, -1], dtype=float)
    
    return X,Y
def get_breast_cancer_data():
    data = genfromtxt("breast-cancer-wisconsin2.data", delimiter=",", dtype=str)
    X = np.array(data[:, 1:-1], dtype=float)
    Y = array(data[:, -1], dtype=float)
    Y=Y-3 #shift label from {2, 4} to {-1, 1}
    return X,Y

def get_blood_transfer_data():
    data = genfromtxt("transfusion.data",delimiter=",", dtype=str)
    X = np.array(data[:, :-1], dtype=float)
    #REMEMBER: devo indicare il tipo della label, cerco di essere il piu generico possibile in modo da classificare anche stringhe
    Y = array(data[:, -1], dtype=float)
    Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}
    return X,Y

def get_occupancy():
    data=genfromtxt("occupancy.csv",delimiter=",", dtype=str)

    majority_class=data[data[:,-1]=="0"]
    print(len(majority_class))
    minority_class=data[data[:,-1]=="1"]
    print(len(minority_class))
    majority_downsampled = resample(majority_class, 
                                replace=False,  # Sample without replacement
                                n_samples=len(minority_class),  # Equalize class sizes
                                random_state=42)
    # Combine the downsampled majority class with the minority class

    df_balanced = np.concatenate([majority_downsampled[:50,:], minority_class[:50,:]])

    X = np.array(df_balanced[:, :-1], dtype=float)
    Y = array(df_balanced[:, -1], dtype=float)
    Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}
    return X,Y
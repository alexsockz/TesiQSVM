from sklearn.model_selection import train_test_split
import pennylane as qml
from pennylane import numpy as np
import numpy 
from sklearn.preprocessing import normalize
from math import floor

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
    norm= normalize(data, norm="l2",axis=0)
    print(norm)
    return norm

#da sistemare, non ha senso?
def accuracy(labels, predictions):
    acc=0
    for l, p in zip(labels, predictions):
        if l==p:  
            acc +=1
    acc = acc / len(labels)
    return acc


# import genLib as GL
# from pennylane import numpy as np
# import pennylane as qml

# DIMENSIONS=4
# NUM_LAYERS=4
# dev = qml.device("lightning.qubit", wires=DIMENSIONS)
# dev2 = qml.device("lightning.qubit", shots=1000, wires=DIMENSIONS)
# @qml.qnode(dev)
# def circuit_expval(weights, x):
#     for _ in range(2):
#         GL.feature_map_layer(x,DIMENSIONS)
#         qml.Barrier(range(DIMENSIONS))
#     GL.ansatz(weights, NUM_LAYERS, DIMENSIONS)
#     #metodo vecchio
#     # for i in range(DIMENSIONS):
#     #     qml.measure(i)
#     #return qml.expval(qml.PauliZ(0))
#     return (qml.expval(qml.Y(0)))

# @qml.qnode(dev2)
# def circuit(weights, x):
#     for _ in range(2):
#         GL.feature_map_layer(x,DIMENSIONS)
#         qml.Barrier(range(DIMENSIONS))
#     GL.ansatz(weights, NUM_LAYERS, DIMENSIONS)
#     #metodo vecchio
#     # for i in range(DIMENSIONS):
#     #     qml.measure(i)
#     #return qml.expval(qml.PauliZ(0))
#     return (qml.counts(qml.Y(0))) 

# weights_init= np.zeros((NUM_LAYERS+1)*DIMENSIONS*2,requires_grad=True)

# print(circuit_expval(weights_init, [0.08351291,0.09887112,0.04681674,0.02069458]))
# print(circuit(weights_init, [0.08351291,0.09887112,0.04681674,0.02069458]))

import autograd.numpy as np  # O la libreria che stai usando
from autograd import grad

SHOTS = 1000
samples = {'A': 400.0, 'B': 600.0}

Y=1

# def get_sample_emp_distribution(weights,x):
#         #equivalente nella cross entropy, consulto il circuito R volte e alcune volte ritorna -1 altre 1
#         samples=variational_classifier(weights,x, [-1,1])
#         #\ calcolo la distribuzione di probabilità che sia 1 o -1
#         return {k:float(v)/float(SHOTS) for k,v in samples.items()}

#tengo single thread perchè può tornare utile su architetture in cui ci sono troppi pochi core
def sig_cost_function_single_thread(X, Y):
    R_emp=0
        #probabilità che la label assegnata sia diversa da quella corretta, va minimizzato 
    R_emp+=[float(v) / SHOTS for k, v in samples.items()][Y]
    return R_emp/len(X) #cost function classica, do ad ogni sample lo stesso peso
    

grad_loss = grad(sig_cost_function_single_thread)
print(grad_loss(samples,Y))
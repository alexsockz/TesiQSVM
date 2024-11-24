import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals

dev = qml.device("default.qubit", shots=10)
def ansatz(weights, iterations):
    for wire in range(dimensions):
        qml.RZ(weights[0,wire,0], wires=wire)
        qml.RY(weights[0,wire,1], wires=wire)

    for layer_weights in weights[1:]:
        ansatz_layer(layer_weights)
        qml.Barrier(range(dimensions))
    
def ansatz_layer(layer_weights):
    #da modificare in base al tipo di entanglement che vogliamo, questo è ciclico
    ansatz_ent("linear")

    for wire in range(dimensions):
        qml.RZ(layer_weights[wire,0], wires=wire)
        qml.RY(layer_weights[wire,1], wires=wire)

def ansatz_ent(ent):
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
# la rotazione avviene grazie ad un gate di rotazione:
# Math: e^{i\lambda}\sin\left(\frac{\theta}{2}\right) e^{i\phi}\sin\left(\frac{\theta}{2}\right) &amp; e^{i(\phi+\lambda)}\cos\left(\frac{\theta}{2}\right)\end{pmatrix}

# questo come possiamo notare theta viene dimezzato e quindi per poter mantenere la rotazione su 2pi è necessario compensare


def feature_map_layer(record):
    for wires in range(dimensions):
        qml.Hadamard(wires)#applica hadamard su tutti i bit
    for wires in range(dimensions):
        phi=2.0*(record[wires])
        qml.PhaseShift(phi, wires=wires)
    feature_map_ent("linear",record)

def feature_map_ent(ent, record):
    if ent=="circular":
        for wires in range(dimensions):
            prv=(wires-1)%dimensions
            phi=2.0*(np.pi-record[prv])*(np.pi-record[wires])
            qml.CNOT([prv, wires])
            qml.PhaseShift(phi, wires=wires)
            qml.CNOT([prv, wires])
    if ent=="linear":
        for wires in range(dimensions-1):
            nxt=(wires+1)%dimensions
            phi=2*(np.pi-record[nxt])*(np.pi-record[wires])
            qml.CNOT([wires, nxt])
            qml.PhaseShift(phi, wires=nxt)
            qml.CNOT([wires, nxt])

#CANCELLARE: esempio proveniente da tutorial, non necessario in qvc dato che i dati vengono caricati in altro modo
def state_preparation(x):
    qml.BasisState(x, wires=[0, 1, 2, 3])

@qml.qnode(dev)
def circuit(weights, x):
    state_preparation(0)

    for _ in range(2):
        feature_map_layer(x)
        qml.Barrier(range(dimensions))
    ansatz(weights, num_layers)
    # for i in range(dimensions):
    #     qml.measure(i)

    #return qml.expval(qml.PauliZ(0))
    return (qml.counts()) 

def print_circuit(weights,x):
    drawer = qml.draw(circuit)
    print(drawer(weights, x))

def variational_classifier(weights, bias, x):
    return circuit(weights, x) + bias

#assumo data sia un multi dimensional dataset
def normalize2pi(data):
    return normalize(data, norm="max",axis=0)*2*np.pi

def square_loss_with_noise(labels, predictions):
    L= np.mean((labels - np.sign(qml.math.stack(predictions))) ** 2)
    noise = 5*np.random.random()
    return L+noise

def square_loss(labels, predictions):
    # We use a call to qml.math.stack to allow subtracting the arrays directly
    return np.mean((labels - np.sign(qml.math.stack(predictions))) ** 2)

def sigmoid_loss(labels,predictions):
    predictions = labels,qml.math.stack(predictions)
    L=np.sig((0.5))

#è sbagliata??
def cross_entropy_loss(labels, predictions):
    
    return -np.mean(
        np.log(
            (1+np.multiply(labels,qml.math.stack(predictions))
                /2)
            )
        )

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc

def cost(weights, bias, X, Y):
    predictions=[variational_classifier(weights, bias, x)for x in X]
    #print(qml.math.stack(predictions))
    #puoi scegliere quale loss function utilizzare
    return square_loss(Y, predictions)

# modificare in caso serva una nuova grad func
# def grad(weights, bias, X, Y):
#     predictions = [variational_classifier(weights, bias, x) for x in X]
    

################### main ###################
#np.random.seed(12345)
data = np.genfromtxt("iris2.csv",delimiter=",", dtype="float")
X = np.array(data[:, :-1])
Y = np.array(data[:, -1])

print("prima riga pre normalizzazione",X[0],"   ",Y[0])
features=normalize2pi(X)
print("prima riga post normalizzazione", features[0],"   ", Y[0])
num_data = len(Y)
if len(Y)==len(features): print("corrette dimesioni")

train_perc=0.75

feats_train, feats_val, Y_train, Y_val = train_test_split(
    features, Y, train_size=train_perc, random_state=algorithm_globals.random_seed
)

#dipende dai dati
#Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}

# print tutti i dati
# for x,y in zip(X, Y):
#     print(f"x = {x}, y = {y}")        


#setup pesi
dimensions = len(X[0])
num_layers = 4

#come descritto nel paper, limito le rotazioni su Pauli Z e Y, nel 
weights_init = 0.01 * np.random.randn(num_layers+1, dimensions, 2, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

print("Weights:", weights_init)
print("Bias: ", bias_init)

opt = NesterovMomentumOptimizer(0.1)
batch_size = 5

weights = weights_init
bias = bias_init
print(feats_train[0])
print_circuit(weights, feats_train[0])

input("Press Enter to continue...")

max_iterations=60
for it in range(max_iterations): ## da cambiare in "fino a che non converge l'ottimizzatore"
    for x in feats_train:
        #dovrei contare, in R shots, quale è quello che capita di piu e mapparlo ad una label,
        #l'ottenimento del max è gia ottenuto nel metodo circuit
        d=circuit(weights, x)
        print(max(d))
    input("rpreprepr")

# for it in range(max_iterations):
#     #CAMBIA METODO DI SAMPLING
#     # non separiamo un gruppo di training e un gruppo di test,
#     # prendiamo solo un sottoinsieme di dati da X e li utilizziamo per valutare i pesi, 
#     # ad ogni passo aumentiamo o diminuiamo di uno step secondo la objective function sul batch
#     batch_index = np.random.randint(0, len(feats_train), (batch_size))
#     X_batch = feats_train[batch_index]
#     Y_batch = Y_train[batch_index]

#     # #prima lo traino sul batch- poi confronto su tutto l'insieme giusto per visualizzare
#     weights, bias = opt.step(cost, weights, bias, X=X_batch, Y=Y_batch)

#     # accuratezza: prima testo su tutti i dati
#     predictions_train = [np.sign(variational_classifier(weights, bias, x)) for x in feats_train]
#     predictions_val = [np.sign(variational_classifier(weights, bias, x)) for x in feats_val]

#     # Compute accuracy on train and validation set
#     acc_train = accuracy(Y_train, predictions_train)
#     acc_val = accuracy(Y_val, predictions_val)

#     if (it + 1) % 2 == 0:
#         _cost = cost(weights, bias, features, Y)
#         print(
#             f"Iter: {it + 1:5d} | Cost: {_cost:0.7f} | "
#             f"Acc train: {acc_train:0.7f} | Acc validation: {acc_val:0.7f}"
#         )


# predictions_test = [np.sign(variational_classifier(weights, bias, x)) for x in feats_val]

# for x,y,p in zip(feats_val, Y_val, predictions_test):
#     print(f"x = {x}, y = {y}, pred={p}")

# acc_test = accuracy(Y_val, predictions_test)
# print("Accuracy on unseen data:", acc_test)
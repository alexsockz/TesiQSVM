import pennylane as qml
from pennylane import numpy as np
import numpy 
from pennylane.optimize import NesterovMomentumOptimizer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from qiskit_algorithms.utils import algorithm_globals
from copy import deepcopy
from collections import Counter
from math import floor

shots=100
dev = qml.device("default.qubit", shots=shots)
def ansatz(weights, iterations):
    for wire in range(dimensions):
        qml.RZ(weights[0*dimensions+wire*2+0], wires=wire)
        qml.RY(weights[0*dimensions+wire*2+1], wires=wire)

    for layer in range(1,iterations):
        ansatz_ent("linear")
        for wire in range(dimensions):
            qml.RZ(weights[layer*dimensions+wire*2+0], wires=wire)
            qml.RY(weights[layer*dimensions+wire*2+1], wires=wire)
            qml.Barrier(range(dimensions))

    #da modificare in base al tipo di entanglement che vogliamo, questo è ciclico
    

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
    #metodo vecchio
    # for i in range(dimensions):
    #     qml.measure(i)
    #return qml.expval(qml.PauliZ(0))
    return (qml.counts()) 

def print_circuit(weights,x):
    drawer = qml.draw(circuit)
    print(drawer(weights, x))

def variational_classifier(weights, x, labels):
    if len(x.shape)==1:
        results= circuit(weights, x)
        #for string in results:
        return mapper(results, labels)
    else:
        raise TypeError("x must be a 1 dim list (a data element)")


def mapper(q_strings,labels):
    #varierà a seconda dei dati, qui in pratica sto usando la parità per decidere se un risultato è +1(se pari) o -1(dispari),
    # questa parte è a piacere e non è molto rilevante
    # la misurazione di pennylane da {0,1}, altri potrebbero dare {-1,1}
    #metodi che useremo "Parity" (per binari, se pari 1 se dispari -1), "modulo" (per n label fai bitstring%n)
    count={el:0 for el in labels}
    n_per_classe=pow(2,dimensions)/len(labels)
    for key, val in q_strings.items():
        count[floor(int(key,2)/n_per_classe)]+=val
    return count
    
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

def softmax_max(x, beta=1.0):
    weights = np.exp(beta * x)  # Calcolo dei pesi esponenziali
    weighted_sum = np.sum(x * weights)  # Somma ponderata dei valori
    weight_total = np.sum(weights)  # Somma dei pesi
    return weighted_sum / weight_total

def prob_x_div_corretto(pmf,correct):
    #dobbiamo capire quale è la probabilità che il predict di x sia diverso dalla sua corretta label e minimizzarlo
    py=pmf[correct]
    rest=[i for i in pmf if i!=correct]
    print(correct, pmf, py, rest)
    f=np.sqrt(shots)*((max(rest)-py)/np.sqrt(2*(1-py)*py))
    p=1/(1+np.exp(-f))
    return p
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

def cost_function(*weights, X, Y):
    R_emp=0
    for x,y in zip(X,Y):
        emp_distribution= [(i/shots)+0.0001 for i in variational_classifier(weights,x, classi)]
        R_emp+=prob_x_div_corretto(emp_distribution,y)
        print(R_emp)
    return R_emp/len(Y)

def old_cost(weights, bias, X, Y):
    predictions=[variational_classifier(weights, bias, x)for x in X]
    #print(qml.math.stack(predictions))
    #puoi scegliere quale loss function utilizzare
    return square_loss(Y, predictions)

def run_optimizer(opt, cost_function, init_param, num_steps, interval, execs_per_step):
    # Copy the initial parameters to make sure they are never overwritten
    weights,X,Y = init_param.copy()

    # Initialize the memory for cost values during the optimization
    cost_history = []
    # Monitor the initial cost value
    cost_history.append(cost_function(weights,X,Y))
    exec_history = [0]

    print(
        f"\nRunning the {opt.__class__.__name__} optimizer for {num_steps} iterations."
    )
    for step in range(num_steps):
        # Print out the status of the optimization
        if step % 2 == 0:
            print(
                f"Step {step:3d}: Circuit executions: {exec_history[step]:4d}, "
                f"Cost = {cost_history[step]}"
            )

        # Perform an update step
        weights,_,_ = opt.step(cost_function, weights, X,Y)
        #print(weights)
        # Monitor the cost value
        cost_history.append(cost_function(weights,X,Y))
        exec_history.append((step + 1) * execs_per_step)

    print(
        f"Step {num_steps:3d}: Circuit executions: {exec_history[-1]:4d}, "
        f"Cost = {cost_history[-1]}"
    )
    return cost_history, exec_history

# modificare in caso serva una nuova grad func
# def grad(weights, bias, X, Y):
#     predictions = [variational_classifier(weights, bias, x) for x in X]


################### main ###################
#np.random.seed(12345)
data = np.genfromtxt("iris2.csv",delimiter=",", dtype="float")
X = np.array(data[:, :-1])
Y = numpy.array(data[:, -1], dtype=int)

classi= [y for y in set(Y)]

print("prima riga pre normalizzazione",X[0],"   ",Y[0])
features=normalize2pi(X)
print("prima riga post normalizzazione", features[0],"   ", Y[0])
num_data = len(Y)
if len(Y)==len(features): print("corrette dimesioni")

train_perc=0.75

feats_train, feats_val, Y_train, Y_val = train_test_split(
    features, Y, train_size=train_perc, random_state=algorithm_globals.random_seed
)
occurrences_Y=Counter(Y_train)
#dipende dai dati
#Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}

# print tutti i dati
# for x,y in zip(X, Y):
#     print(f"x = {x}, y = {y}")        

#setup pesi
dimensions = len(X[0])
num_layers = 4

#come descritto nel paper, limito le rotazioni su Pauli Z e Y, nel 
weights_init = 0.01 * np.random.randn((num_layers+1)*dimensions*2, requires_grad=True)
bias_init = np.array(0.0, requires_grad=True)

print("Weights:", weights_init)
print("Bias: ", bias_init)

batch_size = 5

weights = weights_init
bias = bias_init
print(feats_train[0])
print_circuit(weights, feats_train[0])

input("Press Enter to continue...")

max_iterations=60
#matrice di grandezza 2^n x C, ovvero le possibili stringhe create dal circuito x il numero di classi
#utilizzo in realtà un array di dizionari perchè piu universale, cosi non sono limitato a numeri da 0 a C come etichette
print(occurrences_Y)

#for it in range(max_iterations): ## da cambiare in "fino a che non converge l'ottimizzatore"
num_steps_spsa =100
opt= qml.SPSAOptimizer(num_steps_spsa)

print(weights)

w,f,y = opt.step(cost_function, *weights, feats_train, Y_train)
print(w==weights, f==feats_train, y==Y_train)

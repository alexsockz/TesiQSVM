from sklearn import metrics
import pennylane as qml
from pennylane import numpy as np
import numpy 
from sklearn.preprocessing import normalize
from math import floor
from sklearn.model_selection import train_test_split

shots=100
dev = qml.device("default.qubit", shots=shots)

data = numpy.genfromtxt("iris2.csv",delimiter=",", dtype=str)
X = np.array(data[:, :-1], dtype=float)
#REMEMBER: devo indicare il tipo della label, cerco di essere il piu generico possibile in modo da classificare anche stringhe
Y = numpy.array(data[:, -1])

train_perc=0.75
num_data = len(Y)
dimensions = len(X[0])
num_layers = 4
batch_size = 5
max_iterations=60
num_steps_spsa =100

#TODO: sto facendo un mix strano tra ausare la label come numero e come chiave, va rifatto
#NOTE: set non è utilizzabile perchè non mantiene l'ordine delle label per qualche motivo
classi= set(Y)

print(classi)

def ansatz(weights, iterations):
    for wire in range(dimensions):
        qml.RZ(weights[0*dimensions+wire*2+0], wires=wire)
        qml.RY(weights[0*dimensions+wire*2+1], wires=wire)

    for layer in range(1,iterations):
        ansatz_ent("full_swap")
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

@qml.qnode(dev)
def circuit(weights, x):
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
    #matrice di grandezza 2^n x C, ovvero le possibili stringhe create dal circuito x il numero di classi
#utilizzo in realtà un array di dizionari perchè piu universale, cosi non sono limitato a numeri da 0 a C come etichette

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
        count[labels[floor(int(key,2)/n_per_classe)]]+=val
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
    rest=np.array([x for i,x in enumerate(pmf) if i!=correct])
    #print(correct, pmf, py, rest)
    f=np.sqrt(shots)*((softmax_max(rest)-py)/np.sqrt(2*(1-py)*py))
    p=1/(1+np.exp(-f))
    return p


def sig_cost_function(weights, X, Y):
    R_emp=0
    for x,y in zip(X,Y):
        emp_distribution= [(i/shots) for i in variational_classifier(weights,x, classi).values()]
        R_emp+=prob_x_div_corretto(emp_distribution,y)
        #print(prob_x_div_corretto(emp_distribution,y))
        #print(R_emp)
    return R_emp/len(Y)

def old_cost(weights, bias, X, Y):
    predictions=[variational_classifier(weights, bias, x)for x in X]
    #print(qml.math.stack(predictions))
    #puoi scegliere quale loss function utilizzare
    return square_loss(Y, predictions)

#è sbagliata??
def cross_entropy_cost(weights, X, Y):
    R_emp=0
    for x,y in zip(X,Y):
        emp_distribution= [(i/shots) for i in variational_classifier(weights,x, classi).values()]
        R_emp-=np.log(emp_distribution[y])
    #loss
    return R_emp/len(Y)

def accuracy(labels, predictions):
    acc = sum(abs(l - p) < 1e-5 for l, p in zip(labels, predictions))
    acc = acc / len(labels)
    return acc


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
        if step % 10 == 0:
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
    return cost_history, exec_history, weights

################### main ###################

print("prima riga pre normalizzazione",X[0],"   ",Y[0])
features=normalize2pi(X)
print("prima riga post normalizzazione", features[0],"   ", Y[0])
if len(Y)==len(features): print("corrette dimesioni")

#suddivisione 

feats_train, feats_val, Y_train, Y_val = train_test_split(
    features, Y, train_size=train_perc, random_state=42
)

# feats_train=features
# Y_train=Y

print(feats_train)


#in caso serva in futuro avere, per qualche mappatura particolare
#occurrences_Y=Counter(Y_train)

#dipende dai dati
#Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}

# print tutti i dati
# for x,y in zip(X, Y):
#     print(f"x = {x}, y = {y}")        

########## setup pesi ##########

#come descritto nel paper, limito le rotazioni su Pauli Z e Y,
#uso una lista anziche una matrice
weights_init = 2*np.pi*np.random.random((num_layers+1)*dimensions*2,requires_grad=True) # forse non serve 
#solo per dati binari
#bias_init = np.array(0.0, requires_grad=True)

print("Weights:", weights_init)
#print("Bias: ", bias_init)

weights = weights_init
#bias = bias_init


############# PRINT PER CONTROLLARE IL CIRCUITO #######################
print(feats_train[0])
print_circuit(weights, feats_train[0])

input("Press Enter to continue...")

######################## ZONA DI OBLIO DOVE TUTTO VIENE MODIFICATO ###########################

opt= qml.SPSAOptimizer(num_steps_spsa)

# test=[0.294, 0.564, 0.142]
# py=test[1]
# rest=[x for i,x in enumerate(test) if i!=1]
# print(py)
# print(max(rest))
# f=np.sqrt(shots)*((max(rest)-py)/np.sqrt(2*(1-py)*py))
# p=1/(1+np.exp(-f))
# print(p)

pred=numpy.empty(len(feats_train))
for i in range(len(feats_train)):
    emp=variational_classifier(weights,feats_train[i],classi)
    print(emp)
    l=max(zip(emp.values(), emp.keys()))[1]
    pred[i]=l
confusion_matrix = metrics.confusion_matrix(Y_train, pred)
print(confusion_matrix)

cost_history_spsa, exec_history_spsa, weights = run_optimizer(
opt, cross_entropy_cost, [weights,feats_train,Y_train], num_steps_spsa, 20, 1
)

# weights = SPSA(LossFunction = lambda parameters: cost_function(parameters, feats_train, Y_train),
#                   parameters = weights)

pred=numpy.empty(len(feats_train))
for i in range(len(feats_train)):
    emp=variational_classifier(weights,feats_train[i],classi)
    print(emp)
    pred[i]=int(max(zip(emp.values(), emp.keys()))[1])
    
print(accuracy(Y_train,pred))

# print(weights)

# print(weights==weights2)
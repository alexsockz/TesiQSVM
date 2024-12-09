from sklearn.model_selection import train_test_split
from sklearn import metrics
import pennylane as qml
from pennylane import numpy as np
import numpy 
from sklearn.preprocessing import normalize
from math import floor


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
batch_size = 20
max_iterations=60
num_steps_spsa =300

#TODO: sto facendo un mix strano tra ausare la label come numero e come chiave, va rifatto
#NOTE: set non è utilizzabile perchè non mantiene l'ordine delle label per qualche motivo
#TODO: fixare, per ora scrivo io le lable
classi=numpy.array(['0','1','2'])

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
        phi=(record[wires])
        qml.PhaseShift(phi, wires=wires)
    feature_map_ent("full",record)

def feature_map_ent(ent, record):
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

def softmax_max(d, beta=1.0):
    x=np.array(list(d.values()))
    weights = np.exp(beta * x)  # Calcolo dei pesi esponenziali
    weighted_sum = np.sum(x * weights)  # Somma ponderata dei valori
    weight_total = np.sum(weights)  # Somma dei pesi
    return weighted_sum / weight_total

def prob_x_div_corretto(pmf,correct):
    #dobbiamo capire quale è la probabilità che il predict di x sia diverso dalla sua corretta label e minimizzarlo
    rest=pmf.copy()
    py=rest.pop(correct)
    #print(correct, pmf, py, rest)
    f=np.sqrt(shots)*((softmax_max(rest)-py)/np.sqrt(2*(1-py)*py+1e-10))
    p=1/(1+np.exp(-f))
    return p


def sig_cost_function(weights, X, Y):
    R_emp=0
    for x,y in zip(X,Y):
        samples=variational_classifier(weights,x, classi)
        emp_distribution= {k:v/shots for k,v in samples.items()}
        R_emp+=prob_x_div_corretto(emp_distribution,y)
        #print(prob_x_div_corretto(emp_distribution,y))
        #print(R_emp)
    return R_emp/len(X)

def old_cost(weights, bias, X, Y):
    predictions=[variational_classifier(weights, bias, x)for x in X]
    #print(qml.math.stack(predictions))
    #puoi scegliere quale loss function utilizzare
    return square_loss(Y, predictions)

#TODO: rifattorizzare, tra emp distribution diventa una lista anziche rimanere un array, in pratica inizio a trustare ch ela posizione sia la stessa della label
def cross_entropy_cost(weights, X, Y):
    R_emp=0
    for x,y in zip(X,Y):
        samples=variational_classifier(weights,x, classi)
        emp_distribution= {k:v/shots for k,v in samples.items()}
        if(emp_distribution[y]==0):
            print("errore")
        R_emp-=np.log(emp_distribution[y]+1e-5)
    #loss
    return R_emp/len(X)

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
    cost=cost_function(weights,X,Y)
    cost_history.append(cost)
    exec_history = [0]
    f=open('dump.txt', 'w')
    print(
        f"\nRunning the {opt.__class__.__name__} optimizer for {num_steps} iterations."
    )
    step=0
    while cost> 0.09:
        # Print out the status of the optimization
        if step % 10 == 0:
            print(
                f"Step {step:3d}: Circuit executions: {exec_history[step]:4d}, "
                f"Cost = {cost_history[step]}"
            )
            f.write(f"Step {step:3d}: Circuit executions: {exec_history[step]:4d}, "
                f"Cost = {cost_history[step]}")

        # batch_index = np.random.randint(0, len(X), (batch_size,))
        # feats_train_batch = X[batch_index]
        # Y_train_batch = Y[batch_index]
        # Perform an update step
        # weights, _, _ = opt.step(cost_function, weights, feats_train_batch, Y_train_batch)

        weights, _, _ = opt.step(cost_function, weights, X, Y)
        #print(weights)
        # Monitor the cost value
        cost=cost_function(weights,X,Y)
        cost_history.append(cost)
        exec_history.append((step + 1) * execs_per_step)
        step+=1

    print(
        f"Step {num_steps:3d}: Circuit executions: {exec_history[-1]:4d}, "
        f"Cost = {cost_history[-1]}"
    )
    f.write(f"Step {num_steps:3d}: Circuit executions: {exec_history[-1]:4d}, "
        f"Cost = {cost_history[-1]}")
    return cost_history, exec_history, weights

def grad(L, w, ck):
     
    # number of parameters
    p = len(w)
     
    # bernoulli-like distribution
    deltak = np.random.choice([-1, 1], size=p)
     
    # simultaneous perturbations
    ck_deltak = ck * deltak
 
    # gradient approximation
    up=L(w + ck_deltak)
    down=L(w - ck_deltak)
    DELTA_L = up - down 
    print("-------------------------")
    print(up)
    print(down)
    return (DELTA_L) / (2 * ck_deltak)

def initialize_hyperparameters(alpha, lossFunction, w0, N_iterations):
 
    c = 0.5  # a small number
 
    # A is <= 10% of the number of iterations
    A = N_iterations*0.1
    
    # order of magnitude of first gradients
    magnitude_g0 = np.abs(grad(lossFunction, w0, c).mean())
    print(magnitude_g0)
    # the number 2 in the front is an estimative of
    # the initial changes of the parameters,
    # different changes might need other choices
    a = 2*((A+1)**alpha)/(magnitude_g0+1e-10)
 
    return a, A, c

def SPSA(LossFunction, parameters, alpha=0.602, gamma=0.101, N_iterations=10):
     
    # model's parameters
    w = parameters
    a, A, c = initialize_hyperparameters(
      alpha, LossFunction, w, N_iterations)
    for k in range(1, N_iterations):
 
        # update ak and ck
        ak = a/((k+A)**(alpha))
        ck = c/(k**(gamma))
        # estimate gradient
        gk = grad(LossFunction, w, ck)
        # update parameters
        w -= ak*gk
 
    return w

def confusion_matrix(x, y, labels):
    pred=numpy.empty(len(x),dtype=str)
    for i in range(len(x)):
        emp=variational_classifier(weights,x[i],labels)
        l=max(zip(emp.values(), emp.keys()))[1]
        pred[i]=l
    confusion_matrix = metrics.confusion_matrix(y, pred)
    return confusion_matrix

################### main ###################

print("prima riga pre normalizzazione",X[0],"   ",Y[0])
features=normalize2pi(X)
print("prima riga post normalizzazione", features[0],"   ", Y[0])
if num_data==len(features): print("corrette dimesioni")

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
#TODO: probabilmente i pesi iniziali sono un problema
weights_init = np.random.random_sample((num_layers+1)*dimensions*2,requires_grad=True)-0.5 # forse non serve 
#solo per dati binari
#bias_init = np.array(0.0, requires_grad=True)

print("Weights:", weights_init)
#print("Bias: ", bias_init)

weights = weights_init
#bias = bias_init


############# PRINT PER CONTROLLARE IL CIRCUITO #######################
print(feats_train[0])
print_circuit(weights, feats_train[0])

#input("Press Enter to continue...")

######################## ZONA DI OBLIO DOVE TUTTO VIENE MODIFICATO ###########################

opt= qml.GradientDescentOptimizer(num_steps_spsa,)

# test=[0.294, 0.564, 0.142]
# py=test[1]
# rest=[x for i,x in enumerate(test) if i!=1]
# print(py)
# print(max(rest))
# f=np.sqrt(shots)*((max(rest)-py)/np.sqrt(2*(1-py)*py))
# p=1/(1+np.exp(-f))
# print(p)

confusion_matrix(feats_train,Y_train,classi)

cost_history_spsa, exec_history_spsa, weights = run_optimizer(
opt, cross_entropy_cost, [weights,feats_train,Y_train], num_steps_spsa, 20, 1
)

# weights = SPSA(LossFunction = lambda parameters: cross_entropy_cost(parameters, feats_train, Y_train),
#                    parameters = weights)

confusion_matrix(feats_train,Y_train,classi)


# print(weights)

# print(weights==weights2)

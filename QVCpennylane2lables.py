from sklearn.model_selection import train_test_split
import pennylane as qml
from pennylane import numpy as np
import numpy 
import genLib as GL
import csv


shots=1000
dev = qml.device("default.qubit", shots=shots)

data = numpy.genfromtxt("iris3.csv",delimiter=",", dtype=str)
X = np.array(data[:, :-1], dtype=float)
#REMEMBER: devo indicare il tipo della label, cerco di essere il piu generico possibile in modo da classificare anche stringhe
Y = numpy.array(data[:, -1], dtype=float)

train_perc=0.75
num_data = len(Y)
dimensions = len(X[0])
num_layers = 4
batch_size = 20
num_steps_spsa =300

@qml.qnode(dev)
def circuit(weights, x):
    for _ in range(2):
        GL.feature_map_layer(x,dimensions)
        qml.Barrier(range(dimensions))
    GL.ansatz(weights, num_layers, dimensions)
    #metodo vecchio
    # for i in range(dimensions):
    #     qml.measure(i)
    #return qml.expval(qml.PauliZ(0))
    return (qml.counts(qml.Y(0))) 

def variational_classifier(weights, x, labels):
#matrice di grandezza 2^n x C, ovvero le possibili stringhe create dal circuito x il numero di classi
#utilizzo in realtà un array di dizionari perchè piu universale, cosi non sono limitato a numeri da 0 a C come etichette
    if len(x.shape)==1:
        results= circuit(weights, x)
        return results
    else:
        raise TypeError("x must be a 1 dim list (a data element)")

def prob_x_div_corretto(pmf,correct, bias):
    #dobbiamo capire quale è la probabilità che il predict di x sia diverso dalla sua corretta label e minimizzarlo
    try:
        py=pmf[correct]
    except:
        print(pmf)
        py=0
        #print(correct, pmf, py, rest)
    num=(0.5-(py)-((correct*bias)/2))
    den=np.sqrt(2*(1-py)*py)
    if den==0: print("errore")
    f=np.sqrt(shots)*(num)/(den+1e-10)
    p=1/(1+np.exp(-f))
    return p

def sig_cost_function(weights, bias, X, Y):
    R_emp=0
    for x,y in zip(X,Y):
        samples=variational_classifier(weights,x, [-1,1])
        emp_distribution= {k:v/shots for k,v in samples.items()}
        R_emp+=prob_x_div_corretto(emp_distribution, y, bias)
        #print(prob_x_div_corretto(emp_distribution,y))
    return R_emp/len(X)


def predict(X, weights):
    pred=numpy.empty(len(X),dtype=int)
    for i in range(len(X)):
        emp=variational_classifier(weights,X[i],[-1,1])
        l=int(max(zip(emp.values(), emp.keys()))[1])
        pred[i]=l
    return pred

def run_optimizer(opt, cost_function, init_param, feats_val,Y_val, num_steps, interval, execs_per_step):
    # Copy the initial parameters to make sure they are never overwritten
    weights, bias, X, Y = init_param.copy()

    # Initialize the memory for cost values during the optimization
    cost_history = []
    accuracy_train_history=[]
    accuracy_val_history=[]
    # Monitor the initial cost value
    cost=cost_function(weights,bias, X,Y)
    cost_history.append(cost)
    exec_history = [0]
    pred_train=predict(X,weights)
    accuracy_train=GL.accuracy(Y_train,pred_train)
    
    accuracy_train_history.append(accuracy_train)

    pred_val=predict(feats_val,weights)
    accuracy_val=GL.accuracy(Y_val,pred_val)
    
    accuracy_val_history.append(accuracy_val)

    print(
        f"\nRunning the {opt.__class__.__name__} optimizer for {num_steps} iterations."
    )
    step=0
    for i in range(num_steps):
        # Print out the status of the optimization
        if step % 10 == 0:
            print(
                f"Step {step:3d}: Circuit executions: {exec_history[step]:4d}, "
                f"Cost = {cost_history[step]}"
            )
            print(bias)
            print(
                f"Step {step:3d}: Circuit executions: {exec_history[step]:4d}, "
                f"accuracy = {accuracy_train_history[step]}"
            )

        # batch_index = np.random.randint(0, len(X), (batch_size,))
        # feats_train_batch = X[batch_index]
        # Y_train_batch = Y[batch_index]
        # Perform an update step
        # weights, _, _ = opt.step(cost_function, weights, feats_train_batch, Y_train_batch)

        weights, bias, _, _ = opt.step(cost_function, weights, bias, X, Y)
        #print(weights)
        # Monitor the cost value
        cost=cost_function(weights, bias, X,Y)
        cost_history.append(cost)
        exec_history.append((step + 1) * execs_per_step)

        pred_train=predict(X,weights)
        accuracy_train=GL.accuracy(Y_train,pred_train)
        
        accuracy_train_history.append(accuracy_train)

        pred_val=predict(feats_val,weights)
        accuracy_val=GL.accuracy(Y_val,pred_val)
        
        accuracy_val_history.append(accuracy_val)
        step+=1


    print(
        f"Step {num_steps:3d}: Circuit executions: {exec_history[-1]:4d}, "
        f"Cost = {cost_history[-1]}"
    )
    return cost_history, exec_history, accuracy_train_history, accuracy_val_history, weights, bias


################### main ###################

print("prima riga pre normalizzazione",X[0],"   ",Y[0])
features=GL.normalize2pi(X)
print("prima riga post normalizzazione", features[0],"   ", Y[0])
if num_data==len(features): print("corrette dimesioni")

#suddivisione 
for i in range(10):
    feats_train, feats_val, Y_train, Y_val = train_test_split(
        features, Y, train_size=train_perc
    )

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
    # probabilmente i pesi iniziali sono un problema
    #weights_init = np.random.random_sample((num_layers+1)*dimensions*2,requires_grad=True) # forse non serve 

    weights_init= np.zeros((num_layers+1)*dimensions*2,requires_grad=True)
    #solo per dati binari
    bias_init = np.array(0.0, requires_grad=True)

    print("Weights:", weights_init)
    print("Bias: ", bias_init)

    weights = weights_init
    bias = bias_init


    ############# PRINT PER CONTROLLARE IL CIRCUITO #######################
    print(feats_train[0])
    GL.print_circuit(weights, feats_train[0],circuit)

    #input("Press Enter to continue...")

    ######################## ZONA DI OBLIO DOVE TUTTO VIENE MODIFICATO ###########################

    opt= qml.SPSAOptimizer(num_steps_spsa,)


    cost_history, exec_history, accuracy_train_history, accuracy_val_history, weights, bias = run_optimizer(
    opt, sig_cost_function, [weights,bias,feats_train,Y_train], feats_val,Y_val, num_steps_spsa, 20, 1
    )

    a= list(zip(exec_history, cost_history, accuracy_train_history, accuracy_val_history))
    numpy.savetxt("data"+str(i)+".csv",a, delimiter=",", fmt=['%d','%.11f','%.11f','%.11f'], header="exec_history,cost_history,accuracy_train_history,accuracy_val_history")
    # print(weights)

    # print(weights==weights2)

import pennylane as qml
from pennylane import numpy as np
import numpy 

SHOTS=1000
data = numpy.genfromtxt("iris3.csv",delimiter=",", dtype=str)
X = np.array(data[:, :-1], dtype=float)
#REMEMBER: devo indicare il tipo della label, cerco di essere il piu generico possibile in modo da classificare anche stringhe
Y = numpy.array(data[:, -1], dtype=float)

train_perc=0.75
NUM_DATA = len(Y)
DIMENSIONS = len(X[0])
NUM_LAYERS = 4
BATCH_SIZE = 20
NUM_STEPS_SPSA =300
COST_WITH_BIAS=True

dev = qml.device("lightning.qubit", shots=SHOTS, wires=DIMENSIONS)
@qml.qnode(dev)
def circuit(weights, x):
    for _ in range(2):
        GL.feature_map_layer(x,DIMENSIONS)
        qml.Barrier(range(DIMENSIONS))
    GL.ansatz(weights, NUM_LAYERS, DIMENSIONS)
    #metodo vecchio
    # for i in range(DIMENSIONS):
    #     qml.measure(i)
    #return qml.expval(qml.PauliZ(0))
    return (qml.counts(qml.Y(0))) 

################# GLI IMPORT DEVONO STARE DOPO LA DICHIARAZIONE DEL CIRCUITO PER QUALCHE MOTIVO ###########################################
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
import genLib as GL
import multiprocessing as mp
from math import ceil
import time

################# NECESSARI PER THREAD ###########################################
def variational_classifier(weights, x, labels):
#matrice di grandezza 2^n x C, ovvero le possibili stringhe create dal circuito x il numero di classi
#utilizzo in realtà un array di dizionari perchè piu universale, cosi non sono limitato a numeri da 0 a C come etichette
    if len(x.shape)==1:
        results= circuit(weights, x)
        return results
    else:
        raise TypeError("x must be a 1 dim list (a data element)")

def get_sample_emp_distribution(weights,x):
        #equivalente nella cross entropy, consulto il circuito R volte e alcune volte ritorna -1 altre 1
        samples=variational_classifier(weights,x, [-1,1])
        #\ calcolo la distribuzione di probabilità che sia 1 o -1
        return {k:v/SHOTS for k,v in samples.items()}

def prob_x_div_corretto(pmf,correct, bias):
    #dobbiamo capire quale è la probabilità che il predict di x sia diverso dalla sua corretta label e minimizzarlo
        #pmf è un dict della forma {-1:n_predict_negativi, 1: n_predict_positivi}
        #pmf contiene la probabilità che, dato un sample x, esso sia -1 o 1
        #correct è un singolo int 1/-1 che indica la label corretta secondo il dataset
    try:
        #qui prendo la probabilità che sia la label corretta secondo il dataset
        py=pmf[correct]
    except IndexError:
        #se il circuito per puro caso ritorna un dict della forma {-1:#} o {1:#} allora 
        # rischia di esserci un IndexOutOfBounds, in tal caso semplicemente so che la prob è 0
        print(pmf)
        py=0
    
    num=(0.5-(py)-((correct*bias)/2)) #numeratore
    den=np.sqrt(2*(1-py)*py) #denominatore
    
    f=np.sqrt(SHOTS)*(num)/(den+1e-10) #1e-10 serve per evitare un divBy0 error
    p=1/(1+np.exp(-f)) #sigmoide della funzione 
    return p


def sig_cost_function_process_multithread(weights, bias, X, Y):
    TOT_CORES= mp.cpu_count()
    chunks=ceil(NUM_DATA/TOT_CORES)

    queue = mp.Queue(maxsize=TOT_CORES)
    R_emp=0
    for i in range(0, NUM_DATA, chunks): 
        p=mp.Process(target=sig_worker,args=(queue,weights,bias,X[i:i + chunks],Y[i:i+chunks]),daemon=True)
        p.start()
    
    for i in range(TOT_CORES):
        R_emp += queue.get()
    return R_emp/NUM_DATA

def sig_worker(queue,weights,bias,X,Y):
    R_emp=0
    for x,y in zip(X,Y):
        emp_distribution = get_sample_emp_distribution(weights,x)
        #probabilità che la label assegnata sia diversa da quella corretta, va minimizzato 
        R_emp+=prob_x_div_corretto(emp_distribution, y, bias)
    queue.put(R_emp)

####################### NON NECESSARI AL THREAD ##################
if __name__ == '__main__':

    def sig_cost_function_single_thread(weights, bias, X, Y):
        R_emp=0
        for x,y in zip(X,Y):
            emp_distribution = get_sample_emp_distribution(weights,x)
            #probabilità che la label assegnata sia diversa da quella corretta, va minimizzato 
            R_emp+=prob_x_div_corretto(emp_distribution, y, bias)
        return R_emp/len(X) #cost function classica, do ad ogni sample lo stesso peso
    
    def cross_entropy_cost(weights, X, Y):
        R_emp=0
        for x,y in zip(X,Y):
            emp_distribution = get_sample_emp_distribution(weights,x)
            R_emp+=cross_entropy_loss(emp_distribution, y)
            
        return R_emp/len(X)

    def cross_entropy_loss(emp_distribution, y):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
            corrected_label= (y+1)/2
            p_1=emp_distribution[1]
            return -(((corrected_label)*np.log(p_1))+((1-corrected_label)*np.log(1-p_1)))
    def predict(X, weights):
        pred=numpy.empty(0,dtype=int)
        emp_distr_list=numpy.empty(0, dtype=dict)
        for x in X:
            emp_distribution = get_sample_emp_distribution(weights,x)
            l=int(max(zip(emp_distribution.values(), emp_distribution.keys()))[1])
            pred=numpy.append(pred,l)
            emp_distr_list=numpy.append(emp_distr_list,emp_distribution)

        return pred, emp_distr_list


    def scores(cost_function, weights, X,Y,feats_val,Y_val, bias=None):
        if bias == None:
            cost=cost_function(weights, X, Y)
        else:
            cost=cost_function(weights, bias, X, Y)
            
        pred_train, emp_train=predict(X,weights)

        accuracy_train=accuracy_score(Y,pred_train)
        p_pos_train=[i[1] for i in emp_train]
        auc_train=roc_auc_score(Y,p_pos_train)


        pred_val, emp_val=predict(feats_val,weights)

        accuracy_val=accuracy_score(Y_val,pred_val)
        
        p_pos_val=[i[1] for i in emp_val]
        #print(emp_val[0],p_pos_val[0], pred_val[0])
        auc_val=roc_auc_score(Y_val,p_pos_val)

        print("accuracy: ",accuracy_train," auc: ",auc_train)

        return [cost,accuracy_train,auc_train,accuracy_val,auc_val]

    def run_optimizer(opt, cost_function, init_param, feats_val,Y_val, num_steps, interval, execs_per_step):
        # Copy the initial parameters to make sure they are never overwritten
        history=[]
        if len(init_param)==3:
            weights, X_param, Y_param = init_param.copy()
            history.append([0]+scores(cost_function,weights,X_param,Y_param,feats_val,Y_val))
        elif len(init_param)==4:
            weights, bias, X_param, Y_param = init_param.copy()
            history.append([0]+scores(cost_function,weights,X_param,Y_param,feats_val,Y_val,bias))
        # Initialize the memory for cost values during the optimization
        # Monitor the initial cost value
        

        print(
            f"\nRunning the {opt.__class__.__name__} optimizer for {num_steps} iterations."
        )

        for step in range(num_steps):
            # Print out the status of the optimization
            if step % 10 == 0:
                print(
                    f"Step {step:3d}: Circuit executions: {history[step][0]:4d}, "
                    f"Cost = {history[step][1]}"
                )
                #print(bias)
                print(
                    f"Step {step:3d}: Circuit executions: {history[step][0]:4d}, "
                    f"accuracy = {history[step][2]}"
                )

            # batch_index = np.random.randint(0, len(X), (BATCH_SIZE,))
            # feats_train_batch = X[batch_index]
            # Y_train_batch = Y[batch_index]
            # Perform an update step
            # weights, _, _ = opt.step(cost_function, weights, feats_train_batch, Y_train_batch)
            if len(init_param)==3:
                inizio =time.time()
                
                weights, _, _ = opt.step(cost_function, weights, X_param, Y_param)
                history.append([(step + 1) * execs_per_step]+scores(cost_function,weights, X_param, Y_param, feats_val, Y_val))
                
                fine=time.time()
                print(fine-inizio)
            elif len(init_param)==4:
                inizio =time.time()
                weights, bias, _, _ = opt.step(cost_function, weights, bias, X_param, Y_param)
                history.append([(step + 1) * execs_per_step]+scores(cost_function, weights,  X_param, Y_param, feats_val, Y_val,bias))
                fine=time.time()
                
                print("tempo impiegato",fine-inizio)

        print(
            f"Step {num_steps:3d}: Circuit executions: {history[-1][0]:4d}, "
            f"Cost = {history[-1][1]}"
        )
        return history, weights


################### main ###################

    print("prima riga pre normalizzazione",X[0],"   ",Y[0])
    features=GL.normalize2pi(X)
    print("prima riga post normalizzazione", features[0],"   ", Y[0])
    if NUM_DATA==len(features): print("corrette dimesioni")

    print(len(features))

    #suddivisione 
    for i in range(30):
        feats_train, feats_val, Y_train, Y_val = train_test_split(
            features, Y, train_size=train_perc
        )
        #dipende dai dati
        #Y = Y * 2 - 1  # shift label from {0, 1} to {-1, 1}

        # print tutti i dati
        # for x,y in zip(X, Y):
        #     print(f"x = {x}, y = {y}")        

        ########## setup pesi ##########

        #come descritto nel paper, limito le rotazioni su Pauli Z e Y,
        #uso una lista anziche una matrice
        # probabilmente i pesi iniziali sono un problema
        #weights_init = np.random.random_sample((num_layers+1)*DIMENSIONS*2,requires_grad=True) # forse non serve 

        weights_init= np.zeros((NUM_LAYERS+1)*DIMENSIONS*2,requires_grad=True)
        #solo per dati binari
        bias_init = np.array(0.0, requires_grad=True)

        # print("Weights:", weights_init)
        # print("Bias: ", bias_init)

        weights = weights_init
        bias = bias_init


        ############# PRINT PER CONTROLLARE IL CIRCUITO #######################
        print(feats_train[0])
        GL.print_circuit(weights, feats_train[0],circuit)

        #input("Press Enter to continue...")

        ######################## ZONA DI OBLIO DOVE TUTTO VIENE MODIFICATO ###########################

        opt= qml.SPSAOptimizer(NUM_STEPS_SPSA,)
        if COST_WITH_BIAS == False:
            history, weights = run_optimizer(
            opt, cross_entropy_cost, [weights,feats_train,Y_train], feats_val,Y_val, NUM_STEPS_SPSA, 20, 1
            )
        elif COST_WITH_BIAS == True:
            history, weights = run_optimizer(
            opt, sig_cost_function_single_thread, [weights,bias,feats_train,Y_train], feats_val,Y_val, NUM_STEPS_SPSA, 20, 1
            )

        
        numpy.savetxt("data"+str(i)+".csv",history, delimiter=",", fmt=['%d','%.11f','%.11f','%.11f','%.11f','%.11f'], header="iter,cost,accuracy_train,auc_train,accuracy_val,auc_val")
        # print(weights)

        # print(weights==weights2)

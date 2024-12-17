
import multiprocessing as mp
import genLib as GL
import numpy 
from pennylane import numpy as np
import pennylane as qml

if __name__ == '__main__':
    TOT_CORES= mp.cpu_count()
    pool = mp.Pool(TOT_CORES)
    
    X,Y = GL.get_iris()
    print(X[0])

    #Nota a bene NUM_DATA non deve essere usato per i dati dopo che sono stati suddivisi in train e test batch
    NUM_DATA = len(Y)

#NOTE: DIMENSIONS deve essere modificato a seconda del database, non è estratto dai dati in quando necessario per i sottoprocessi, e per evitare overhead evitiamo anche 
DIMENSIONS = 4
train_perc=0.75

NUM_LAYERS = 4
BATCH_SIZE = 20
NUM_STEPS_SPSA =300
COST_WITH_BIAS=True

SHOTS=1000
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
import multiprocessing as mp
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, accuracy_score,f1_score,precision_recall_curve,auc
    from math import ceil
    import time
    from functools import partial

################# NECESSARI PER THREAD ###########################################
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

def cross_entropy_loss(emp_distribution, y):
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
            corrected_label= (y+1)/2
            p_1=emp_distribution[1]
            return -(((corrected_label)*np.log(p_1))+((1-corrected_label)*np.log(1-p_1)))

def get_sample_emp_distribution(weights,x):
        #equivalente nella cross entropy, consulto il circuito R volte e alcune volte ritorna -1 altre 1
        samples=variational_classifier(weights,x, [-1,1])
        #\ calcolo la distribuzione di probabilità che sia 1 o -1
        return {k:v/SHOTS for k,v in samples.items()}

def sig_worker(weights,bias,data):
    try:
        R_emp=0
        for x,y in data:
            emp_distribution = get_sample_emp_distribution(weights,x)
            #probabilità che la label assegnata sia diversa da quella corretta, va minimizzato 
            R_emp+=prob_x_div_corretto(emp_distribution, y, bias)
        return R_emp
    except KeyboardInterrupt:
        return
    
def entropy_worker(weights, data):
    try:
        R_emp=0
        for x,y in data:
            emp_distribution = get_sample_emp_distribution(weights,x)
            R_emp+=cross_entropy_loss(emp_distribution, y)
        return R_emp
    except KeyboardInterrupt:
            return

def predict_worker(weights, X):
    try:
        pred=numpy.empty(0,dtype=int)
        emp_distr_list=numpy.empty(0, dtype=dict)
        for x in X:
            emp_distribution = get_sample_emp_distribution(weights,x)
            l=int(max(zip(emp_distribution.values(), emp_distribution.keys()))[1])
            pred=numpy.append(pred,l)
            emp_distr_list=numpy.append(emp_distr_list,emp_distribution)

        return pred, emp_distr_list #forse meglio aggiungere .copy()? nah
    except KeyboardInterrupt:
        return
########################### NON NECESSARI AL THREAD, EVITA OVERHEAD ###########################
if __name__ == '__main__':
    def sig_cost_function_process_multithread(weights, bias, X, Y):
        try:
            chunks=ceil(len(X)/TOT_CORES)
            R_emp=0
            data=[]
            for i in range(0, len(X),chunks):
                data.append(zip(X[i:i + chunks],Y[i:i+chunks]))
            #print(data)
            results= pool.imap(partial(sig_worker,weights,bias), data)
            #count=0
            for result in results:
                #count+=1
                R_emp+=result
            #print("se corretto è uguale al numero di core", count)
            return R_emp/len(X)
        except KeyboardInterrupt:
            pool.terminate()
            raise KeyboardInterrupt
    
    def cross_entropy_process_multithread(weights, X, Y):
        try:
            chunks=ceil(len(X)/TOT_CORES)
            R_emp=0
            data=[]
            for i in range(0, len(X),chunks):
                data.append(zip(X[i:i + chunks],Y[i:i+chunks]))
            #print(data)
            results= pool.imap(partial(entropy_worker,weights), data)
            #count=0
            for result in results:
                #count+=1
                R_emp+=result
            #print("se corretto è uguale al numero di core", count)
            return R_emp/len(X)
        except KeyboardInterrupt:
            pool.terminate()
            raise KeyboardInterrupt

    def predict(X, weights):
        try:
            pred=numpy.empty(0,dtype=int)
            emp_distr_list=numpy.empty(0, dtype=dict)
            chunks=ceil(len(X)/TOT_CORES)
            data=[]
            for i in range(0, len(X),chunks):
                a=iter(X[i:i + chunks])
                data.append(a)
            #pool map mantiene l'ordine dei risultati
            results= pool.map(partial(predict_worker,weights), data)
            for result in results:
                p,e= result
                pred=numpy.append(pred,p)
                emp_distr_list=numpy.append(emp_distr_list,e)
            return pred, emp_distr_list
        except KeyboardInterrupt:
            pool.terminate()
            raise KeyboardInterrupt
    
    def predict_single_thread(X, weights):
        pred=numpy.empty(0,dtype=int)
        emp_distr_list=numpy.empty(0, dtype=dict)
        for x in X:
            emp_distribution = get_sample_emp_distribution(weights,x)
            l=int(max(zip(emp_distribution.values(), emp_distribution.keys()))[1])
            pred=numpy.append(pred,l)
            emp_distr_list=numpy.append(emp_distr_list,emp_distribution)

        return pred, emp_distr_list

    #tengo single thread perchè può tornare utile su architetture in cui ci sono troppi pochi core
    def sig_cost_function_single_thread(weights, bias, X, Y):
        R_emp=0
        for x,y in zip(X,Y):
            emp_distribution = get_sample_emp_distribution(weights,x)
            #probabilità che la label assegnata sia diversa da quella corretta, va minimizzato 
            R_emp+=prob_x_div_corretto(emp_distribution, y, bias)
        return R_emp/len(X) #cost function classica, do ad ogni sample lo stesso peso
    
    def cross_entropy_cost_single_thread(weights, X, Y):
        R_emp=0
        for x,y in zip(X,Y):
            emp_distribution = get_sample_emp_distribution(weights,x)
            R_emp+=cross_entropy_loss(emp_distribution, y)
        return R_emp/len(X)


    def scores(cost_function, weights, X,Y,feats_val,Y_val, bias=None):
        if bias == None:
            cost=cost_function(weights, X, Y)
        else:
            cost=cost_function(weights, bias, X, Y)
            
        pred_train, emp_train=predict(X,weights)

        accuracy_train=accuracy_score(Y,pred_train)
        p_pos_train=[i[1] for i in emp_train]
        auc_train=roc_auc_score(Y,p_pos_train)
        f1_train=f1_score(Y,pred_train)

        precision, recall, thresholds = precision_recall_curve(Y, p_pos_train)
        auc_pr_train = auc(recall, precision)

        pred_val, emp_val=predict(feats_val,weights)

        accuracy_val=accuracy_score(Y_val,pred_val)
        
        p_pos_val=[i[1] for i in emp_val]
        #print(emp_val[0],p_pos_val[0], pred_val[0])
        auc_val=roc_auc_score(Y_val,p_pos_val)
        f1_val=f1_score(Y_val,pred_val)
        precision, recall, thresholds = precision_recall_curve(Y_val, p_pos_val)
        auc_pr_val = auc(recall, precision)

        print("accuracy: ",accuracy_train," auc roc: ",auc_train,"f1: ",f1_train,"auc pr: ",auc_pr_train)

        return [cost,accuracy_train,auc_train,accuracy_val,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val]

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


########################### main ###########################
    
    print("prima riga pre normalizzazione",X[0],"   ",Y[0])
    features=GL.normalize2pi(X)
    print("prima riga post normalizzazione", features[0],"   ", Y[0])
    if NUM_DATA==len(features): print("corrette dimesioni")

    # print tutti i dati
    # for x,y in zip(X, Y):
    #     print(f"x = {x}, y = {y}")        

    print(len(features))

    #suddivisione 
    for i in range(15):
        feats_train, feats_val, Y_train, Y_val = train_test_split(
            features, Y, train_size=train_perc
        )


########################### setup pesi ###########################

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


########################### PRINT PER CONTROLLARE IL CIRCUITO ###########################
        print(feats_train[0])
        GL.print_circuit(weights, feats_train[0],circuit)

        #input("Press Enter to continue...")

########################### ZONA DI OBLIO DOVE TUTTO VIENE MODIFICATO ###########################

        opt= qml.SPSAOptimizer(NUM_STEPS_SPSA,)
        if COST_WITH_BIAS == False:
            history, weights = run_optimizer(
            opt, cross_entropy_process_multithread, [weights,feats_train,Y_train], feats_val,Y_val, NUM_STEPS_SPSA, 20, 1
            )
        elif COST_WITH_BIAS == True:
            history, weights = run_optimizer(
            opt, sig_cost_function_process_multithread, [weights,bias,feats_train,Y_train], feats_val,Y_val, NUM_STEPS_SPSA, 20, 1
            )

        
        numpy.savetxt("iris\data"+str(i)+".csv",history, delimiter=",", fmt=['%d','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f','%.11f'], header="iter,cost,accuracy_train,auc_train,accuracy_val,auc_val,f1_train,f1_val,auc_pr_train,auc_pr_val")
        # print(weights)
###################################################################################################################################################
        # print(weights==weights2)
    pool.close()
    pool.join()

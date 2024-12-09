#tentativo di mappatura, la mappatura da stringa a label può essere molto piu semplice, cio non toglie che magari in futuro può tornare utile
#dic_of_classes={el:0.0 for el in classi}
#mappa=numpy.zeros(pow(2, dimensions))
#for it in range(max_iterations): ## da cambiare in "fino a che non converge l'ottimizzatore"
    # risultati=numpy.empty(pow(2, dimensions), dtype=object)
    # for i in range(len(risultati)):
    #     risultati[i] = deepcopy(dic_of_classes)    
    # for x,y in zip(feats_train,Y_train):
    #     #dovrei contare, in R shots, quale è quello che capita di piu e mapparlo ad una label,
    #     #l'ottenimento del max è gia ottenuto nel metodo circuit
    #     s=circuit(weights, x)
    #     print(s)
    #     print(int(max(s, key=s.get),2))
    #     qubits_results=int(max(s, key=s.get),2)
    #     risultati[qubits_results][y]+=1
    # print(risultati)
    # #ora per trovare il peso ad un determinato numero di certa label devo dare tot=sum(mappa[pos][Yi]/occurrances[Yi]) e poi (mappa[pos][Yi]/occurrances[Yi])/tot
    # for i in range(pow(2,dimensions)):
    #     for cl in classi:
    #         risultati[i][cl]=risultati[i][cl]/occurrences_Y[cl]
    #     mappa[i]=max(risultati[i], key=risultati[i].get)
        
    # print(mappa)
    # input("rpreprepr")
    
    


#metodo vecchio
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


# 1: Input Labeled training samples T = {~x ∈ Ω ⊂ Rn} × {y ∈ C}, Optimization routine,
# 2: Parameters Number of measurement shots R, and initial parameter ~θ0.
# 3: Calibrate the quantum Hardware to generate short depth trial circuits.
# 4: Set initial values of the variational parameters ~θ = ~θ0 for the short-depth circuit W(~θ)
# 5: while Optimization (e.g. SPSA) of Remp(~θ) has not converged do
# 6: for i = 1 to |T| do
# 7: Set the counter ry = 0 for every y ∈ C.
# 8: for shot = 1 to R do
# 9: Use UΦ(~xi) to prepare initial feature-map state | Φ(~xi)ihΦ(~xi)|
# 10: Apply discriminator circuit W(~θ) to the initial feature-map state .
# 11: Apply |C| - outcome measurement {My}y∈C
# 12: Record measurement outcome label y by setting ry → ry + 1
# 13: end for
# 14: Construct empirical distribution ˆpy(~xi) = ryR^−1.
# 15: Evaluate Pr ( ˜m( ~xi) 6= yi|m(~x) = yi) with ˆpy(~xi) and yi
# 16: Add contribution Pr ( ˜m( ~xi) 6= yi|m(~x) = yi) to cost function Remp(~θ).
# 17: end for
# 18: Use optimization routine to propose new ~θ with information from Remp(~θ)
# 19: end while
# 20: return the final parameter ~θ ∗ and value of the cost function Remp(θ ∗ )


# def run_optimizer(opt, cost_function, init_param, num_steps, interval, execs_per_step):
#     # Copy the initial parameters to make sure they are never overwritten
#     weights,X,Y = init_param.copy()

#     # Initialize the memory for cost values during the optimization
#     cost_history = []
#     # Monitor the initial cost value
#     cost_history.append(cost_function(weights,X,Y))
#     exec_history = [0]

#     print(
#         f"\nRunning the {opt.__class__.__name__} optimizer for {num_steps} iterations."
#     )
#     for step in range(num_steps):
#         # Print out the status of the optimization
#         if step % 2 == 0:
#             print(
#                 f"Step {step:3d}: Circuit executions: {exec_history[step]:4d}, "
#                 f"Cost = {cost_history[step]}"
#             )

#         # Perform an update step
#         weights,_,_ = opt.step(cost_function, weights, X,Y)
#         #print(weights)
#         # Monitor the cost value
#         cost_history.append(cost_function(weights,X,Y))
#         exec_history.append((step + 1) * execs_per_step)

#     print(
#         f"Step {num_steps:3d}: Circuit executions: {exec_history[-1]:4d}, "
#         f"Cost = {cost_history[-1]}"
#     )
#     return cost_history, exec_history




from qiskit_algorithms.utils import algorithm_globals
from qiskit_machine_learning.datasets import ad_hoc_data
import plot 
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.algorithms import VQC
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit,QuantumRegister
# algorithm_globals.random_seed = 12345
adhoc_dimension = 5
# train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
#     training_size=20,
#     test_size=5,
#     n=adhoc_dimension,
#     gap=0.3,
#     plot_data=False,
#     one_hot=False,
#     include_sample_total=True,
# )

# plot.plot_dataset(train_features, train_labels, test_features, test_labels, adhoc_total)

#la feature map è la mappa di rotazioni theta applicate ad un determinato livello e altezza dell' unitario M
#feature dimension sono il numero di feature, nel paper con 2 feature sono 5 dim,
#reps è il numero di ripetizioni aka livelli
adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")

sampler = Sampler()

fidelity = ComputeUncompute(sampler=sampler)


adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)

<<<<<<< HEAD


#evaluate sarà la funzione che dato
adhoc_svc = SVC(kernel=adhoc_kernel.evaluate)
=======
#qsvc = QSVC(quantum_kernel=adhoc_kernel)
>>>>>>> ec59e20c60be5913b8a2f907dfd1bb9006cb96b3

#qsvc.fit(train_features, train_labels)

#qsvc_score = qsvc.score(test_features, test_labels)


#print(f"Callable kernel classification test score: {qsvc_score}")

vqc = VQC(num_qubits=adhoc_dimension,)

vqc.circuit.decompose().draw(output="mpl",style="clifford")
plt.show()








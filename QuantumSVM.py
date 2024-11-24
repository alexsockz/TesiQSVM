from qiskit_algorithms.utils import algorithm_globals
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes

from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.algorithms import VQC
import matplotlib.pyplot as plt
from qiskit import transpile
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from qiskit_ibm_runtime import  QiskitRuntimeService, Session, Sampler
from qiskit.primitives import BaseSampler
from qiskit_machine_learning.datasets import ad_hoc_data
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


################################# dataset creation #############################
iris_data = load_iris()
algorithm_globals.random_seed = 12345

features = iris_data.data
labels = iris_data.target

dimensions= features.shape[1]

dimensions=3

train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, train_size=0.8, random_state=algorithm_globals.random_seed
)
# features generate
# train_features, train_labels, test_features, test_labels, adhoc_total = ad_hoc_data(
#     training_size=20,
#     test_size=5,
#     n=dimensions,
#     gap=0.3,
#     plot_data=False,
#     one_hot=False,
#     include_sample_total=True,
# )

########################## quantum classifier creator #####################################
ent=["linear","reverse_linear","full","pairwise","circular","sca"] 
repetitions=4

#key per runnare
#https://quantum.ibm.com/
#6b09d2a8268c75b78b5ec4031c7381f880fc20e6c2f103e029c3a8b7e76c953ca0332015e7462b28f6bcb5d9737465ef19a76d7d3cd60af9c349aa45ace4e6f8

#run online
# service = QiskitRuntimeService(
#     channel='ibm_quantum', 
#     token='6b09d2a8268c75b78b5ec4031c7381f880fc20e6c2f103e029c3a8b7e76c953ca0332015e7462b28f6bcb5d9737465ef19a76d7d3cd60af9c349aa45ace4e6f8'
#     )
# backend = service.least_busy(
#     operational=True, simulator=False, min_num_qubits=dimensions
# )
# pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
# sampler=Sampler(mode=backend)

# #run offline
#aerSimulator prende il posto di qiskitRuntimeService
#senza noise dato che non do nulla in input, specificando una qpu allorà simulerà il suo noise
aer_sim= AerSimulator(method='statevector')# , device='GPU' su fisso
pm = generate_preset_pass_manager(backend=aer_sim, optimization_level=1)
sampler = Sampler(mode=aer_sim)

#richiede reimplementazione di ComputeUncompute


#la feature map è la mappa di rotazioni theta applicate ad un determinato livello e altezza dell' unitario M
#feature dimension sono il numero di feature, nel paper con 2 feature sono 5 dim,
#reps è il numero di ripetizioni aka livelli
adhoc_feature_map = ZZFeatureMap(feature_dimension=dimensions, reps=2, entanglement="full", insert_barriers=True)
trans_feature_map = pm.run(adhoc_feature_map)
adhoc_ansatz = RealAmplitudes(num_qubits=dimensions, entanglement="full", reps=repetitions, insert_barriers=True)
trans_ansatz = pm.run(adhoc_ansatz)

print(adhoc_feature_map.decompose().draw())
print(adhoc_ansatz.decompose().draw())


vqc = VQC(feature_map=trans_feature_map,ansatz=trans_ansatz,sampler=sampler)
print("training")
vqc.fit(train_features, train_labels)

print("training completato")

print(vqc.score(test_features, test_labels))

circ=vqc.circuit

#forse non necessario
# circ.measure_all()
# simulator = AerSimulator()
# transp = transpile(vqc.circuit, simulator)

# RUN
# result = simulator.run(circ).result()
# counts = result.get_counts(circ)
# print(counts)








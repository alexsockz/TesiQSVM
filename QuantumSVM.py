from qiskit_algorithms.utils import algorithm_globals
import plot 
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import Sampler, Estimator
from qiskit_algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import VQC
import matplotlib.pyplot as plt
from qiskit_ibm_runtime import Session, SamplerV2 as Sampler, EstimatorV2 as Estimator

algorithm_globals.random_seed = 12345
adhoc_dimension = 5
adhoc_reps=4
ent=["linear","reverse_linear","full","pairwise","circular","sca"] 
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
adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement=ent, insert_barriers=True)

backend = service.least_busy(operational=True, simulator=False)
session = Session(backend=backend)
sampler = Sampler(mode=session
)
estimator = Estimator()

fidelity = ComputeUncompute(sampler=sampler)

adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)
adhoc_ansatz = RealAmplitudes(num_qubits=adhoc_dimension, entanglement=ent, reps=adhoc_reps, insert_barriers=True)

vqc = VQC(num_qubits=adhoc_dimension,feature_map=adhoc_feature_map, ansatz=adhoc_ansatz)

vqc.circuit.decompose().draw(output="mpl",style="clifford")

plt.show()








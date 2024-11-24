from qiskit import QuantumCircuit
import numpy as np
from qiskit import Aer

#Changing the simulator 
backend = Aer.get_backend('unitary_simulator')
n=4

circ = QuantumCircuit(n)

for wire in range(n):
    circ.h(wire)
    s=0
    for ctrl in range(wire+1,n):
        s=s+1
        circ.cp((np.pi/(2**s)),ctrl,wire)
    circ.barrier()

for wire in range(int(n/2)):
    circ.swap(wire, n-wire)

print(circ.draw())

#job execution and getting the result as an object
job = execute(circ, backend)
result = job.result()

#get the unitary matrix from the result object
print(result.get_unitary(circ, decimals=3))
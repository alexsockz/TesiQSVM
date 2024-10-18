import qiskit as qk

# Creating Qubits
q = qk.QuantumRegister(2)
# Creating Classical Bits
c = qk.ClassicalRegister(2)
circuit = qk.QuantumCircuit(q, c)
print(circuit)
# Initialize empty circuit
circuit = qk.QuantumCircuit(q, c)
# # Hadamard Gate on the first Qubit
# circuit.h(q[0])
# # CNOT Gate on the first and second Qubits
# circuit.cx(q[0], q[1])
# # Measuring the Qubits
# circuit.measure(q, c)



print (circuit)
# Using Qiskit Aer's Qasm Simulator: Define where do you want to run the simulation.
simulator = qk.BasicAer.get_backend('qasm_simulator')

# Simulating the circuit using the simulator to get the result
job = qk.execute(circuit, simulator, shots=100)
result = job.result()

# Getting the aggregated binary outcomes of the circuit.
counts = result.get_counts(circuit)
print (counts)
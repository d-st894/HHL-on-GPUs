#imports

import matplotlib.pyplot as plt
%matplotlib inline
#plt.rcParams["figure.figsize"]=20,10
import numpy as np
import math
import qiskit
from qiskit import IBMQ, Aer, transpile, assemble,execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram
from qiskit.extensions import UnitaryGate
from qiskit.tools.monitor import job_monitor
from scipy import linalg
from scipy.stats import ortho_group

# rotations and binary-decimal conversions

def R(i) :
    if i==0 : return ([[1,0],[0,1]])
    theta=np.arcsin(1/i)
    return([np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)])

def dec_to_bin(d) :
    b=bin(d)[2:]
    while len(b)<nclock :
        b='0'+b
    return b

# qft

def qft_rotations(circuit,n) :
    if n==0 :
        return circuit
    n-=1
    circuit.h(n)
    for qubit in range(n) :
        circuit.cp(-np.pi/2**(n-qubit),qubit,n)
    qft_rotations(circuit,n)

def swap_registers(circuit,n) :
    for qubit in range(n//2):
        circuit.swap(qubit,n-qubit-1)
    return circuit

def qft_dagger(qc,n) :
    qft_rotations(qc,n)
    swap_registers(qc,n)
    return qc

# qpe and iqpe.

def qpe(clocks,states,gate) :
    clock=QuantumRegister(clocks)
    state=QuantumRegister(states)
    dummy=QuantumRegister(1)
    cdummy=ClassicalRegister(states)
    qpe=QuantumCircuit(clock,state,dummy,cdummy)
    qpe.h(clock)
    repetitions=1
    for a in range(clocks):
        for i in range(repetitions):
            target=[clock[a]]
            for i in range(states) :
                target.append(state[i])
            qpe.append(gate,target)
        repetitions*=2
    qpe.barrier()
    qft_dagger(qpe,clocks)
    qpe.barrier()
    return qpe

def iqpe(clocks,states,gate) :
    temp=qpe(clocks,states,gate)
    iqpe=temp.inverse()
    iqpe.barrier()
    return iqpe

# hhl in all it's glory.

def hhl(circ,ancilla,clock,b,anc_measure,b_measure,Ugate) :
    temp1=qpe(clock.size,b.size,Ugate)
    circ.compose(temp1,inplace=True)
    
    target=[clock[i] for i in range(nclock)]
    target.append(ancilla)
    for i in range(2**nclock) :
        circ.append(UnitaryGate(R(i),label=dec_to_bin(i)).control(nclock,ctrl_state=dec_to_bin(i)),target)
    circ.barrier()
    
    circ.measure(ancilla,anc_measure)
    
    temp2=iqpe(clock.size,b.size,Ugate)
    circ.compose(temp2,inplace=True)
    
    circ.measure(b,b_measure)

# result

def simulate(circuit) :
    
    aer_sim=Aer.get_backend('aer_simulator')
    #aer_sim.set_options(method='statevector',device='GPU')
    ##aer_sim.set_options(cusvaer_enable=false)

    ##simulator_gpu = Aer.get_backend('aer_simulator')
    ###sim = AerSimulator(method='statevector', device='GPU')
    ###simulator_gpu.set_options(device='GPU')
    
    shots=10**3
    t_circuit=transpile(circuit,aer_sim)
    qobj=assemble(t_circuit,shots=shots)
    
    avg_answer={}
    ratios=np.zeros((50,8))
    for i in range(50) :
        results=aer_sim.run(qobj).result()
        answer=results.get_counts()
        for key in answer :
            if key[-1]=='1' :
                if key in avg_answer :
                    avg_answer[key]=(avg_answer[key]*i+answer[key])/(i+1)
                else :
                    avg_answer[key]=answer[key]
                if key[-1]=='1' :
                    j=int(key[:-1],2)
                    ratios[i][j]=answer[key]
    for i in range(len(ratios)) :
        ratios[i]=[ratios[i][j]/ratios[i][0] for j in range(len(ratios[i]))]
    #print(avg_answer['1 1']/avg_answer['0 1'])
    return [avg_answer,ratios]

# params and circuit diagram

#A=np.array([[1,-1/2],[-1/2,1]])
N=8
maximum=8.0
eigs=np.random.uniform(low=0.0,high=maximum,size=N)
eigs=np.ceil(eigs)
print(eigs)
m=ortho_group.rvs(dim=len(eigs))
A=np.diag(eigs)
A=np.matmul(np.matmul(np.transpose(m),A),m)

#bmat=np.array([1,0])
b_mat=np.random.rand(8)
bmat=np.array(b_mat/np.linalg.norm(b_mat))

nclock=6

#t=np.pi
t=2*np.pi/((2**nclock)*min(eigs))

exact=np.linalg.solve(A,bmat)
exactratios=[(exact[i]/exact[0])**2 for i in range(len(exact))]

circuit=build_circuit(A,bmat,t,nclock)
#circuit.draw('mpl',filename='exact-encoding-circuit1.png',)

# multiple runs

[answer,ratios]=simulate(circuit)

#[fig,ax]=plt.subplots(1,2)
#for i in range(2) :
#    ax[i].hist(ratios[:,i])
#plt.show()

#[fig,ax]=plt.subplots(1,1)
#ax.hist(ratios[:,1])
#ax.grid()
#ax.set_title('')
#ax.set_xlabel(r'Prob(|11>)/Prob(|01>)',fontsize=15)
#ax.set_ylabel(r'Frequency',fontsize=15)
#ax.tick_params(axis='both', which='major',labelsize=10)
#plt.savefig('exact-encoding-ratio.png')
#plt.show()

totalanswer=sum(list(answer.values()))
for key in answer :
    answer[key]=answer[key]/totalanswer
#[fig,ax]=plt.subplots(1,1)
#bars=ax.bar(range(len(answer)),list(answer.values()), tick_label=list(answer.keys()),color='cadetblue')
#ax.grid(axis='y')
#ax.set_title('')
#ax.set_ylabel(r'Probabilities',fontsize=15)
#ax.set_xlabel(r'States',fontsize=15)
#ax.tick_params(axis='both', which='major',labelsize=15)
#ax.bar_label(bars,fontsize=12)
#plt.savefig('exact-encoding-probs.png')
#plt.show()

avg_ratios=[]
std_ratios=[]
for i in range(len(ratios[0])) :
    avg_ratios.append(sum(ratios[:,i])/len(ratios[:,i]))
    std_ratios.append(np.sqrt(sum([(ratios[j,i]-avg_ratios[-1])**2 for j in range(len(ratios[:,i]))])/len(ratios[:,i])))
print(exactratios)
print(avg_ratios)
print(std_ratios)
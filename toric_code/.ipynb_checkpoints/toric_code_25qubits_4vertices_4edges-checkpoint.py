#!/usr/bin/env python
# coding: utf-8

# In[1]:


#imports

#import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#plt.rcParams["figure.figsize"]=20,10
import numpy as np
import math
import qiskit
from qiskit import IBMQ, Aer, transpile, assemble,execute
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.visualization import plot_histogram
from qiskit.extensions import UnitaryGate
from qiskit.circuit.library import RYGate,UGate
from qiskit.tools.monitor import job_monitor
from scipy import linalg
from scipy.stats import ortho_group
from time import perf_counter
from functools import reduce
import scipy.sparse as sps
import scipy.sparse.linalg as spslin

gpu=False
wilson_loop=False

# In[2]:


# rotations and binary-decimal conversions

def R(i) :
    if i==0 : return ([[1,0],[0,1]])
    theta=np.arcsin(1/i)
    return([np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)])

def get_U_gate(i, label) :
    if i==0 : return UGate(0,0,0,label=label)
    theta=np.arcsin(1/i)
    return UGate(2*theta,0,0,label=label)

def dec_to_bin(d) :
    b=bin(d)[2:]
    while len(b)<nclock :
        b='0'+b
    return b


# In[3]:


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


# In[4]:


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


# In[5]:


# hhl in all it's glory.

def hhl(circ,ancilla,clock,b,anc_measure,b_measure,Ugate) :
    temp1=qpe(clock.size,b.size,Ugate)
    circ.compose(temp1,inplace=True)
    
    target=[clock[i] for i in range(nclock)]
    target.append(ancilla)
    for i in range(2**nclock) :
        gate = get_U_gate(i, dec_to_bin(i)).control(nclock,ctrl_state=dec_to_bin(i))
        circ.append(gate, target)
    circ.barrier()
    
#    circ.measure(ancilla,anc_measure)
    
    temp2=iqpe(clock.size,b.size,Ugate)
    circ.compose(temp2,inplace=True)
    
    circ.measure(ancilla,anc_measure)
    circ.measure(b,b_measure)


# In[6]:


#parameters and circuit

def build_circuit(A,bmat,t,nclock) :
    N=len(bmat)
    nb=(int)(np.log2(N))
        
    U=linalg.expm(t*A*1j)
    Ugate=UnitaryGate(U,label="e^iAt").control()

    clock=QuantumRegister(nclock,name='clock')
    b=QuantumRegister(nb,name='b')
    ancilla=QuantumRegister(1,name='ancilla')
    a_measure=ClassicalRegister(1,name='ancilla measurement')
    b_measure=ClassicalRegister(nb,name='b measurement')
    circuit=QuantumCircuit(clock,b,ancilla,a_measure,b_measure)
    
    circuit.initialize(bmat/np.linalg.norm(bmat),b)
    circuit.barrier()
    hhl(circuit,ancilla,clock,b,a_measure,b_measure,Ugate)
    
    return circuit


# In[7]:


# result

def simulate(size,tmp_circuit) :
    
    aer_sim=Aer.get_backend('aer_simulator', method='statevector')
    if gpu :
        aer_sim.set_options(method='statevector',device='GPU')
    ##aer_sim.set_options(cusvaer_enable=false)

    ##simulator_gpu = Aer.get_backend('aer_simulator')
    ###sim = AerSimulator(method='statevector', device='GPU')
    ###simulator_gpu.set_options(device='GPU')
    print(aer_sim._basis_gates())
    
    shots=5*10**3
    start = perf_counter()
    circuit=QuantumCircuit(tmp_circuit.num_qubits,tmp_circuit.num_clbits)
    circuit.compose(tmp_circuit,inplace=True)
    stop = perf_counter()
    print(f'copy labelless: {stop-start}s')

    start = perf_counter()
    t_circuit=transpile(circuit,aer_sim)
    print(f'{t_circuit.size()} gates.')
    stop = perf_counter()
    print(f'transpile: {stop-start}s')
#    qobj=assemble(t_circuit,shots=shots)
    
    avg_answer={}
    ratios=np.zeros((50,size))
    results=aer_sim.run(t_circuit,shots=shots).result()
    for i in range(50) :
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
        if ratios[i][0]!=0 : ratios[i]=[ratios[i][j]/ratios[i][0] for j in range(len(ratios[i]))]
    #print(avg_answer['1 1']/avg_answer['0 1'])
    return [avg_answer,ratios]


# In[ ]:


# params and circuit diagram

np.random.seed(42)
#np.random.seed(66)
#np.random.seed(18)

# Constructing A as the 2 x 2 toric code Hamiltonian
Lx,Ly=2,2
N_toric=2*Lx*Ly
# pauli matrices
Id=sps.eye(2)
Sx=sps.csr_matrix(np.array([[0,1.],[1.,0]]))
Sy=sps.csr_matrix(np.array([[0,-1.j],[1.j,0]]))
Sz=sps.csr_matrix(np.array([[1.,0.],[0,-1.]]))
s=np.arange(N_toric)
# adjaceny lists
T1=np.array([3,2,4,5,7,6,0,1])
T2=np.array([2,3,5,4,6,7,1,0])
I1=np.array([6,7,1,0,2,3,5,4])
I2=np.array([7,6,0,1,3,2,4,5])
Tx=np.array([1,0,3,2,5,4,7,6])
# operation list
J_tc=1
As=[[-J_tc,i,Tx[i],T1[i],I2[i]] for i in [0,1,4,5]]
Bp=[[-J_tc,i,Tx[i],T1[i],I2[i]] for i in [2,3,6,7]]
operations=[["xxxx",As],["zzzz",Bp]]
# with a loop
J_tcw=1
gz_w=0.1
As_w=[[-J_tcw,i,Tx[i],T1[i],I2[i]] for i in [0,1,4,5]]
Bp_w=[[-J_tcw,i,Tx[i],T1[i],I2[i]] for i in [2,3,6,7]]
Wx=[[-gz_w,0,1]]
Wy=[[-gz_w,2,6]]
operations_w=[["xxxx",As_w],["zzzz",Bp_w],["zz",Wx],["zz",Wy]]
def toric_hamiltonian(L,ops):
    ham=np.zeros(shape=(2**L,2**L),dtype=float)
    for item in ops:
        string=item[0]
        for inside in item[1:len(item)]:
            for j in range(len(inside)):
                op_list=[Id]*L
                for k in range(len(string)):
                    if string[k]=='x':
                        op_list[inside[j][k+1]]=Sx
                    elif string[k]=='y':
                        op_list[inside[j][k+1]]=Sy
                    elif string[k]=='z':
                        op_list[inside[j][k+1]]=Sz
                #print(reduce(sps.kron,op_list).toarray())
                ham+=inside[j][0]*reduce(sps.kron,op_list).toarray()
    return ham
if wilson_loop :
    A=toric_hamiltonian(N_toric,operations_w)
else :
    A=toric_hamiltonian(N_toric,operations)
N=len(A) # 256, i.e., 2**8

#A=np.array([[1,-1/2],[-1/2,1]])
#N=32
#maximum=32.0
#eigs=[1.0,2.0]#,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.0,13.0,14.0,15.0,16.0]
#eigs=np.random.uniform(low=0.0,high=maximum,size=N)
#eigs=np.ceil(eigs)
#print(eigs)
#m=ortho_group.rvs(dim=len(eigs))
#A=np.diag(eigs)
#A=np.matmul(np.matmul(np.transpose(m),A),m)
#print(A)

#b_mat=np.array([1,0])#,0,0,1,0,1,1,0,0,1,1,0,0,1,1])
b_mat=np.random.rand(N)
bmat=np.array(b_mat/np.linalg.norm(b_mat))

nclock=16

#t=np.pi
t=2*np.pi*((2**nclock)-1)/((2**nclock)*10.0)

exact=np.linalg.solve(A,bmat)
exactratios=[(exact[i]/exact[0])**2 for i in range(len(exact))]
print(exactratios)

circuit=build_circuit(A,bmat,t,nclock)
#circuit.draw('mpl',filename='exact-encoding-circuit1.png',)


# In[ ]:


# multiple runs

[answer,ratios]=simulate(N,circuit)

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

print(avg_ratios)
print(std_ratios)

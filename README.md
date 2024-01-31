# HHL on GPUs using cuquantum

## Folders :

### old

Contains files from before the Nvidia team was able to make everything properly functional and compatible.

### poster

Contains poster ppt and keynote. I will convert to and add pdf when finalised.

### ahehn_mods

Contains the fixed file supplied by Nvidia and modifications of it trying out more qubits, more compicated matrices, etc. File name shows size of problem number of qubits used. For eg *_10q_8m.py means 10 qubits solving a 10 X 10 problem (regardless of rest of filename).

###  toric_code

Files with a 2 cross 2 toric code hamiltonian fed into HHL. Based on the Nvidia fixed file. Matrix size is always 256 X 256 (4 vertices + 4 edges = 8 qubits = 2^8 matrix size). Different files for different number of clock qubits. There is a flag within the code to run it with or without a wilson loop.

### 22jan

Codes for stuff put in the final poster.

## Summary :

- We have a regular qiksit implementation of HHL.
- We have a cuQuantum implementation of HHL that requires modifying the code in ways. I don't yet understand why these particular modificationsa were needed, but I want to.
- We have compared run times of these two in the 22jan folders for different types of matrices :
    * Diagonal matrices whose elements are binary fractions.
    * Diagonal matrices whose elements are irrational (randomly generated).
    * Randomly generated matrices.
- There are still some issues that need to be fixed :
    * The IQPE needs to be more explicit.
    * mpirun sometimes creates independent copies of the run on different devices.

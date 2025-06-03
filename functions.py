import numpy as np
import qutip as qt
################################## FUNCTIONS ##################################
def rhobuild(eigenenergies, eigenvectors, temp):
    """
    Build the density matrix for a given set of eigenenergies and eigenvectors at a specific temperature.

    Parameters:
    - eigenenergies (numpy.ndarray): Array of eigenenergies.
    - eigenvectors (numpy.ndarray): Array of eigenvectors.
    - temp (float): Temperature.

    Returns:
    - numpy.ndarray: Density matrix.
    """

    rhoT=np.zeros([int(N/2)+1, int(N/2)+1],dtype="complex")
    minim=np.min(eigenenergies)
    eigenenergies=eigenenergies-minim
    length=len(eigenenergies)
    # for y in range(length):
    #     newlength = y
    #     if (eigenenergies[y]/temp > 1000):
    #         break
    # length = newlength
    length = 30
    for y in range(length):
        pk=0
        for x in range(length):
            pk += np.exp(-1/temp*(eigenenergies[x]-eigenenergies[y]))
        pk=pk**-1
        rhoT=rhoT+(eigenvectors[y]*eigenvectors[y].conj().trans() ).data*pk
    return rhoT

def mean_value_rho(rho, operator):
    """
    Calculate the mean value of an operator with respect to a density matrix.

    Parameters:
    - rho (numpy.ndarray): Density matrix.
    - operator (numpy.ndarray): Operator.

    Returns:
    - float: Mean value of the operator.
    """

    return np.trace(rho*operator)

def covariance_matrix_pure(rho,Jx_2, Qyz_2, Jy_2, Qzx_2, Dxy_2, Qxy_2, Y_2, Jz_2, op, op_conj):
    """
    Calculate the covariance matrix for a pure state with given symmetry.

    Parameters:
    - rho (numpy.ndarray): Density matrix.
    - Jx_2, Qyz_2, Jy_2, Qzx_2, Dxy_2, Qxy_2, Y_2, Jz_2 (numpy.ndarray): Operators.
    - op (numpy.ndarray): Operator.
    - op_conj (numpy.ndarray): Conjugate of the operator.

    Returns:
    - numpy.ndarray: Covariance matrix.
    """ 

    COV = np.zeros([8,8])
    COV[0,0] = mean_value_rho(rho, Jx_2)
    COV[1,1] = mean_value_rho(rho, Qyz_2)
    COV[2,2] = COV[0,0]
    COV[3,3] = COV[1,1]
    COV[4,4] = mean_value_rho(rho, Dxy_2)
    COV[5,5] = COV[4,4]
    COV[6,6] = mean_value_rho(rho, Y_2) - mean_value_rho(rho, Y)**2
    COV[0,1] = COV[1,0] = mean_value_rho(rho, -1j*(op-op_conj) )
    COV[2,3] = COV[3,2] = -COV[1,0]
    return COV

def fidelity(rho_state, O, temp):
    """
    Calculate fidelity for a given state, operator, and temperature.

    Parameters:
    - rho_state (numpy.ndarray): Density matrix.
    - O (numpy.ndarray): Operator.
    - temp (float): Temperature.

    Returns:
    - float: Fidelity.
    """

    return (((O*O*rho_state).tr() - ((O*rho_state).tr())**2)/temp**4)

def full_covariance_matrix(rho_state, Jx, Qzx, Dxy, Jz, Y, Qxy, Jy, Qyz):
    """
    Calculate the full covariance matrix for a given density matrix and operators.

    Parameters:
    - rho_state (numpy.ndarray): Density matrix.
    - Jx, Qzx, Dxy, Jz, Y, Qxy, Jy, Qyz (numpy.ndarray): Operators.

    Returns:
    - numpy.ndarray: Covariance matrix.
    """
    
    energy, vectors = np.linalg.eig(rho_state)
    Lambda = [Jx, Qyz, Jy, Qzx, Dxy, Qxy, Y, Jz]
    COV = np.zeros([8,8],dtype='complex')
    maximum_no = len(energy)
    for p in range(8):
        for o in range(p, 8):
            suma = 0
            for k in range(maximum_no):
                for n in range(maximum_no):
                    if(abs(energy[k] + energy[n])> 10**-8):
                        suma+= 1/2*(energy[k]-energy[n])**2/(energy[k]+energy[n])*np.real(np.conjugate(np.transpose(vectors[:,k]))@Lambda[p]@vectors[:,n]*np.transpose(np.conjugate(vectors[:,n]))@Lambda[o]@vectors[:,k])
            COV[p,o]=suma
            COV[o,p]=COV[p,o]
    return COV

def covariance_matrix(rho_state, Jx_2, Qyz_2, Jy_2, Qzx_2, Dxy_2, Qxy_2, Y_2, Jz_2, op, op_conj):
    """
    Calculate the covariance matrix for a given density matrix and operators 
    for basis in subspace M=0.

    Parameters:
    - rho_state (numpy.ndarray): Density matrix.
    - Jx_2, Qyz_2, Jy_2, Qzx_2, Dxy_2, Qxy_2, Y_2, Jz_2 (numpy.ndarray): Operators.
    - op (numpy.ndarray): Operator.
    - op_conj (numpy.ndarray): Conjugate of the operator.

    Returns:
    - numpy.ndarray: Covariance matrix.
    """

    energy2, vectors2 = np.linalg.eig(rho_state)
    Lambda = [Jx_2.data, Qyz_2.data, Jy_2.data, Qzx_2.data, Dxy_2.data, Qxy_2.data, Y_2.data, Jz_2.data]
    COV = np.zeros([8,8],dtype='complex')
    maximum_no = len(energy2)
    
    for k in range(maximum_no):
        for l in range(0,6):
           COV[l,l]+= energy2[k]*np.conjugate(np.transpose(vectors2[:,k]))@Lambda[l]@vectors2[:,k]
        COV[0,1] += energy2[k]*np.conjugate(np.transpose(vectors2[:,k]))@(1j*(op-op_conj))@vectors2[:,k]
        COV[2,3] -= COV[0,1]
        COV[6,6] += energy2[k]*(np.conjugate(np.transpose(vectors2[:,k]))@Lambda[6]@vectors2[:,k] - (np.conjugate(np.transpose(vectors2[:,k]))@Y.data@vectors2[:,k])**2)
        
    COV[1,0] = COV[0,1]
    COV[2,3] = COV[3,2]
    return COV


def genOrt3(vector):
    """
    Generate an orthogonal set of three vectors based on the input vector.

    Parameters:
    - vector (numpy.ndarray): Input vector.

    Returns:
    - list of numpy.ndarray: List containing three orthogonal vectors.
    """

    vector = vector/np.sqrt(np.dot(vector,vector))
   
    x = np.cross(vector, [1.,1.,np.exp(1)])
    if np.sum(np.abs(x)**2)<1e-6:
        x = np.cross(vector, [1.,-1.,np.exp(1)])    
    x = x/np.sqrt(np.dot(x,x))    
    y = np.cross(vector, x)
    y = y/np.sqrt(np.dot(y,y))
   
    return [x,y]

def expectationsS(N, state, dir1, dir2, dir3):
    """
    Calculate the expectations of spin operators for a given state.

    Parameters:
    - N (int): Size of the system.
    - state (numpy.ndarray): Quantum state.
    - dir1, dir2, dir3 (numpy.ndarray): Spin operators.

    Returns:
    - dict: Dictionary containing expectation values.
    """

    return {'Sx': qt.expect(dir1, state),
            'Sy': qt.expect(dir2, state),
            'Sz': qt.expect(dir3, state),
            'Sx2': qt.expect(dir1*dir1, state),
            'Sy2': qt.expect(dir2*dir2, state),
            'Sz2': qt.expect(dir3*dir3, state),
    }

def squeezingParSq(optimizationAngle,state,N,expectVal):
    """
    Calculate the squeezing parameter for a given optimization angle, state, system size, and expectation values.

    Parameters:
    - optimizationAngle (float): Optimization angle.
    - state (numpy.ndarray): Quantum state.
    - N (int): Size of the system.
    - expectVal (dict): Dictionary containing expectation values.

    Returns:
    - float: Squeezing parameter.
    """

    orthVecs = genOrt3([expectVal['Sx'],expectVal['Sy'],expectVal['Sz']])  
    orthVec = np.sin(optimizationAngle)*orthVecs[0] + np.cos(optimizationAngle)*orthVecs[1]

    StotalSq = expectVal['Sx']**2+expectVal['Sy']**2+expectVal['Sz']**2
   
    dev1 = orthVec[0]*Jx+\
           orthVec[1]*Jy+\
           orthVec[2]*Jz
    dev1 = dev1.data.toarray()
    if state.type == "ket":
        state = (state.data.toarray())[:,0]
        
        varianceOrth = np.dot(np.conj(state),np.matmul(dev1,dev1).dot(state)) - np.dot(np.conj(state),dev1.dot(state))**2
    else:
        varianceOrth = np.trace(np.dot(state,np.matmul(dev1,dev1))) - np.trace(np.dot(state,dev1))**2
        
    squeezeSq = np.real(N*varianceOrth/StotalSq)
    # squeezeSq = np.real(N*varianceOrth)
    return squeezeSq

def minimalSqueezingEvo(N, H, times, initState, dir1, dir2, dir3):
    """
    Perform minimal squeezing evolution for a given system size, Hamiltonian, time array, initial state, and spin operators.

    Parameters:
    - N (int): Size of the system.
    - H (numpy.ndarray): Hamiltonian.
    - times (numpy.ndarray): Array of time values.
    - initState (numpy.ndarray): Initial quantum state.
    - dir1, dir2, dir3 (numpy.ndarray): Spin operators.

    Returns:
    - dict: Dictionary containing evolution information.
    """
    
    result = qt.mesolve(H, initState, times, e_ops=[])
    states = result.states
    
    diagonal_elements = []
    minSqueezeParams = []
    expected_V = {'Sx': [], 'Sy': [], 'Sz': [], 'Sx2': [], 'Sy2': [], 'Sz2': []}
    variances = {'Sx': [], 'Sy': [], 'Sz': []}
    averages = np.zeros([N+1, len(times)], dtype='complex')
    st_no=0
    for state in states:
        expectValuesS = expectationsS(N, state, dir1, dir2, dir3)
            
        if state.type == 'oper':
            diagonal_elements.append(np.diagonal(state))
        else:
            diagonal_elements.append(np.diagonal(qt.ket2dm(state)))

        for key, value in expected_V.items():
            value.append(expectValuesS[key])
        variances['Sx'].append(expectValuesS['Sx2'] - expectValuesS['Sx']**2)
        variances['Sy'].append(expectValuesS['Sy2'] - expectValuesS['Sy']**2)
        variances['Sz'].append(expectValuesS['Sz2'] - expectValuesS['Sz']**2)
        # res = minimize(squeezingParSq, np.pi/2, args=(state, N, expectValuesS), method='L-BFGS-B',bounds=[(0,2*np.pi)])
        
        # minSqueezeParams.append(res.fun)
        st_no+=1
   
       
    return {'states': states, 
            'squeezing': minSqueezeParams,
            'expected_values': expected_V,
            'variances': variances,
            'diagonal': diagonal_elements,
            'average_pop': averages}

def minimalSqueezing(N, H, dir1, dir2, dir3):
    """
    Perform minimal squeezing for a given system size, Hamiltonian, and spin operators.

    Parameters:
    - N (int): Size of the system.
    - H (numpy.ndarray): Hamiltonian.
    - dir1, dir2, dir3 (numpy.ndarray): Spin operators.

    Returns:
    - dict: Dictionary containing squeezing information.
    """

    states = H.eigenstates()[1]
    
    
    minSqueezeParams = []
    expected_V = {'Sx': [], 'Sy': [], 'Sz': [], 'Sx2': [], 'Sy2': [], 'Sz2': []}
    variances = {'Sx': [], 'Sy': [], 'Sz': []}
    averages = np.zeros([N+1], dtype='complex')
    st_no=0
    for state in states:
        expectValuesS = expectationsS(N, state, dir1, dir2, dir3)

        for key, value in expected_V.items():
            value.append(expectValuesS[key])
        variances['Sx'].append(expectValuesS['Sx2'] - expectValuesS['Sx']**2)
        variances['Sy'].append(expectValuesS['Sy2'] - expectValuesS['Sy']**2)
        variances['Sz'].append(expectValuesS['Sz2'] - expectValuesS['Sz']**2)
        #res = minimize(squeezingParSq, np.pi/2, args=(state, N, expectValuesS), method='L-BFGS-B',bounds=[(0,2*np.pi)])
        
        #minSqueezeParams.append(res.fun)
        st_no+=1
   
       
    return {'states': states, 
            'squeezing': minSqueezeParams,
            'expected_values': expected_V,
            'variances': variances,
            'average_pop': averages}

import re

def atoi(text):
    """
    Convert a string to an integer.

    Parameters:
    - text (str): Input string.

    Returns:
    - int or str: Converted integer or original string.
    """

    return int(text) if text.isdigit() else text

def natural_keys(text):
    """
    Sort a list of strings in a human-friendly order.

    Parameters:
    - text (str): Input string.

    Returns:
    - list: List of sorted strings.
    """
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

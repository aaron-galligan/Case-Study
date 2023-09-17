
import numpy as np

#################### Functions ######################


def MyJacobian(fnc, x0, h):
    '''
    Creates a jacobian matrix of the function defined in fnc()
    Input
    ----------
    x0 :     array, shape (n)
            is the array of values that we're finding the jacobian at
    h :     The h that is used to numerically find the derivative 
    Returns
    -------
    df :   array, shape (n, N)
            Jacobian matrix.
           
    '''

 
    df = np.empty([x0.size , fnc(x0).size])
    df[:] = np.nan

    for j in range(x0.size):
        
        j_equal_1 = np.zeros(x0.size)
        j_equal_1[j] = j_equal_1[j] + 1
            
        df[j] = (fnc(x0 + j_equal_1*h) - fnc(x0 - j_equal_1*h))/(2*h)
    

    return df.T

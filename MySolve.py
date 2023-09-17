import numpy as np

def MySolve( f, x0, df, tol, maxit) :
    '''
    MySolve solves a function in the form f(x) = 0 using the Newton iteration.
    Note: For 1D functions, the x0 passed in must be a numpy array of length 1
    eg: x0 = np.array([0.1])
    Parameters
    ----------
    f : function handle
        Takes input x
    x0: array 
        The starting point for newton iteration.
    df : function handle
        Function that returns the jacobian
    tol: float
        defines the allowed tolerance between xn and xn+1 as well
        as the value of f(xn).
    maxit: int
        maximum number of iterations mysolve can run.
    Returns
    -------
    x:  The last iteration (hopfully solution to f(x) = 0) 
    converged : bool
        A true false statment on if it converged.
    df(x): array (n x n)
        The jacobian of the last iteration of x.
    '''
    
    
    x = x0.copy()

    for i in range(maxit):
        xn_minus_1 = x.copy()
        f_of_x = f(x)

        jac_inv = np.linalg.inv(df(x))
        x = x - jac_inv.dot(f_of_x)

        converged = np.linalg.norm(xn_minus_1 - x) <= tol
        is_zero = np.linalg.norm(f(x)) <= tol

        if(converged and is_zero):#with in tol then break out of loop
            break
    
    return x , converged , df(x)
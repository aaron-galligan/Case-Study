import numpy as np

def MyIVP(f, x0, tspan, h):
    N = int((tspan[1] - tspan[0])/h) #Total number of steps
    
    #t = np.array([])
    t = np.array([tspan[0]])
    tn = tspan[0] #the current time f is being calculated at.
    
    xn = np.copy(x0)  #The current xn we and finding xn+1 of.
    xt = np.array([xn]) #Array of all xn points
    
    for i in range(N):
        
        #Define two arrays K1 = (k1, l1, m1, ...) and K2 = (k2, l2, m2, ...)  
        K1 = h*f(tn, xn)
        K2 = h*f(tn + h, K1 + xn)

        xn = xn + 0.5*(K1 + K2)

        tn += h
        t = np.append(t, tn)
        xt = np.append(xt, [xn], axis = 0)

    return xt, t, xn
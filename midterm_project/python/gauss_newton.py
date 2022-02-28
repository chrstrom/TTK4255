import numpy as np

def jacobian2point(resfun, p, epsilon):
    r = resfun(p)
    J = np.empty((len(r), len(p)))
    for j in range(len(p)):
        pj0 = p[j]
        p[j] = pj0 + epsilon
        rpos = resfun(p)
        p[j] = pj0 - epsilon
        rneg = resfun(p)
        p[j] = pj0
        J[:,j] = rpos - rneg
    return J/(2.0*epsilon)

def gauss_newton(resfun, jacfun, p0, step_size, num_steps=0, xtol=1e-6):
    r = resfun(p0)
    J = jacfun(p0)
    p = p0.copy()

    #for iteration in range(num_steps):
    step = 0
    p_last = np.zeros_like(p0)
    
    while True:
        step += 1
        A = J.T@J
        b = -J.T@r
        d = np.linalg.solve(A, b)
        p = p + step_size*d
        r = resfun(p)
        J = jacfun(p)
    
        if np.linalg.norm(p - p_last, 2) < xtol:
            print(f"Steps used to converge within tol {xtol}: {step}")
            break

        if step >= num_steps:
            print("Max step count reached!")
            break

        p_last = p
        


    return p

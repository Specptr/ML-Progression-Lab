import numpy as np

def map_feature(x1, x2, degree=6):
    """
    Generate polynomial features:
        1, x1, x2, x1^2, x1*x2, x2^2, ..., x1^6, x1^5*x2, ..., x2^6

    This is essential for ex2data2, whose boundary is nonlinear.
    """
    out = [np.ones(len(x1))]
    for i in range(1, degree+1):
        for j in range(i+1):
            out.append((x1**(i-j)) * (x2**j))
    return np.column_stack(out)
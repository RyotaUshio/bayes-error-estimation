import numpy as np

def B_inner(E, m, t):
    """The function inside the inf in our computable bias bound B(E, m) (see Corollary 2)"""
    return (
        (1/m) * ((t * (1 - t)) / (1 - 2*t))
        +
        np.sqrt(np.pi / (2 * m)) * np.minimum(1, E / t)
    )

def existing_bias_bound(n, m):
    """The bias bound presented by Ishida et al. (2023)"""
    return 1 / (2 * np.sqrt(m)) + np.sqrt((np.log(2 * n * np.sqrt(m))) / m)

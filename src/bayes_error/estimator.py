import numpy as np
import numpy.typing as npt

def bayes_error(soft_labels: npt.NDArray[np.float32]):
    """
    Compute the Bayes error from the given soft labels.

    Args:
        soft_labels: The soft labels as a numpy array of shape (n_samples). Each element is a posterior probability of the positive class.

    Returns:
        The Bayes error.
    """
    return np.mean(
        np.minimum(soft_labels, 1 - soft_labels)
    )

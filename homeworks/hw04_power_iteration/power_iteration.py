import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps
    
    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE

    return  [prev := [0, np.random.randn(len(data))]] and [prev := [float(prev[1].dot(data.dot(prev[1])) / (prev[1].dot(prev[1]))), data.dot(prev[1])/np.linalg.norm(data.dot(prev[1]))] for _ in range(1, num_steps)][-1]
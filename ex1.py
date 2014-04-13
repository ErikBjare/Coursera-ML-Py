import numpy as np

def computeCost(X, y, theta):
    """Does the same thing as computeCost.m and computeCostMulti.m

    Expressed in LaTeX:
        J(\theta) = \frac{1}{2m} \sum_{i=1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2"""

    m = len(y)
    J = 1/(2*m) * ((np.dot(X, theta.T)-y)**2).sum()
    return J

def normalEqn(X, y):
    return np.dot(
            np.linalg.pinv(np.dot(X.T, X)), 
            np.dot(X.T, y))

def test():
    X = np.array([[1], [2]])
    y = np.array([[1], [2]])
    theta = np.array([[1]])

    j = computeCost(X, y, theta)
    print(j)
    assert j == 0

    nTheta = normalEqn(X, y)
    print(nTheta)
    assert theta == nTheta

if __name__ == "__main__":
    test()

from numpy import *
import unittest

def linear_hypothesis(x, theta):
    """The linear hypothesis"""
    return dot(x, theta)

def computeCost(x, y, theta, reg=None):
    """Does the same thing as computeCost.m and computeCostMulti.m"""
    m = len(x)
    J = multiply(1/(2*m), power((linear_hypothesis(x, theta)-y), 2).sum())
    if reg:
        J += regularize(reg, m, theta)
    return J

def normalEqn(x, y):
    """The normal equation"""
    return dot(
               linalg.pinv(dot(x.T, x)),
               dot(x.T,
                   y))

def regularize(reg, m, theta):
    """Regularization"""
    return reg/(2*m)*(theta**2).sum()


class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.xs = []
        self.ys = []
        self.thetas = []

        def add_set(x, y, theta):
            self.xs.append(x)
            self.ys.append(y)
            self.thetas.append(theta)
            #print("X:\n", x)
            #print("Y:\n", y)
            #print("Theta:\n", theta)

        x = matrix([[1, 2]]).T
        y = matrix([[1, 2]]).T
        theta = matrix([1])
        add_set(x, y, theta)


        x = matrix("[[1, 1, 1, 1]; [1, 2, 2, 2]; [1, 2, 3, 3]; [1, 2, 3, 4]]")
        y = matrix("[4; 7; 9; 10]")
        theta = matrix([[1, 1, 1, 1]]).T
        add_set(x, y, theta)

    def test_linear_cost(self):
        for i in range(len(self.xs)):
            #print("Trying case", i)
            j = computeCost(self.xs[i], self.ys[i], self.thetas[i])
            self.assertEqual(j, 0)

    def test_normal_equation(self):
        for i in range(len(self.xs)):
            #print("Trying case", i)
            nTheta = normalEqn(self.xs[i], self.ys[i])
            self.assertTrue(allclose(self.thetas[i], nTheta))


if __name__ == "__main__":
    unittest.main()

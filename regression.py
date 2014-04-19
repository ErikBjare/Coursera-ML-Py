from numpy import *
import unittest

def linear_hypothesis(x, theta):
    """The linear hypothesis"""
    return dot(theta.T, x)

def computeCost(x, y, theta, reg=None):
    """Does the same thing as computeCost.m and computeCostMulti.m"""
    m = len(x)
    diff_squared = square(linear_hypothesis(x, theta)-y)
    print("diff_squared = ", diff_squared)
    J = 1/(2*m) * diff_squared.sum()
    print("J = ", J)
    if reg:
        J += regularize(reg, m, theta)
    return J

def normalEqn(x, y):
    """The normal equation"""
    s1 = linalg.pinv(dot(x.T, x))
    s2 = dot(x, y)
    s3 = s1 * s2
    return s3

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
            print("Case", len(self.xs)-1)
            print("X:\n", x)
            print("Y:\n", y)
            print("Theta:\n", theta)

        x = array([[1], [2]])
        y = array([[5]])
        theta = array([[1], [2]])
        add_set(x, y, theta)

        x = array([[1, 1, 1, 1], [1, 2, 2, 2], [1, 2, 3, 3], [1, 2, 3, 4]])
        y = array([[4], [7], [9], [10]])
        theta = array([[1], [1], [1], [1]])
        add_set(x, y, theta)

    def test_g(self):
        for i in range(len(self.xs)):
            ylh = linear_hypothesis(self.xs[i], self.thetas[i]).T
            thetane = normalEqn(self.xs[i], self.ys[i])
            self.assertTrue(allclose(ylh, self.ys[i]),
                    msg="ERROR on case {}\nY: {}\nY_LH: {}".format(i, self.ys[i], ylh))
            self.assertTrue(allclose(thetane, self.thetas[i]),
                    msg="ERROR on case {}\nTheta: {}\nTheta_NE: {}".format(i, self.thetas[i], thetane))

    def test_linear_cost(self):
        for i in range(len(self.xs)):
            j = computeCost(self.xs[i], self.ys[i], self.thetas[i])
            self.assertEqual(j, 0, msg="ERROR on case {}".format(i))

    def test_normal_equation(self):
        for i in range(len(self.xs)):
            nTheta = normalEqn(self.xs[i], self.ys[i])
            print(self.thetas[i], nTheta)
            self.assertTrue(allclose(self.thetas[i], nTheta),
                    msg="ERROR on case {}:\nTheta: {}\nTheta_NE: {}".format(i, self.thetas[i], nTheta))


if __name__ == "__main__":
    #print(x, theta)

    unittest.main()

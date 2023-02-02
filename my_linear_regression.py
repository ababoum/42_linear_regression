'''A class that contains all methods necessary to perform linear regression'''

import copy
import numpy as np
import matplotlib.pyplot as plt


class MyLinearRegression():
    """
    Description:
    My personnal linear regression class to fit like a boss.
    """

    def __init__(self, thetas, alpha=0.001, max_iter=1000):

        if not isinstance(alpha, (int, float)) or alpha <= 0 or alpha >= 1:
            raise Exception("Alpha must be strictly between 0 and 1")
        if not isinstance(max_iter, int) or max_iter < 0:
            raise Exception("Max_iter must be a positive integer")
        if not MyLinearRegression.is_theta_valid(thetas):
            raise Exception("Wrong format for thetas")

        self.alpha = alpha
        self.max_iter = max_iter
        self.thetas = np.copy(thetas.astype('float64'))
        self.loss_hist = []

    @staticmethod
    def normalize_minmax(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    @staticmethod
    def is_vector_valid(x):
        if not isinstance(x, np.ndarray):
            return False
        if len(x.shape) == 1 and x.shape[0] < 1:
            return False
        if len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] != 1):
            return False
        if x.size == 0:
            return False
        return True

    @staticmethod
    def is_theta_valid(theta):
        if not isinstance(theta, np.ndarray):
            return False
        if len(theta.shape) == 1 and theta.shape != (2,):
            return False
        if len(theta.shape) == 2 and theta.shape != (2, 1):
            return False
        return True

    @staticmethod
    def add_intercept(x):
        try:
            if not isinstance(x, np.ndarray):
                return None

            new_col = np.empty((x.shape[0], 1))
            np.ndarray.fill(new_col, 1.)

            if len(x.shape) == 1:
                return np.concatenate(
                    [new_col, np.array(list([item] for item in x))], axis=1)
            return np.concatenate([new_col, x], axis=1)
        except:
            return None

    @staticmethod
    def simple_gradient(x, y, theta):
        if not MyLinearRegression.is_vector_valid(x) or \
                not MyLinearRegression.is_vector_valid(y):
            return None
        if x.size != y.size:
            return None
        x_p = MyLinearRegression.add_intercept(x)
        res = np.zeros(theta.shape)
        res = (x_p.T @ ((x_p @ theta) - y))
        return res / y.size

    def fit_(self, x, y, normalize=False):
        try:
            if not MyLinearRegression.is_vector_valid(x) or \
                    not MyLinearRegression.is_vector_valid(y):
                return None
            if x.size != y.size:
                return None

            if normalize:
                x_n = copy.deepcopy(MyLinearRegression.normalize_minmax(x))
                y_n = copy.deepcopy(y)
            else:
                x_n = copy.deepcopy(x)
                y_n = copy.deepcopy(y)

            for i in range(self.max_iter):
                gradient = MyLinearRegression.simple_gradient(
                    x_n, y_n, self.thetas)
                tt0 = self.thetas[0] - self.alpha * gradient[0]
                tt1 = self.thetas[1] - self.alpha * gradient[1]
                self.thetas[0] = tt0
                self.thetas[1] = tt1
                if i % 100 == 0:
                    l = self.loss_(y_n, self.predict_(x_n))
                    print("Loss: ", l)

                    # Adapt alpha depending on loss
                    if len(self.loss_hist) > 0 and self.loss_hist[-1] > l:
                        self.alpha *= 1.05
                    elif len(self.loss_hist) > 0 and self.loss_hist[-1] < l:
                        self.alpha /= 10
                    elif len(self.loss_hist) > 0 and self.loss_hist[-1] == l:
                        print(
                            f"Warning: loss has converged. Alpha is now {self.alpha}.")
                        break
                    self.loss_hist += [l]

            if normalize:
                self.thetas[1] /= (x.max() - x.min())
                self.thetas[0] -= x.min() * self.thetas[1]

            # show the loss evolution
            plt.plot(self.loss_hist, label="Loss evolution")
            plt.axhline(y = 0, color = 'r', linestyle = '--')
            plt.title("Loss evolution")
            plt.legend()
            plt.show()

        except Exception as e:
            print("Warning: ", e, " in fit_")
            return None

    def predict_(self, x, normalize=False):
        try:
            if not isinstance(x, np.ndarray) or (len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] != 1))\
                    or (len(x.shape) == 1 and x.shape[0] < 1):
                return None

            if normalize:
                x_n = copy.deepcopy(MyLinearRegression.normalize_minmax(x))
            else:
                x_n = copy.deepcopy(x)

            X = MyLinearRegression.add_intercept(x_n)
            if not isinstance(X, np.ndarray):
                return None
            return np.matmul(X, self.thetas)
        except:
            return None

    def loss_elem_(self, y, y_hat):
        if not MyLinearRegression.is_vector_valid(y) or \
                not MyLinearRegression.is_vector_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None
        return ((y - y_hat) ** 2) / (2 * y.size)

    def loss_(self, y, y_hat):
        if not MyLinearRegression.is_vector_valid(y) or \
                not MyLinearRegression.is_vector_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None
        try:
            return np.sum(self.loss_elem_(y, y_hat))
        except:
            return None

    @staticmethod
    def mse_(y, y_hat):
        if not MyLinearRegression.is_vector_valid(y) or \
                not MyLinearRegression.is_vector_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None
        return np.sum((y-y_hat) ** 2) / y.size

    @staticmethod
    def rmse_(y, y_hat):
        if not MyLinearRegression.is_vector_valid(y) or \
                not MyLinearRegression.is_vector_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None

        return np.sqrt(MyLinearRegression.mse_(y, y_hat))

    @staticmethod
    def mae_(y, y_hat):

        if not MyLinearRegression.is_vector_valid(y) or \
                not MyLinearRegression.is_vector_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None
        return np.sum(np.abs(y - y_hat)) / y.size

    @staticmethod
    def r2score_(y, y_hat):
        if not MyLinearRegression.is_vector_valid(y) or \
                not MyLinearRegression.is_vector_valid(y_hat):
            return None
        if y.size != y_hat.size:
            return None
        return 1 - (MyLinearRegression.mse_(y, y_hat) / np.var(y))

    def plot(self, x, y, plot_options=None, normalize=False):
        if not MyLinearRegression.is_vector_valid(x) or \
                not MyLinearRegression.is_vector_valid(y) or \
                x.size != y.size:
            print("Warning: plotting is impossible (wrong parameters)")
            return None
        try:
            y_hat = self.predict_(x, normalize)
            if plot_options != None:
                plt.xlabel(plot_options['xlabel'])
                plt.ylabel(plot_options['ylabel'])
                plt.scatter(x, y, c='b', label=f'{plot_options["xdatalabel"]}')
                plt.plot(x, y_hat, 'xg--',
                         label=f'{plot_options["ydatalabel"]}')
                plt.legend()
            else:
                plt.scatter(x, y, c='b')
                plt.plot(x, y_hat, 'xg--')
            plt.show()
        except Exception as e:
            print("Warning: plotting is impossible (wrong parameters)")
            print("Error: ", e)
        return None

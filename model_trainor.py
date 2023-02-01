import csv
import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR


def zscore(x):
    return (x - x.mean()) / x.std()

def inputs_err_case():
    print("Wrong format for inputs mileage/price in data.csv")
    exit(1)



if __name__ == "__main__":

    model = MyLR([0, 0], alpha=1e-4, max_iter=10000)

    df = pd.read_csv("data.csv")
    X = np.array(df["km"])
    Y = np.array(df["price"])

    X_norm = zscore(X)
    Y_norm = zscore(Y)

    model.fit_(X_norm, Y_norm)

    params_file = open("params.csv", "w")
    params_file.write("theta0,theta1\n")
    params_file.write(f"{model.theta0:.2f},{model.theta1:.2f}\n")
    params_file.close()
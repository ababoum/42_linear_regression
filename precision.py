import csv
import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR

theta0 = 0
theta1 = 0

def params_err_case():
    print("Wrong format for parameters θ₀ and θ₁ in params.csv")
    exit(1)


with open('params.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for index, row in enumerate(reader):
        if index == 0:  # CSV file header
            continue
        if len(row) != 2:
            params_err_case()
        try:
            theta0 = float(row[0])
            theta1 = float(row[1])
            break
        except ValueError:
            params_err_case()

if __name__ == "__main__":

    model = MyLR(np.array([[theta0], [theta1]]))

    df = pd.read_csv("data.csv")
    X = np.array(df["km"])
    Y = np.array(df["price"])
    Y_hat = model.predict_(X, normalize=False)

    print(f'\nThe model precision is:')
    print(f'{"  MSE:":<11} {model.mse_(Y, Y_hat)}')
    print(f'{"  RMSE:":<11} {model.rmse_(Y, Y_hat)}')
    print(f'{"  MAE:":<11} {model.mae_(Y, Y_hat)}')
    print(f'{"  R2 Score:":<11} {model.r2score_(Y, Y_hat)}')

import csv
import sys
import numpy as np
import pandas as pd
from my_linear_regression import MyLinearRegression as MyLR


theta0 = 0
theta1 = 0

def params_err_case():
    print("Wrong format for parameters θ₀ and θ₁ in params.csv")
    exit(1)

def inputs_err_case():
    print("Wrong format for inputs mileage/price in data.csv")
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

    if (len(sys.argv) == 2):
        if (sys.argv[1] == "reset"):
            theta0 = 0
            theta1 = 0
    model = MyLR(np.array([[theta0], [theta1]]), alpha=1e-10, max_iter=1000000)

    df = pd.read_csv("data.csv")
    X = np.array(df["km"]).reshape(-1, 1)
    Y = np.array(df["price"]).reshape(-1, 1)

    model.fit_(X, Y, normalize=True)

    params_file = open("params.csv", "w")
    params_file.write("theta0,theta1\n")
    print(
        f"{float(model.thetas[0]):.2f},{float(model.thetas[1]):.2f}")
    params_file.write(
        f"{float(model.thetas[0]):.2f},{float(model.thetas[1]):.2f}\n")
    params_file.close()

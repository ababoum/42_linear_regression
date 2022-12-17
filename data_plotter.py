import csv
import matplotlib.pyplot as plt
import numpy as np

# x axis values
x = []
# corresponding y axis values
y = []

def inputs_err_case():
    print("Wrong format for inputs mileage/price in data.csv")
    exit(1)

def params_err_case():
    print("Wrong format for parameters θ₀ and θ₁ in params.csv")
    exit(1)

with open('data.csv') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for index, row in enumerate(reader):
        if index == 0:  # CSV file header
            continue
        if len(row) != 2:
            inputs_err_case()
        try:
            x.append(float(row[0]))
            y.append(float(row[1]))
        except ValueError:
            inputs_err_case()

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


# plotting the points 
plt.scatter(x, y)
  
# naming the x axis
plt.xlabel('Mileage')
# naming the y axis
plt.ylabel('Price')
  
# giving a title to my graph
plt.title('Car price by mileage')


# Add the line of the linear regression
plt.plot(x, list(val * theta1 + theta0 for val in x), 'r')

# function to show the plot
plt.show()
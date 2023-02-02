import csv

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

print("\033[33;1mWelcome to the car price estimator!\033[0m")
while True:
    try:
        mileage = float(input("Enter a mileage to get an estimated price:\n"))
        if mileage < 0:
            print("\033[31mThe value must be a positive number\033[0m")
            continue
        price = mileage * theta1 + theta0
        if price < 0:
            print(
                f'The car is too used to be sold, the estimated price is {0:.2f}')
        else:
            print(f'The estimated price is {mileage * theta1 + theta0:.2f}')
    except EOFError:
        print("Bye!")
        exit(1)
    except ValueError:
        print("\033[31mThe value must be a positive number\033[0m")

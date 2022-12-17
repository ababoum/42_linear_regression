import csv


def inputs_err_case():
    print("Wrong format for inputs mileage/price in data.csv")
    exit(1)


class trainor:

    def __init__(self):
        self.theta0 = 0
        self.theta1 = 0
        self.mileages = []
        self.prices = []

        # read data from file
        with open('data.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for index, row in enumerate(reader):
                if index == 0:  # CSV file header
                    continue
                if len(row) != 2:
                    inputs_err_case()
                try:
                    self.mileages.append(float(row[0]))
                    self.prices.append(float(row[1]))
                except ValueError:
                    inputs_err_case()
            if not self.mileages or not self.prices:
                print("Empty dataset in data.csv")
                exit(1)
            self.m = len(self.mileages)

    def estimated_price(self, mileage: float):
        return mileage * self.theta1 + self.theta0

    def compute_rmsd(self):
        return sum(abs(self.estimated_price(mileage) - price) for (mileage, price) in zip(self.mileages, self.prices)) / self.m

    def update_coeffs(self, learning_rate):
        tmpθ0 = learning_rate * ((1 / self.m) *
                                 sum((self.estimated_price(mileage) - price) for (mileage, price) in zip(self.mileages, self.prices)))
        tmpθ1 = learning_rate * ((1 / self.m) *
                                 sum((self.estimated_price(mileage) - price) * mileage for (mileage, price) in zip(self.mileages, self.prices)))
        self.theta0 -= tmpθ0
        self.theta1 -= tmpθ1

    def get_current_accuracy(self):
        prices_pred = list(self.estimated_price(x) for x in self.mileages)
        p, e = prices_pred, self.prices
        n = len(prices_pred)
        return 1 - sum(
            [
                abs(p[i]-e[i])/e[i]
                for i in range(n)
                if e[i] != 0]
        ) / n


def main():
    rmsd_history = []
    learning_rate = 0.00001
    iterations = 0
    steps = 100
    mod = trainor()

    while True:
        rmsd_history.append(mod.compute_rmsd())
        mod.update_coeffs(learning_rate)
        iterations += 1
        if iterations % steps == 0:
            print(iterations, "epochs elapsed")
            print("Current accuracy is :", mod.get_current_accuracy())
            stop = input("Do you want to stop (y/*)?")
            if stop == "y":
                break

    params_file = open("params.csv", "w")
    params_file.write("theta0,theta1\n")
    params_file.write(f"{mod.theta0:.2f},{mod.theta1:.2f}\n")
    params_file.close()


if __name__ == "__main__":
    main()

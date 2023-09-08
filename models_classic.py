import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from prepare_data import FlightDataPreparator


class FlightDataModeler:
    def __init__(self, prepared_file='./data/prepared_data.pkl'):
        self.prepared_file = prepared_file
        self.model = None

        preparator = FlightDataPreparator(prepared_file=self.prepared_file)
        if not os.path.exists(self.prepared_file):
            preparator.prepare_data()

        self.train_df, self.test_df = pd.read_pickle(self.prepared_file)

    def train(self, regressor):
        X_train = self.train_df.drop('Emissions', axis=1)
        y_train = self.train_df['Emissions']

        self.model = regressor
        self.model.fit(X_train, y_train)

    def test(self):
        if self.model is None:
            print("Please train the model first.")
            return

        X_test = self.test_df.drop('Emissions', axis=1)
        y_test = self.test_df['Emissions']

        predictions = self.model.predict(X_test)
        error = mean_squared_error(y_test, predictions)

        # Return the test data's index, actual emissions, and predictions for saving to CSV later
        return y_test.index, y_test, predictions


regressors = [
    ('LinearRegression', LinearRegression()),
    ('Ridge', Ridge()),
    ('Lasso', Lasso())
]

modeler = FlightDataModeler()
results = {}

for name, reg in regressors:
    modeler.train(reg)
    indices, actuals, preds = modeler.test()

    # Create DataFrame and save to CSV
    results_df = pd.DataFrame({
        'Index': indices,
        'Actual_Emissions': actuals,
        'Predicted_Emissions': preds,
        'Difference': actuals - preds
    })

    results_df.to_csv(f"./data/results_{name}.csv", index=False)
    error = mean_squared_error(actuals, preds)
    results[name] = error

print(results)

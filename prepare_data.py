import pandas as pd
import os
from sklearn.model_selection import train_test_split
from preprocess_data import FlightDataProcessor


class FlightDataPreparator:
    def __init__(self, processed_file='./data/processed_data.pkl', prepared_file='./data/prepared_data.pkl'):
        self.processed_file = processed_file
        self.prepared_file = prepared_file

        # Initialize train_df and test_df
        self.train_df, self.test_df = None, None

        # Check if processed data exists
        if not os.path.exists(self.processed_file):
            processor = FlightDataProcessor()
            processor.preprocess_data()

        # If prepared data exists, load them
        if os.path.exists(self.prepared_file):
            self.train_df, self.test_df = pd.read_pickle(self.prepared_file)

    def prepare_data(self, test_size=0.2):
        # Load processed data
        df = pd.read_pickle(self.processed_file)

        # Split data, while preserving date order with shuffle=False
        self.train_df, self.test_df = train_test_split(df, test_size=test_size, shuffle=False)

        # Save prepared data (both train and test) to a pickle file
        with open(self.prepared_file, 'wb') as f:
            pd.to_pickle((self.train_df, self.test_df), f)  # <-- Note the change here

    def data_summary(self):
        if self.train_df is None or self.test_df is None:
            print("Data has not been prepared yet. Please run prepare_data method first.")
            return

        # Print summaries
        print("==== Training Data Summary ====")
        print(f"Shape: {self.train_df.shape}")
        print("Head:")
        print(self.train_df.head())
        print("Descriptive Statistics:")
        print(self.train_df.describe())

        print("\n==== Testing Data Summary ====")
        print(f"Shape: {self.test_df.shape}")
        print("Head:")
        print(self.test_df.head())
        print("Descriptive Statistics:")
        print(self.test_df.describe())


# Usage (in the same data_preparator.py file or elsewhere after importing FlightDataPreparator)
preparator = FlightDataPreparator()
preparator.prepare_data()
preparator.data_summary()

import pandas as pd


class FlightDataProcessor:
    def __init__(self, file_name='./data/Flights Ionut_play.csv'):
        self.file_name = file_name
        self.df = pd.read_csv(self.file_name)

    def preprocess_data(self):
        # Convert 'Date' column to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%m-%y')
        # Sort by date without setting it as index
        self.df = self.df.sort_values(by='Date')

        # Extract season of year
        self.df['Season'] = self.df['Date'].apply(self._get_season)

        # Extract weekday/weekend and convert it to binary
        self.df['Weekday_Weekend'] = self.df['Date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

        # Drop Departure, Destination, and Fuel_Type (if it has only one unique value)
        columns_to_drop = ['Departure', 'Destination', 'Emissions_Status', 'Date']
        if 'Fuel_Type' in self.df.columns and self.df['Fuel_Type'].nunique() == 1:
            columns_to_drop.append('Fuel_Type')

        self.df.drop(columns=columns_to_drop, inplace=True)

        # One hot encoding for Aircraft_Type and Season
        self.df = pd.get_dummies(self.df, columns=['Aircraft_Type', 'Season'], drop_first=True)

        # Save dataframe as a pickle file
        self.df.to_pickle('./data/processed_data.pkl')

    def _get_season(self, date):
        month = date.month
        if month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
        else:
            return 'Winter'


# Usage
processor = FlightDataProcessor()
processor.preprocess_data()

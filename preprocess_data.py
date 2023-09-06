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

        # Extract weekday/weekend
        self.df['Weekday_Weekend'] = self.df['Date'].dt.dayofweek.apply(lambda x: 'Weekend' if x >= 5 else 'Weekday')

        # Drop Departure and Destination columns
        self.df.drop(columns=['Departure', 'Destination'], inplace=True)

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

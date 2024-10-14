import pandas as pd
import requests
import zipfile
import io

START_DATE = "2023-01-01"
FF5_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

def load_ff5_data(url):
    try:
        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses

        # Extract the CSV file from the zip
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            print("Files in the zip:", z.namelist())
            file_name = z.namelist()[0]
            with z.open(file_name) as f:
                # Read the CSV file, skipping the first three rows
                df = pd.read_csv(f, skiprows=3, sep=',', skipinitialspace=True)

        # Print the first few rows of the DataFrame to inspect it
        print("Raw DataFrame before processing:")
        print(df.head())

        # Rename the first column to 'Date'
        df.rename(columns={df.columns[0]: 'Date'}, inplace=True)

        # Drop any rows where 'Date' is not a valid date
        df = df[pd.to_datetime(df['Date'], format='%Y%m%d', errors='coerce').notna()]

        # Convert 'Date' column to datetime
        df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

        # Filter the DataFrame for dates greater than or equal to START_DATE
        return df[df['Date'] >= START_DATE]

    except Exception as e:
        print(f"An error occurred: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

df_ff5_daily = load_ff5_data(FF5_URL)

print("Processed DataFrame:")
print(df_ff5_daily.head())

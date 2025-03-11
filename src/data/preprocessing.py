import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preprocessing.log')
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path):
    """
    Load the raw Online Retail dataset.
    
    Parameters:
    -----------
    file_path : str
        Path to the raw data file
    
    Returns:
    --------
    pandas.DataFrame
        Loaded raw data
    """
    logger.info(f"Loading data from {file_path}")

    # determine file extension
    _, file_extension = os.path.splitext(file_path)

    if file_extension.lower() == '.csv':
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
    elif file_extension.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    logger.info(f"Data loaded with shape: {df.shape}")
    return df

def clean_data(df):
    """
    Clean the raw Online Retail dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Raw data
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned data
    """
    logger.info("Starting data cleaning")

    # create a copy of the dataframe
    df_clean = df.copy()

    # 1. remove rows with missing values
    initial_rows = len(df_clean)
    df_clean = df_clean.dropna()
    logger.info(f"Removed {initial_rows - len(df_clean)} rows with missing values")

    # 2. remove rows with invalid InvoiceNo (they should start with a number)
    df_clean = df_clean[df_clean['InvoiceNo'].str.match(r'^[0-9]')]

    # 3. remove rows with Quantity <= 0
    df_clean = df_clean[df_clean['Quantity'] > 0]

    # 4. remove rows with UnitPrice <= 0
    df_clean = df_clean[df_clean['UnitPrice'] > 0]

    # 5. convert InvoiceDate to datetime if it's not already
    if not pd.api.types.is_datetime64_any_dtype(df_clean['InvoiceDate']):
        df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

    # 6. add TotalPrice column
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

    # 7. add columns for day, month, year, hour, day of week
    df_clean['Day'] = df_clean['InvoiceDate'].dt.day
    df_clean['Month'] = df_clean['InvoiceDate'].dt.month
    df_clean['Year'] = df_clean['InvoiceDate'].dt.year
    df_clean['Hour'] = df_clean['InvoiceDate'].dt.hour
    df_clean['DayOfWeek'] = df_clean['InvoiceDate'].dt.dayofweek

    logger.info(f"Data cleaning completed. Final shape: {df_clean.shape}")
    return df_clean

def process_data(input_filepath, output_filepath):
    """
    Main function to load, clean and save the processed data.
    
    Parameters:
    -----------
    input_filepath : str
        Path to the raw data file
    output_filepath : str
        Path where the processed file will be saved
    """
    # load data
    df = load_data(input_filepath)

    # clean data
    df_clean = clean_data(df)

    # create directory if it doesn't exist
    output_dir = os.path.dirname(output_filepath)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # determine file extension
    _, file_extension = os.path.splitext(output_filepath)
    
    if file_extension.lower() == '.csv':
        df_clean.to_csv(output_filepath, index=False)
    elif file_extension.lower() == '.parquet':
        df_clean.to_parquet(output_filepath, index=False)
    elif file_extension.lower() in ['.xlsx', '.xls']:
        df_clean.to_excel(output_filepath, index=False)
    else:
        raise ValueError(f"Unsupported output file extension: {file_extension}")
    
    logger.info("Data processing completed successfully")
    return df_clean

def main():
    """Run the data processing scripts."""
    # define file paths
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent.parent

    input_filepath = project_dir / "data" / "raw" / "OnlineRetail.csv"
    output_filepath = project_dir / "data" / "processed" / "online_retail_cleaned.parquet"

    # process data
    process_data(str(input_filepath), str(output_filepath))

if __name__ == "__main__":
    main()
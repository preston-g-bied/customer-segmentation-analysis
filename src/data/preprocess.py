import pandas as pd
import numpy as np
from datetime import datetime

def load_data(filepath):
    """
    Load the Online Retail dataset

    Args:
        filepath (str): Path to the data file

    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        # load data
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None
    
def clean_data(df):
    """
    Clean the dataset by handling missing values, duplicates, and outliers
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    # create a copy of the dataframe
    df_clean = df.copy()

    # display initial info
    print("Initial data info:")
    print(f"Shape: {df_clean.shape}")
    print(f"Missing values:\n{df_clean.isnull().sum()}")

    # remove rows with missing customer ID
    df_clean = df_clean.dropna(subset=['CustomerID'])

    # convert customer ID to integer
    df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)

    # remove rows with negative quantity or price
    df_clean = df_clean[(df_clean['Quantity'] > 0) & (df_clean['UnitPrice'] > 0)]

    # remove duplicates
    df_clean = df_clean.drop_duplicates()

    # convert InvoiceDate to datetime
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])

    # add total price column
    df_clean['TotalPrice'] = df_clean['Quantity'] * df_clean['UnitPrice']

    # display final info
    print("\nCleaned data info:")
    print(f"Shape: {df_clean.shape}")
    print(f"Missing values:\n{df_clean.isnull().sum()}")

    return df_clean

def save_processed_data(df, output_path):
    """
    Save the processed dataframe to a CSV file
    
    Args:
        df (pd.DataFrame): Processed dataframe
        output_path (str): Path to save the processed data
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
    except Exception as e:
        print(f"Error saving processed data: {e}")

if __name__ == "__main__":
    # define file paths
    input_path = "../../data/outline_retail.csv"
    output_path = "../../data/preprocessed_data.csv"

    # load and process data
    raw_data = load_data(input_path)
    if raw_data is not None:
        processed_data = clean_data(raw_data)
        save_processed_data(processed_data, output_path)
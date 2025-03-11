import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('features.log')
    ]
)
logger = logging.getLogger(__name__)

def create_rfm_features(df):
    """
    Create RFM (Recency, Frequency, Monetary) features for customer segmentation.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned data with transactions
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with RFM features for each customer
    """
    logger.info("Creating RFM features")

    # make sure we have a datetime column
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

    # get the max date to calculate recency
    max_date = df['InvoiceDate'].max() + timedelta(days=1)

    # group by customer
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                              # Frequency
        'TotalPrice': 'sum'                                  # Monetary
    })

    # rename columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    # log info about the features
    logger.info(f"RFM features created with shape: {rfm.shape}")
    logger.info(f"RFM statistics:\n{rfm.describe()}")
    
    return rfm

def segment_customers(rfm_df, n_clusters=4):
    """
    Segment customers based on RFM features.
    
    Parameters:
    -----------
    rfm_df : pandas.DataFrame
        DataFrame with RFM features
    n_clusters : int, optional
        Number of clusters for KMeans, by default 4
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with RFM features and segment labels
    """
    logger.info(f"Segmenting customers into {n_clusters} groups")

    # copy the dataframe
    df = rfm_df.copy()

    # scale the data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df)

    # apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Segment'] = kmeans.fit_predict(rfm_scaled)

    # analyze segments
    segment_analysis = df.groupby('Segment').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean'
    })

    logger.info(f"Segment analysis:\n{segment_analysis}")

    # add segment labels
    segment_names = {
        segment_analysis['Monetary'].argmax(): 'High Value',
        segment_analysis['Recency'].argmin(): 'Recent Customers',
        segment_analysis['Frequency'].argmax(): 'Loyal Customers'
    }

    # if we have a segment that's not in the special categories, label it as 'Standard'
    for i in range(n_clusters):
        if i not in segment_names:
            segment_names[i] = 'Standard'

    # map segment numbers to names
    df['Segment_Name'] = df['Segment'].map(segment_names)

    logger.info(f"Customer segmentation completed with segments: {set(df['Segment_Name'])}")
    
    return df

def create_product_features(df):
    """
    Create features for product analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned data with transactions
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with product features
    """
    logger.info("Creating product features")

    # group by product (StockCode)
    product_features = df.groupby('StockCode').agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'Description': 'first'
    })
    
    # rename columns
    product_features.columns = [
        'TotalQuantity', 
        'TotalRevenue', 
        'TransactionCount', 
        'CustomerCount', 
        'Description'
    ]

    # add average price per unit
    product_features['AvgPrice'] = product_features['TotalRevenue'] / product_features['TotalQuantity']

    # add average quantity per transaction
    product_features['AvgQuantityPerTransaction'] = product_features['TotalQuantity'] / product_features['TransactionCount']
    
    # add average revenue per transaction
    product_features['AvgRevenuePerTransaction'] = product_features['TotalRevenue'] / product_features['TransactionCount']
    
    logger.info(f"Product features created with shape: {product_features.shape}")
    
    return product_features

def create_time_features(df):
    """
    Create features for time-based analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Cleaned data with transactions
    
    Returns:
    --------
    tuple of pandas.DataFrame
        DataFrames with daily, weekly, and monthly aggregated features
    """
    logger.info("Creating time-based features")

    # make sure we have a datetime column
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # add date components if they don't exist
    if 'Date' not in df.columns:
        df['Date'] = df['InvoiceDate'].dt.date
    
    # daily features
    daily_features = df.groupby('Date').agg({
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    })
    
    daily_features.columns = [
        'TransactionCount', 
        'CustomerCount', 
        'Revenue', 
        'QuantitySold'
    ]
    
    # add average revenue per transaction
    daily_features['AvgRevenuePerTransaction'] = daily_features['Revenue'] / daily_features['TransactionCount']
    
    # add average items per transaction
    daily_features['AvgItemsPerTransaction'] = daily_features['QuantitySold'] / daily_features['TransactionCount']
    
    # weekly features
    df['Week'] = pd.to_datetime(df['Date']).dt.to_period('W')
    weekly_features = df.groupby('Week').agg({
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    })
    
    weekly_features.columns = [
        'TransactionCount', 
        'CustomerCount', 
        'Revenue', 
        'QuantitySold'
    ]

    # monthly features
    df['Month'] = pd.to_datetime(df['Date']).dt.to_period('M')
    monthly_features = df.groupby('Month').agg({
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum',
        'StockCode': 'nunique'
    })
    
    monthly_features.columns = [
        'TransactionCount', 
        'CustomerCount', 
        'Revenue', 
        'QuantitySold',
        'UniqueProducts'
    ]
    
    logger.info(f"Time features created with shapes: daily={daily_features.shape}, weekly={weekly_features.shape}, monthly={monthly_features.shape}")
    
    return daily_features, weekly_features, monthly_features

def build_features(input_filepath, output_dir):
    """
    Main function to build features from processed data.
    
    Parameters:
    -----------
    input_filepath : str
        Path to the processed data file
    output_dir : str
        Directory where to save the feature files
    """
    # load processed data
    logger.info(f"Loading processed data from {input_filepath}")

    # determine file extension
    _, file_extension = os.path.splitext(input_filepath)
    
    if file_extension.lower() == '.csv':
        df = pd.read_csv(input_filepath)
    elif file_extension.lower() == '.parquet':
        df = pd.read_parquet(input_filepath)
    elif file_extension.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_filepath)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    # create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # build RFM features
    rfm_features = create_rfm_features(df)
    rfm_features.to_parquet(os.path.join(output_dir, 'customer_segments.parquet'))

    # segment customers
    customer_segments = segment_customers(rfm_features)
    customer_segments.to_parquet(os.path.join(output_dir, 'customer_segments.parquet'))

    # build product features
    product_features = create_product_features(df)
    product_features.to_parquet(os.path.join(output_dir, 'product_features.parquet'))

    # build time features
    daily, weekly, monthly = create_time_features(df)
    daily.to_parquet(os.path.join(output_dir, 'daily_features.parquet'))
    weekly.to_parquet(os.path.join(output_dir, 'weekly_features.parquet'))
    monthly.to_parquet(os.path.join(output_dir, 'monthly_features.parquet'))
    
    logger.info("Feature building completed successfully")

def main():
    """Build the features."""
    # define file paths
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent.parent
    
    input_filepath = project_dir / "data" / "processed" / "online_retail_cleaned.parquet"
    output_dir = project_dir / "data" / "processed" / "features"
    
    # build features
    build_features(str(input_filepath), str(output_dir))

if __name__ == "__main__":
    main()
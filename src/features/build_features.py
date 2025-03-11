import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os

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
    
    # Make sure we have a datetime column
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Get the max date to calculate recency
    max_date = df['InvoiceDate'].max() + timedelta(days=1)
    
    # Group by customer
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (max_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    })
    
    # Rename columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary']
    
    # Log info about the features
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
    dict
        Dictionary with clustering metrics
    """
    logger.info(f"Segmenting customers into {n_clusters} groups")
    
    # Copy the dataframe
    df = rfm_df.copy()
    
    # Handle outliers by capping at 99th percentile
    for col in df.columns:
        max_val = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=max_val)
    
    # Scale the data
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(df)
    
    # Try different numbers of clusters to find optimal
    if n_clusters <= 0:  # Auto-detect number of clusters
        logger.info("Auto-detecting optimal number of clusters")
        
        sil_scores = []
        for k in range(2, 11):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(rfm_scaled)
            sil_score = silhouette_score(rfm_scaled, labels)
            sil_scores.append(sil_score)
            logger.info(f"  k={k}, silhouette score={sil_score:.3f}")
        
        # Choose k with highest silhouette score
        optimal_k = np.argmax(sil_scores) + 2  # +2 because we started at k=2
        logger.info(f"Optimal number of clusters: {optimal_k} (silhouette score: {sil_scores[optimal_k-2]:.3f})")
        n_clusters = optimal_k
    
    # Apply KMeans clustering with the selected number of clusters
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(rfm_scaled)
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(rfm_scaled, df['Cluster'])
    logger.info(f"Silhouette score with {n_clusters} clusters: {silhouette_avg:.3f}")
    
    # Analyze segments
    segment_analysis = df.groupby('Cluster').agg({
        'Recency': 'mean',
        'Frequency': 'mean',
        'Monetary': 'mean',
        'Recency': ['mean', 'median', 'count']
    })
    
    logger.info(f"Segment analysis:\n{segment_analysis}")
    
    # Add segment labels
    segment_names = {
        segment_analysis['Monetary']['mean'].idxmax(): 'High Value',
        segment_analysis['Recency']['mean'].idxmin(): 'Recent Customers',
        segment_analysis['Frequency']['mean'].idxmax(): 'Loyal Customers'
    }
    
    # If we have a segment that's not in the special categories, label it as 'Standard'
    for i in range(n_clusters):
        if i not in segment_names:
            segment_names[i] = 'Standard'
    
    # Map segment numbers to names
    df['Segment_Name'] = df['Cluster'].map(segment_names)
    
    logger.info(f"Customer segmentation completed with segments: {set(df['Segment_Name'])}")
    
    # Create a metrics dictionary
    metrics = {
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'segment_sizes': df['Segment_Name'].value_counts().to_dict(),
        'segment_details': {
            segment: {
                'recency_mean': df[df['Segment_Name'] == segment]['Recency'].mean(),
                'frequency_mean': df[df['Segment_Name'] == segment]['Frequency'].mean(),
                'monetary_mean': df[df['Segment_Name'] == segment]['Monetary'].mean(),
                'count': df[df['Segment_Name'] == segment].shape[0],
                'percentage': df[df['Segment_Name'] == segment].shape[0] / df.shape[0] * 100
            } for segment in set(df['Segment_Name'])
        }
    }
    
    # Save metrics to a JSON file
    metrics_file = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "features" / "clustering_metrics.json"
    import json
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info(f"Clustering metrics saved to {metrics_file}")
    
    return df, metrics

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
    
    # Group by product (StockCode)
    product_features = df.groupby('StockCode').agg({
        'Quantity': 'sum',
        'TotalPrice': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'Description': 'first'
    })
    
    # Rename columns
    product_features.columns = [
        'TotalQuantity', 
        'TotalRevenue', 
        'TransactionCount', 
        'CustomerCount', 
        'Description'
    ]
    
    # Add average price per unit
    product_features['AvgPrice'] = product_features['TotalRevenue'] / product_features['TotalQuantity']
    
    # Add average quantity per transaction
    product_features['AvgQuantityPerTransaction'] = product_features['TotalQuantity'] / product_features['TransactionCount']
    
    # Add average revenue per transaction
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
    
    # Make sure we have a datetime column
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Add date components if they don't exist
    if 'Date' not in df.columns:
        df['Date'] = df['InvoiceDate'].dt.date
    
    # Daily features
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
    
    # Add average revenue per transaction
    daily_features['AvgRevenuePerTransaction'] = daily_features['Revenue'] / daily_features['TransactionCount']
    
    # Add average items per transaction
    daily_features['AvgItemsPerTransaction'] = daily_features['QuantitySold'] / daily_features['TransactionCount']
    
    # Weekly features
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
    
    # Monthly features
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
    # Load processed data
    logger.info(f"Loading processed data from {input_filepath}")
    
    # Determine file extension
    _, file_extension = os.path.splitext(input_filepath)
    
    if file_extension.lower() == '.csv':
        df = pd.read_csv(input_filepath)
    elif file_extension.lower() == '.parquet':
        df = pd.read_parquet(input_filepath)
    elif file_extension.lower() in ['.xlsx', '.xls']:
        df = pd.read_excel(input_filepath)
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")
    
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Build RFM features
    rfm_features = create_rfm_features(df)
    rfm_features.to_parquet(os.path.join(output_dir, 'rfm_features.parquet'))
    
    # Segment customers
    customer_segments, clustering_metrics = segment_customers(rfm_features)
    customer_segments.to_parquet(os.path.join(output_dir, 'customer_segments.parquet'))
    
    # Save silhouette score separately for easy access
    with open(os.path.join(output_dir, 'silhouette_score.txt'), 'w') as f:
        f.write(f"{clustering_metrics['silhouette_score']}")
    
    # Build product features
    product_features = create_product_features(df)
    product_features.to_parquet(os.path.join(output_dir, 'product_features.parquet'))
    
    # Build time features
    daily, weekly, monthly = create_time_features(df)
    daily.to_parquet(os.path.join(output_dir, 'daily_features.parquet'))
    weekly.to_parquet(os.path.join(output_dir, 'weekly_features.parquet'))
    monthly.to_parquet(os.path.join(output_dir, 'monthly_features.parquet'))
    
    logger.info("Feature building completed successfully")

def main():
    """Build the features."""
    # Define file paths
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent.parent
    
    input_filepath = project_dir / "data" / "processed" / "online_retail_cleaned.parquet"
    output_dir = project_dir / "data" / "processed" / "features"
    
    # Import os module needed for build_features
    import os
    
    # Build features
    build_features(str(input_filepath), str(output_dir))

if __name__ == "__main__":
    main()
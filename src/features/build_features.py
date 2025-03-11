import pandas as pd
import numpy as np
from datetime import datetime

def create_rfm_features(df, end_date=None):
    """
    Create RFM (Recency, Frequency, Monetary) features from transaction data
    
    Args:
        df (pd.DataFrame): Processed transaction dataframe
        end_date (datetime, optional): Reference date for recency calculation
        
    Returns:
        pd.DataFrame: Dataframe with RFM features
    """
    # if end date is not provided, use max date in the dataset plus one day
    if end_date is None:
        end_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    else:
        end_date = pd.to_datetime(end_date)
    
    # group by CustomerID
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (end_date - x.max()).dats,     # Recency
        'InvoiceNo': 'nunique',                                 # Frequency
        'TotalPrice': 'sum'                                     # Monetary

    })

    # rename columns
    rfm.columns = ['Recency', 'Frequency', 'Monetary']

    print(f"RFM features created for {len(rfm)} customers")

    return rfm

def add_additional_features(df, rfm_df):
    """
    Add additional features to enhance customer segmentation
    
    Args:
        df (pd.DataFrame): Processed transaction dataframe
        rfm_df (pd.DataFrame): RFM dataframe
        
    Returns:
        pd.DataFrame: Enhanced RFM dataframe with additional features
    """
    # create a copy of the RFM dataframe
    enhanced_rfm = rfm_df.copy()

    # calculate average order value
    customer_orders = df.groupby(['CustomerID', 'InvoiceNo']).agg({
        'TotalPrice': 'sum'
    }).reset_index()

    avg_order_value = customer_orders.groupby('CustomerID').agg({
        'TotalPrice': 'mean'
    })
    enhanced_rfm['AvgOrderValue'] = avg_order_value

    # calculate purchase variability (std of order values)
    order_std = customer_orders.groupby('CustomerID').agg({
        'TotalPrice': 'std'
    })
    enhanced_rfm['OrderValueStd'] = order_std.fillna(0) # fill NaN for customers with only one order

    # calculate days between first and last purchase (customer tenure)
    customer_tenure = df.groupby('CustomerID').agg({
        'InvoiceDate': [min, max]
    })
    customer_tenure.columns = ['FirstPurchase', 'LastPurchase']
    customer_tenure['Tenure'] = (customer_tenure['LastPurchase'] - customer_tenure['FirstPurchase']).dt.days
    enhanced_rfm['Tenure'] = customer_tenure['Tenure']

    # calculate average purchase frequency (days between purchases)
    enhanced_rfm['AvgDaysBetweenPurchases'] = enhanced_rfm['Tenure'] / enhanced_rfm['Frequency']
    enhanced_rfm['AvgDaysBetweenPurchases'] = enhanced_rfm['AvgDaysBetweenPurchases'].fillna(0)

    # calculate product diversity (number of unique products)
    product_diversity = df.groupby('CustomerID')['StockCode'].nunique()
    enhanced_rfm['ProductDiversity'] = product_diversity

    # calculate return rate (assuming returns have negative quantities in the original data)
    if 'Quantity' in df.columns:
        returns = df[df['Quantity'] < 0].groupby('CustomerID').size()
        total_orders = df.groupby('CustomerID')['InvoiceNo'].nunique()
        returns_ratio = returns / total_orders
        enhanced_rfm['ReturnRate'] = returns_ratio.fillna(0)

    print("Additional features added to RFM data")

    return enhanced_rfm

def scale_features(df):
    """
    Scale features to prepare for clustering
    
    Args:
        df (pd.DataFrame): Feature dataframe
        
    Returns:
        pd.DataFrame: Scaled feature dataframe
    """
    from sklearn.preprocessing import StandardScaler

    # create a copy of the dataframe
    df_scaled = df.copy()

    # select numeric columns
    numeric_cols = df_scaled.select_dtypes(include=['float64', 'int64']).columns

    # scale numeric features
    scaler = StandardScaler()
    df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])

    print("Features scaled using StandardScaler")

    return df_scaled

def handle_feature_outliers(df, columns=None, threshold=3):
    """
    Handle outliers in the feature dataframe using capping
    
    Args:
        df (pd.DataFrame): Feature dataframe
        columns (list, optional): List of columns to process. If None, process all numeric columns
        threshold (float, optional): Z-score threshold for outlier detection
        
    Returns:
        pd.DataFrame: Dataframe with outliers handled
    """
    # create a copy of the dataframe
    df_no_outliers = df.copy()

    # if columns not specified, use all numeric columns
    if columns is None:
        columns = df.select_dtypes(include=['float64', 'int64']).columns

    for col in columns:
        # calculate z-scores
        z_scores = np.abs((df_no_outliers[col] - df_no_outliers[col].mean()) / df_no_outliers[col].std())

        # identify outliers
        outliers = z_scores > threshold

        if outliers.sum() > 0:
            # calculate upper and lower bounds
            upper_bound = df_no_outliers[col].mean() + threshold * df_no_outliers[col].std()
            lower_bound = df_no_outliers[col].mean() - threshold * df_no_outliers[col].std()

            # cap outliers
            df_no_outliers.loc[z_scores > threshold, col] = df_no_outliers.loc[z_scores > threshold, col].clip(lower=lower_bound, upper=upper_bound)

            print(f"Handled {outliers.sum()} outliers in column '{col}'")

    return df_no_outliers

def save_features(df, output_path):
    """
    Save the feature dataframe to a CSV file
    
    Args:
        df (pd.DataFrame): Feature dataframe
        output_path (str): Path to save the feature data
    """
    try:
        df.to_csv(output_path)
        print(f"Feature data saved to {output_path}")
    except Exception as e:
        print(f"Error saving feature data: {e}")

if __name__ == "__main__":
    # define file paths
    input_path = "../../data/processed_data.csv"
    output_path = "../../data/rfm_features.csv"

    # load processed data
    df = pd.read_csv(input_path, parse_dates=['InvoiceDate'])

    # create RFM features
    rfm_df = create_rfm_features(df)

    # add additional features
    enhanced_rfm = add_additional_features(df, rfm_df)

    # handle outliers
    rfm_no_outliers = handle_feature_outliers(enhanced_rfm)

    # scale features
    rfm_scaled = scale_features(rfm_no_outliers)

    # save both original and scaled features
    rfm_no_outliers.to_csv(output_path.replace('.csv', '_original.csv'))
    rfm_scaled.to_csv(output_path.replace('.csv', '_scaled.csv'))
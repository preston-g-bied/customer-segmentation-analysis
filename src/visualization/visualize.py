import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
from pathlib import Path
import os
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('visualization.log')
    ]
)
logger = logging.getLogger(__name__)

# Set style for matplotlib
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def plot_sales_over_time(time_features_df, time_period='monthly', output_path=None):
    """
    Create a plot showing sales over time.
    
    Parameters:
    -----------
    time_features_df : pandas.DataFrame
        DataFrame with time features
    time_period : str, optional
        Time period to use (daily, weekly, or monthly), by default 'monthly'
    output_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    logger.info(f"Creating {time_period} sales over time plot")
    
    # Reset index to get time period as a column
    df = time_features_df.reset_index()
    
    # Get the correct column name from the index
    # This is the fix - using the actual column name from the index
    if time_period.lower() == 'monthly':
        time_col = 'Month'  # or 'YearMonth' depending on your actual column
    elif time_period.lower() == 'weekly':
        time_col = 'Week'
    elif time_period.lower() == 'daily':
        time_col = 'Date'
    else:
        time_col = df.columns[0]  # Default to first column if unknown
    
    # Check if the column exists
    if time_col not in df.columns:
        # Print available columns for debugging
        logger.error(f"Column '{time_col}' not found. Available columns: {df.columns.tolist()}")
        # Use the first column as a fallback
        time_col = df.columns[0]
        logger.info(f"Using '{time_col}' as the time column instead")
    
    # Convert to string for plotting
    df[time_col] = df[time_col].astype(str)
    
    # Create plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add revenue line
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df['Revenue'],
            name='Revenue',
            line=dict(color='#1f77b4', width=3)
        ),
        secondary_y=False
    )
    
    # Add transaction count line
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df['TransactionCount'],
            name='Transactions',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=f"{time_period.capitalize()} Sales Analysis",
        xaxis_title=time_period.capitalize(),
        yaxis_title="Revenue",
        yaxis2_title="Transaction Count",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=600
    )
    
    # Save figure if output_path is provided
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Plot saved to {output_path}")
    
    return fig

def plot_customer_segments(customer_segments_df, output_path=None):
    """
    Create plots for customer segmentation analysis.
    
    Parameters:
    -----------
    customer_segments_df : pandas.DataFrame
        DataFrame with customer segments
    output_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    logger.info("Creating customer segments plot")
    
    # Create a copy of the dataframe with segment as a column
    df = customer_segments_df.reset_index()
    
    # Create a 3D scatter plot
    fig = px.scatter_3d(
        df, 
        x='Recency', 
        y='Frequency', 
        z='Monetary', 
        color='Segment_Name',
        hover_name='CustomerID',
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.G10
    )
    
    # Update layout
    fig.update_layout(
        title="Customer Segmentation: RFM Analysis",
        scene=dict(
            xaxis_title='Recency (days)',
            yaxis_title='Frequency (# of orders)',
            zaxis_title='Monetary (total spend)'
        ),
        height=800
    )
    
    # Save figure if output_path is provided
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Plot saved to {output_path}")
    
    return fig

def plot_product_analysis(product_features_df, top_n=20, output_path=None):
    """
    Create plots for product analysis.
    
    Parameters:
    -----------
    product_features_df : pandas.DataFrame
        DataFrame with product features
    top_n : int, optional
        Number of top products to show, by default 20
    output_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    logger.info(f"Creating top {top_n} products plot")
    
    # Create a copy of the dataframe with product id as a column
    df = product_features_df.reset_index()
    
    # Sort by total revenue and get top N products
    top_products = df.sort_values('TotalRevenue', ascending=False).head(top_n)
    
    # Create a horizontal bar chart
    fig = px.bar(
        top_products, 
        y='StockCode', 
        x='TotalRevenue', 
        orientation='h',
        hover_data=['Description', 'TotalQuantity', 'CustomerCount'],
        color='AvgPrice',
        color_continuous_scale=px.colors.sequential.Viridis,
        height=800
    )
    
    # Update layout
    fig.update_layout(
        title=f"Top {top_n} Products by Revenue",
        xaxis_title="Total Revenue",
        yaxis_title="Product Code",
        coloraxis_colorbar_title="Avg. Price",
        yaxis=dict(autorange="reversed")
    )
    
    # Save figure if output_path is provided
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Plot saved to {output_path}")
    
    return fig

def plot_country_analysis(df, output_path=None):
    """
    Create plots for country analysis.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original cleaned DataFrame with transactions
    output_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    logger.info("Creating country analysis plot")
    
    # Group by country
    country_data = df.groupby('Country').agg({
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'TotalPrice': 'sum',
        'Quantity': 'sum'
    }).reset_index()
    
    country_data.columns = ['Country', 'Transactions', 'Customers', 'Revenue', 'QuantitySold']
    
    # Sort by revenue
    country_data = country_data.sort_values('Revenue', ascending=False)
    
    # Create a bar chart
    fig = px.bar(
        country_data.head(15), 
        x='Country', 
        y='Revenue',
        color='Customers',
        hover_data=['Transactions', 'QuantitySold'],
        color_continuous_scale=px.colors.sequential.Blues
    )
    
    # Update layout
    fig.update_layout(
        title="Revenue by Country (Top 15)",
        xaxis_title="Country",
        yaxis_title="Revenue",
        coloraxis_colorbar_title="Number of Customers"
    )
    
    # Save figure if output_path is provided
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Plot saved to {output_path}")
    
    return fig

def plot_hourly_patterns(df, output_path=None):
    """
    Create plot for hourly sales patterns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Original cleaned DataFrame with transactions
    output_path : str, optional
        Path to save the plot, by default None
    
    Returns:
    --------
    plotly.graph_objects.Figure
        Plotly figure object
    """
    logger.info("Creating hourly patterns plot")
    
    # Make sure we have Hour column
    if 'Hour' not in df.columns:
        df['Hour'] = df['InvoiceDate'].dt.hour
    
    # Group by hour and day of week
    if 'DayOfWeek' not in df.columns:
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
    
    # Get day names
    df['DayName'] = df['DayOfWeek'].map({
        0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 
        3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
    })
    
    # Group by hour and day name
    hourly_data = df.groupby(['Hour', 'DayName']).agg({
        'InvoiceNo': 'nunique',
        'TotalPrice': 'sum'
    }).reset_index()
    
    hourly_data.columns = ['Hour', 'DayName', 'Transactions', 'Revenue']
    
    # Create heatmap
    fig = px.density_heatmap(
        hourly_data, 
        x='Hour', 
        y='DayName',
        z='Revenue',
        nbinsx=24,
        category_orders={"DayName": [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
            'Friday', 'Saturday', 'Sunday'
        ]},
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Update layout
    fig.update_layout(
        title="Revenue by Hour and Day of Week",
        xaxis_title="Hour of Day",
        yaxis_title="Day of Week",
        coloraxis_colorbar_title="Revenue"
    )
    
    # Save figure if output_path is provided
    if output_path:
        fig.write_html(output_path)
        logger.info(f"Plot saved to {output_path}")
    
    return fig

def generate_all_visualizations(project_dir):
    """
    Generate all visualizations and save them to the output directory.
    
    Parameters:
    -----------
    project_dir : str or Path
        Project directory path
    """
    logger.info("Generating all visualizations")
    
    # Define paths
    data_dir = Path(project_dir) / "data"
    processed_dir = data_dir / "processed"
    features_dir = processed_dir / "features"
    output_dir = Path(project_dir) / "reports" / "figures"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    logger.info("Loading data for visualization")
    
    try:
        # Load cleaned data
        df = pd.read_parquet(processed_dir / "online_retail_cleaned.parquet")
        
        # Load feature data
        monthly_features = pd.read_parquet(features_dir / "monthly_features.parquet")
        daily_features = pd.read_parquet(features_dir / "daily_features.parquet")
        customer_segments = pd.read_parquet(features_dir / "customer_segments.parquet")
        product_features = pd.read_parquet(features_dir / "product_features.parquet")
        
        logger.info("Data loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return
    
    # Generate visualizations
    logger.info("Generating visualizations")
    
    # Sales over time
    monthly_plot = plot_sales_over_time(
        monthly_features, 
        time_period='monthly',
        output_path=str(output_dir / "monthly_sales.html")
    )
    
    daily_plot = plot_sales_over_time(
        daily_features, 
        time_period='daily',
        output_path=str(output_dir / "daily_sales.html")
    )
    
    # Customer segments
    segments_plot = plot_customer_segments(
        customer_segments,
        output_path=str(output_dir / "customer_segments.html")
    )
    
    # Product analysis
    products_plot = plot_product_analysis(
        product_features,
        output_path=str(output_dir / "top_products.html")
    )
    
    # Country analysis
    countries_plot = plot_country_analysis(
        df,
        output_path=str(output_dir / "country_analysis.html")
    )
    
    # Hourly patterns
    hourly_plot = plot_hourly_patterns(
        df,
        output_path=str(output_dir / "hourly_patterns.html")
    )
    
    # Save a visualization index
    viz_index = {
        "visualizations": [
            {"name": "Monthly Sales", "path": "monthly_sales.html", "description": "Monthly sales and transaction trends"},
            {"name": "Daily Sales", "path": "daily_sales.html", "description": "Daily sales and transaction trends"},
            {"name": "Customer Segments", "path": "customer_segments.html", "description": "3D visualization of customer segments based on RFM analysis"},
            {"name": "Top Products", "path": "top_products.html", "description": "Top products by revenue"},
            {"name": "Country Analysis", "path": "country_analysis.html", "description": "Sales analysis by country"},
            {"name": "Hourly Patterns", "path": "hourly_patterns.html", "description": "Sales patterns by hour and day of week"}
        ]
    }
    
    with open(output_dir / "visualization_index.json", "w") as f:
        json.dump(viz_index, f, indent=4)
    
    logger.info(f"All visualizations generated and saved to {output_dir}")

def main():
    """Run the visualization scripts."""
    # Define file paths
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent.parent
    
    # Generate all visualizations
    generate_all_visualizations(project_dir)

if __name__ == "__main__":
    main()
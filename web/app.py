from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import plotly
import plotly.express as px
import json
import os
from pathlib import Path
import logging

# Import visualization functions
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from visualization.visualize import (
    plot_sales_over_time, 
    plot_customer_segments, 
    plot_product_analysis,
    plot_country_analysis,
    plot_hourly_patterns
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('web_app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Get project directory
project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
processed_dir = data_dir / "processed"
features_dir = processed_dir / "features"
reports_dir = project_dir / "reports"
figures_dir = reports_dir / "figures"

# Global data store
data_store = {}

def load_data():
    """Load all necessary data for the dashboard."""
    logger.info("Loading data for the dashboard")
    
    try:
        # Load cleaned data
        data_store['df'] = pd.read_parquet(processed_dir / "online_retail_cleaned.parquet")
        
        # Load feature data
        data_store['monthly_features'] = pd.read_parquet(features_dir / "monthly_features.parquet")
        data_store['daily_features'] = pd.read_parquet(features_dir / "daily_features.parquet")
        data_store['weekly_features'] = pd.read_parquet(features_dir / "weekly_features.parquet")
        data_store['customer_segments'] = pd.read_parquet(features_dir / "customer_segments.parquet")
        data_store['product_features'] = pd.read_parquet(features_dir / "product_features.parquet")
        
        # Basic data stats
        data_store['stats'] = {
            'total_orders': len(data_store['df']['InvoiceNo'].unique()),
            'total_customers': len(data_store['df']['CustomerID'].unique()),
            'total_revenue': data_store['df']['TotalPrice'].sum(),
            'total_products': len(data_store['df']['StockCode'].unique()),
            'average_order_value': data_store['df'].groupby('InvoiceNo')['TotalPrice'].sum().mean(),
            'top_country': data_store['df'].groupby('Country')['TotalPrice'].sum().idxmax()
        }
        
        logger.info("Data loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return False

@app.route('/')
def index():
    """Render the dashboard homepage."""
    return render_template('index.html', stats=data_store['stats'])

@app.route('/sales')
def sales():
    """Render the sales analysis page."""
    return render_template('sales.html')

@app.route('/api/sales_over_time')
def api_sales_over_time():
    """API endpoint for sales over time visualization."""
    time_period = request.args.get('period', 'monthly')
    
    if time_period == 'monthly':
        features_df = data_store['monthly_features']
    elif time_period == 'weekly':
        features_df = data_store['weekly_features']
    elif time_period == 'daily':
        features_df = data_store['daily_features']
    else:
        return jsonify({'error': 'Invalid time period'}), 400
    
    fig = plot_sales_over_time(features_df, time_period=time_period)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/customers')
def customers():
    """Render the customer analysis page."""
    return render_template('customers.html')

@app.route('/api/customer_segments')
def api_customer_segments():
    """API endpoint for customer segmentation visualization."""
    fig = plot_customer_segments(data_store['customer_segments'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/products')
def products():
    """Render the product analysis page."""
    return render_template('products.html')

@app.route('/api/product_analysis')
def api_product_analysis():
    """API endpoint for product analysis visualization."""
    top_n = int(request.args.get('top_n', 20))
    fig = plot_product_analysis(data_store['product_features'], top_n=top_n)
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/geography')
def geography():
    """Render the geographic analysis page."""
    return render_template('geography.html')

@app.route('/api/country_analysis')
def api_country_analysis():
    """API endpoint for country analysis visualization."""
    fig = plot_country_analysis(data_store['df'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/patterns')
def patterns():
    """Render the sales patterns page."""
    return render_template('patterns.html')

@app.route('/api/hourly_patterns')
def api_hourly_patterns():
    """API endpoint for hourly sales patterns visualization."""
    fig = plot_hourly_patterns(data_store['df'])
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

@app.route('/api/data_summary')
def api_data_summary():
    """API endpoint for data summary."""
    return jsonify(data_store['stats'])

@app.route('/api/top_products')
def api_top_products():
    """API endpoint for top products data."""
    top_n = int(request.args.get('top_n', 10))
    
    # Get top products by revenue
    top_products = data_store['product_features'].reset_index() \
        .sort_values('TotalRevenue', ascending=False) \
        .head(top_n)[['StockCode', 'Description', 'TotalRevenue', 'TotalQuantity', 'CustomerCount']] \
        .to_dict('records')
    
    return jsonify(top_products)

@app.route('/api/top_customers')
def api_top_customers():
    """API endpoint for top customers data."""
    top_n = int(request.args.get('top_n', 10))
    
    # Get top customers by monetary value
    top_customers = data_store['customer_segments'].reset_index() \
        .sort_values('Monetary', ascending=False) \
        .head(top_n)[['CustomerID', 'Recency', 'Frequency', 'Monetary', 'Segment_Name']] \
        .to_dict('records')
    
    return jsonify(top_customers)

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)

@app.route('/reports/<path:path>')
def serve_report(path):
    """Serve report files."""
    return send_from_directory(str(reports_dir), path)

@app.route('/download/<path:path>')
def download_data(path):
    """Endpoint to download data files."""
    return send_from_directory(str(data_dir), path)

# Initialize data at startup (replacing the @app.before_first_request)
with app.app_context():
    logger.info("Initializing application data")
    success = load_data()
    if not success:
        logger.error("Failed to load data. Application may not function correctly.")

def main():
    """Run the web application."""
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
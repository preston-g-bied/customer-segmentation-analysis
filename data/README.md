# Data

This directory contains the raw and processed data for the customer segmentation project.

## Data Files

- `online_retail.xlsx`: Original dataset from Kaggle
- `processed_data.csv`: Cleaned and preprocessed dataset
- `rfm_data.csv`: Dataset with RFM (Recency, Frequency, Monetary) features
- `clustered_data.csv`: Final dataset with cluster assignments

## Data Overview

The Online Retail dataset contains transactions from a UK-based online retail store from December 2010 to December 2011. Each row represents a transaction, including:

- **InvoiceNo**: Invoice number (unique to each transaction)
- **StockCode**: Product code
- **Description**: Product name
- **Quantity**: Number of products purchased
- **InvoiceDate**: Date and time of the transaction
- **UnitPrice**: Price per unit
- **CustomerID**: Customer identifier
- **Country**: Country where the customer resides

## Data Acquisition

1. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail)
2. Place the `online_retail.xlsx` file in this directory
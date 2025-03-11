import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('validation.log')
    ]
)
logger = logging.getLogger(__name__)

def validate_schema(df):
    """
    Validate that the dataset has the expected columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to validate
    
    Returns:
    --------
    bool
        True if schema is valid, False otherwise
    """
    expected_columns = [
        'InvoiceNo', 'StockCode', 'Description', 'Quantity', 
        'InvoiceDate', 'UnitPrice', 'CustomerID', 'Country'
    ]

    # check if all expected columns are present
    missing_columns = [col for col in expected_columns if col not in df.columns]

    if missing_columns:
        logger.error(f"Missing columns in dataset: {missing_columns}")
        return False
    
    logger.info("Schema validation passed")
    return True

def validate_data_types(df):
    """
    Validate that the dataset has the expected data types.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to validate
    
    Returns:
    --------
    bool
        True if data types are valid, False otherwise
    """
    # define expected data types
    expected_types = {
        'InvoiceNo': 'object',
        'StockCode': 'object',
        'Description': 'object',
        'Quantity': 'int64',
        'UnitPrice': 'float64',
        'CustomerID': 'float64',  # usually float because it can have NaN values
        'Country': 'object'
    }

    # check if InvoiceDate is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['InvoiceDate']):
        logger.error("InvoiceDate is not datetime type")
        return False
    
    # check other columns
    type_issues = []
    for col, expected_type in expected_types.items():
        if col in df.columns:
            actual_type = str(df[col].dtype)
            if actual_type != expected_type:
                type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")

    if type_issues:
        logger.error(f"Data type issues: {type_issues}")
        return False
    
    logger.info("Data type validation passed")
    return True

def validate_data_values(df):
    """
    Validate that the dataset has valid values.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to validate
    
    Returns:
    --------
    bool
        True if data values are valid, False otherwise
    """
    valid = True

    # check for negative quantities
    neg_quantity = df[df['Quantity'] <= 0].shape[0]
    if neg_quantity > 0:
        logger.error(f"Found {neg_quantity} rows with Quantity <= 0")
        valid = False

    # check for negative prices
    neg_price = df[df['UnitPrice'] <= 0].shape[0]
    if neg_price > 0:
        logger.error(f"Found {neg_price} rows with UnitPrice <= 0")
        valid = False
    
    # check for future dates
    future_dates = df[df['InvoiceDate'] > pd.Timestamp.now()].shape[0]
    if future_dates > 0:
        logger.error(f"Found {future_dates} rows with future InvoiceDate")
        valid = False
    
    # check for valid countries (this is a basic check)
    empty_countries = df[df['Country'].isna() | (df['Country'] == '')].shape[0]
    if empty_countries > 0:
        logger.error(f"Found {empty_countries} rows with empty Country")
        valid = False
    
    # check for duplicate invoices (this is a simple check - might need refinement)
    duplicate_invoices = df.duplicated(['InvoiceNo', 'StockCode', 'Quantity']).sum()
    if duplicate_invoices > 0:
        logger.warning(f"Found {duplicate_invoices} potentially duplicate transactions")
    
    if valid:
        logger.info("Data value validation passed")
    
    return valid

def validate_data(df):
    """
    Run all validations on the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data to validate
    
    Returns:
    --------
    bool
        True if all validations pass, False otherwise
    """
    validations = [
        validate_schema(df),
        validate_data_types(df),
        validate_data_values(df)
    ]

    if all(validations):
        logger.info("All validations passed")
        return True
    else:
        logger.error("One or more validations failed")
        return False
    
def main():
    """Validate the processed data."""
    # define file paths
    current_dir = Path(__file__).resolve().parent
    project_dir = current_dir.parent.parent
    processed_filepath = project_dir / "data" / "processed" / "online_retail_cleaned.parquet"
    
    # load processed data
    logger.info(f"Loading processed data from {processed_filepath}")
    try:
        df = pd.read_parquet(processed_filepath)
    except Exception as e:
        logger.error(f"Error loading processed data: {e}")
        return False
    
    # validate data
    validation_result = validate_data(df)
    
    if validation_result:
        logger.info("Data validation completed successfully")
    else:
        logger.error("Data validation failed")
    
    return validation_result

if __name__ == "__main__":
    main()
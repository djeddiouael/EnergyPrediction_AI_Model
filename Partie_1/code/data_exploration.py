# data_exploration.py
import pandas as pd
import numpy as np


def load_data(file_path):
    """Load the CSV file into a DataFrame"""
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Data loaded successfully: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"❌ File {file_path} not found!")
        return None
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return None


def explore_data(df):
    """Comprehensive data exploration"""
    print("=" * 50)
    print("📊 DATA EXPLORATION REPORT")
    print("=" * 50)

    # Basic information
    print(f"Dataset Shape: {df.shape}")
    print(f"\nColumns: {list(df.columns)}")

    # Data types
    print("\n📋 DATA TYPES:")
    print(df.dtypes)

    # Basic statistics
    print("\n📈 BASIC STATISTICS:")
    print(df.describe())

    # Missing values
    print("\n🔍 MISSING VALUES:")
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100

    missing_df = pd.DataFrame({
        'Missing_Count': missing_data,
        'Missing_Percentage': missing_percent
    })
    print(missing_df[missing_df['Missing_Count'] > 0])

    if missing_df['Missing_Count'].sum() == 0:
        print("✅ No missing values found!")

    # Duplicate analysis
    print("\n🔍 DUPLICATE ANALYSIS:")
    total_duplicates = df.duplicated().sum()
    print(f"Total duplicate rows: {total_duplicates}")

    return {
        'shape': df.shape,
        'missing_values': missing_data.to_dict(),
        'data_types': df.dtypes.to_dict(),
        'duplicates': total_duplicates
    }


def analyze_missing_patterns(df):
    """Analyze pattern of missing values"""
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100

    print("🔍 MISSING VALUES ANALYSIS:")
    print("-" * 40)

    for col in df.columns:
        if missing_data[col] > 0:
            print(f"{col}: {missing_data[col]} missing ({missing_percent[col]:.2f}%)")

    return missing_data


if __name__ == "__main__":
    # Test the exploration
    df = load_data('data.csv')
    if df is not None:
        explore_data(df)
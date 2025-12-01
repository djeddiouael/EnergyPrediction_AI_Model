# data_cleaning.py
import pandas as pd
import numpy as np


def handle_missing_values(df, strategy='auto'):
    """
    Handle missing values based on data type and strategy

    Parameters:
    - df: DataFrame
    - strategy: 'auto', 'median', 'mean', 'mode', 'drop'
    """
    print("\n🔄 HANDLING MISSING VALUES...")
    initial_missing = df.isnull().sum().sum()

    if initial_missing == 0:
        print("✅ No missing values to handle")
        return df, 0

    fixes_applied = 0

    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if strategy == 'auto':
                # Auto-detect best strategy
                if df[col].dtype in ['float64', 'int64']:
                    # Numerical - use median (robust to outliers)
                    fill_value = df[col].median()
                    method = 'median'
                else:
                    # Categorical - use mode
                    fill_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    method = 'mode'
            elif strategy == 'median' and df[col].dtype in ['float64', 'int64']:
                fill_value = df[col].median()
                method = 'median'
            elif strategy == 'mean' and df[col].dtype in ['float64', 'int64']:
                fill_value = df[col].mean()
                method = 'mean'
            elif strategy == 'mode':
                fill_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                method = 'mode'
            elif strategy == 'drop':
                df = df.dropna(subset=[col])
                method = 'drop'
                print(f"🗑️  Dropped rows with missing {col}")
                fixes_applied += df[col].isnull().sum()
                continue
            else:
                fill_value = 0
                method = 'zero'

            if strategy != 'drop':
                missing_count = df[col].isnull().sum()
                df[col] = df[col].fillna(fill_value)
                print(f"📝 Filled {missing_count} missing values in {col} with {method}: {fill_value}")
                fixes_applied += missing_count

    final_missing = df.isnull().sum().sum()
    print(f"✅ Missing values handled: {initial_missing} → {final_missing}")

    return df, fixes_applied


def remove_duplicates(df, keep='first'):
    """
    Remove duplicate rows from the dataset

    Parameters:
    - df: DataFrame
    - keep: 'first', 'last', or False
    """
    print("\n🔄 REMOVING DUPLICATES...")
    initial_count = len(df)

    # Remove exact duplicates
    df_cleaned = df.drop_duplicates(keep=keep)
    duplicates_removed = initial_count - len(df_cleaned)

    print(f"📊 Before: {initial_count} rows")
    print(f"📊 After: {len(df_cleaned)} rows")
    print(f"🗑️  Removed: {duplicates_removed} duplicate rows")

    if duplicates_removed > 0:
        print("✅ Duplicates successfully removed")
    else:
        print("✅ No duplicates found")

    return df_cleaned, duplicates_removed


def detect_outliers_iqr(df):
    """
    Detect outliers using Interquartile Range (IQR) method
    Returns outlier information for each numerical column
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    outlier_info = {}

    print("📊 OUTLIER DETECTION (IQR METHOD)")
    print("-" * 50)

    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_count = len(outliers)

        outlier_info[col] = {
            'count': outlier_count,
            'percentage': (outlier_count / len(df)) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'min_value': df[col].min(),
            'max_value': df[col].max(),
            'outliers': outliers.index.tolist()
        }

        status = "⚠️" if outlier_count > 0 else "✅"
        print(f"{status} {col}: {outlier_count} outliers ({outlier_info[col]['percentage']:.2f}%)")

    return outlier_info


def handle_outliers(df, method='cap', outlier_info=None):
    """
    Handle outliers using specified method

    Parameters:
    - method: 'cap', 'remove', 'ignore'
    - outlier_info: Pre-computed outlier information
    """
    if outlier_info is None:
        outlier_info = detect_outliers_iqr(df)

    print(f"\n🔄 HANDLING OUTLIERS ({method.upper()} METHOD)...")

    df_processed = df.copy()
    numerical_cols = df.select_dtypes(include=[np.number]).columns

    total_outliers_handled = 0

    for col in numerical_cols:
        if outlier_info[col]['count'] > 0:
            if method == 'cap':
                # Cap outliers at the bounds
                lower_bound = outlier_info[col]['lower_bound']
                upper_bound = outlier_info[col]['upper_bound']

                df_processed[col] = np.where(
                    df_processed[col] < lower_bound, lower_bound, df_processed[col]
                )
                df_processed[col] = np.where(
                    df_processed[col] > upper_bound, upper_bound, df_processed[col]
                )

                handled_count = outlier_info[col]['count']
                print(f"📏 Capped {handled_count} outliers in {col}")
                total_outliers_handled += handled_count

            elif method == 'remove':
                # Remove rows with outliers
                outlier_indices = outlier_info[col]['outliers']
                df_processed = df_processed.drop(outlier_indices)
                print(f"🗑️  Removed {len(outlier_indices)} rows with outliers in {col}")
                total_outliers_handled += len(outlier_indices)

    print(f"✅ Total outliers handled: {total_outliers_handled}")
    return df_processed, total_outliers_handled


def clean_numerical_data(df, missing_strategy='auto', outlier_method='cap'):
    """
    Combined function to clean numerical data (missing values + outliers)
    """
    print("🧹 CLEANING NUMERICAL DATA...")

    # Handle missing values
    df_cleaned, missing_fixes = handle_missing_values(df, strategy=missing_strategy)

    # Handle outliers
    df_cleaned, outlier_fixes = handle_outliers(df_cleaned, method=outlier_method)

    return df_cleaned, missing_fixes, outlier_fixes


if __name__ == "__main__":
    # Test the core cleaning functions
    from data_exploration import load_data

    df = load_data('data.csv')
    if df is not None:
        df_cleaned, missing_fixes, outlier_fixes = clean_numerical_data(df)
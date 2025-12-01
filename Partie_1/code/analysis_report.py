# analysis_report.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def setup_visualization():
    """Set up matplotlib and seaborn styling"""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)


def create_cleaning_report(df_original, df_cleaned, cleaning_stats):
    """Create visual comparison between original and cleaned data"""
    setup_visualization()

    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Data Cleaning Impact Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Missing values comparison
    missing_original = df_original.isnull().sum()
    missing_cleaned = df_cleaned.isnull().sum()

    axes[0, 0].bar(missing_original.index, missing_original.values, alpha=0.7, label='Original')
    axes[0, 0].bar(missing_cleaned.index, missing_cleaned.values, alpha=0.7, label='Cleaned')
    axes[0, 0].set_title('Missing Values Comparison')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].legend()

    # Plot 2: Data distribution for key numerical features
    numerical_features = ['Relative_Compactness', 'Surface_Area', 'Heating_Load', 'Cooling_Load']

    for i, feature in enumerate(numerical_features[:2]):
        if feature in df_original.columns:
            axes[0, 1].hist(df_original[feature], alpha=0.7, label=f'Original {feature}', bins=20)
            axes[0, 1].hist(df_cleaned[feature], alpha=0.7, label=f'Cleaned {feature}', bins=20)
    axes[0, 1].set_title('Distribution Comparison (Features 1-2)')
    axes[0, 1].legend()

    for i, feature in enumerate(numerical_features[2:]):
        if feature in df_original.columns:
            axes[0, 2].hist(df_original[feature], alpha=0.7, label=f'Original {feature}', bins=20)
            axes[0, 2].hist(df_cleaned[feature], alpha=0.7, label=f'Cleaned {feature}', bins=20)
    axes[0, 2].set_title('Distribution Comparison (Features 3-4)')
    axes[0, 2].legend()

    # Plot 3: Correlation matrix for cleaned data
    numerical_df = df_cleaned.select_dtypes(include=[np.number])
    if not numerical_df.empty:
        correlation_matrix = numerical_df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                    ax=axes[1, 0], fmt='.2f')
        axes[1, 0].set_title('Cleaned Data Correlation Matrix')

    # Plot 4: Cleaning statistics
    fix_categories = list(cleaning_stats.keys())[:-1]  # Exclude total
    fix_values = [cleaning_stats[key] for key in fix_categories]

    axes[1, 1].bar(fix_categories, fix_values, color=['red', 'orange', 'blue', 'green'])
    axes[1, 1].set_title('Data Cleaning Statistics')
    axes[1, 1].tick_params(axis='x', rotation=45)

    # Plot 5: Data quality summary
    quality_metrics = {
        'Complete Data': len(df_cleaned) / len(df_original) * 100,
        'Missing Free': 100 - (df_cleaned.isnull().sum().sum() / (len(df_cleaned) * len(df_cleaned.columns)) * 100),
        'Duplicate Free': 100 - (df_cleaned.duplicated().sum() / len(df_cleaned) * 100)
    }

    axes[1, 2].bar(quality_metrics.keys(), quality_metrics.values(), color=['green', 'blue', 'orange'])
    axes[1, 2].set_title('Data Quality Metrics (%)')
    axes[1, 2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig('data_cleaning_report.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_statistical_summary(df_cleaned):
    """Generate statistical summary of cleaned data"""
    print("📊 STATISTICAL SUMMARY OF CLEANED DATA")
    print("=" * 50)

    # Basic statistics
    print("\nBasic Statistics:")
    print(df_cleaned.describe())

    # Data quality metrics
    print(f"\nData Quality Metrics:")
    print(f"Total records: {len(df_cleaned)}")
    print(f"Total features: {len(df_cleaned.columns)}")
    print(f"Missing values: {df_cleaned.isnull().sum().sum()}")
    print(f"Duplicate rows: {df_cleaned.duplicated().sum()}")

    # Feature information
    print(f"\nFeature Information:")
    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns

    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")

    # Memory usage
    print(f"\nMemory usage: {df_cleaned.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")


def create_feature_analysis(df_cleaned):
    """Create detailed analysis of features"""
    print("\n🔍 FEATURE ANALYSIS")
    print("=" * 40)

    numerical_cols = df_cleaned.select_dtypes(include=[np.number]).columns

    for col in numerical_cols:
        print(f"\n{col}:")
        print(f"  Range: {df_cleaned[col].min():.2f} to {df_cleaned[col].max():.2f}")
        print(f"  Mean: {df_cleaned[col].mean():.2f}")
        print(f"  Std: {df_cleaned[col].std():.2f}")
        print(f"  Skewness: {df_cleaned[col].skew():.2f}")


if __name__ == "__main__":
    # Test visualization
    from quality_validation import DataCleaningPipeline

    # Load or clean data
    pipeline = DataCleaningPipeline('data.csv')
    cleaned_data = pipeline.run_pipeline()

    if cleaned_data is not None:
        original_data = pipeline.df_original
        cleaning_stats = pipeline.cleaning_stats
        create_cleaning_report(original_data, cleaned_data, cleaning_stats)
        generate_statistical_summary(cleaned_data)
        create_feature_analysis(cleaned_data)
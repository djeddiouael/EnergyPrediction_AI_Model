# quality_validation.py
import pandas as pd
import numpy as np
from datetime import datetime


def validate_building_data(df):
    """
    Validate data based on building physics constraints
    Returns validation results and fixes invalid values
    """
    print("🏗️  BUILDING DATA VALIDATION")
    print("-" * 50)

    validation_results = {}
    fixes_applied = 0

    # Define validation rules
    validation_rules = {
        'Relative_Compactness': {
            'check': lambda x: (x >= 0) & (x <= 1),
            'fix': lambda x: np.clip(x, 0, 1),
            'description': 'Should be between 0 and 1'
        },
        'Surface_Area': {
            'check': lambda x: x > 0,
            'fix': lambda x: np.where(x <= 0, x.median(), x),
            'description': 'Must be positive'
        },
        'Wall_Area': {
            'check': lambda x: x > 0,
            'fix': lambda x: np.where(x <= 0, x.median(), x),
            'description': 'Must be positive'
        },
        'Roof_Area': {
            'check': lambda x: x > 0,
            'fix': lambda x: np.where(x <= 0, x.median(), x),
            'description': 'Must be positive'
        },
        'Overall_Height': {
            'check': lambda x: x > 0,
            'fix': lambda x: np.where(x <= 0, x.median(), x),
            'description': 'Must be positive'
        },
        'Orientation': {
            'check': lambda x: x.isin([2, 3, 4, 5]),
            'fix': lambda x: x.apply(lambda val: val if val in [2, 3, 4, 5] else x.mode()[0]),
            'description': 'Should be 2, 3, 4, or 5'
        },
        'Glazing_Area': {
            'check': lambda x: (x >= 0) & (x <= 1),
            'fix': lambda x: np.clip(x, 0, 1),
            'description': 'Should be between 0 and 1'
        },
        'Glazing_Area_Distribution': {
            'check': lambda x: x.isin([0, 1, 2, 3, 4, 5]),
            'fix': lambda x: np.clip(x, 0, 5),
            'description': 'Should be integer between 0 and 5'
        },
        'Heating_Load': {
            'check': lambda x: x >= 0,
            'fix': lambda x: np.where(x < 0, x.median(), x),
            'description': 'Must be non-negative'
        },
        'Cooling_Load': {
            'check': lambda x: x >= 0,
            'fix': lambda x: np.where(x < 0, x.median(), x),
            'description': 'Must be non-negative'
        }
    }

    # Perform validation and apply fixes
    df_validated = df.copy()

    for column, rules in validation_rules.items():
        if column in df.columns:
            invalid_count = (~rules['check'](df[column])).sum()
            validation_results[column] = {
                'valid': invalid_count == 0,
                'invalid_count': invalid_count,
                'invalid_percentage': (invalid_count / len(df)) * 100,
                'description': rules['description']
            }

            if invalid_count > 0:
                print(f"⚠️  {column}: {invalid_count} invalid values ({(invalid_count / len(df)) * 100:.2f}%)")
                print(f"   Description: {rules['description']}")
                df_validated[column] = rules['fix'](df[column])
                fixes_applied += invalid_count
                print(f"   ✅ Fixed using: {rules['fix'].__name__}")
            else:
                print(f"✅ {column}: All values valid - {rules['description']}")

    print(f"\n🔧 Total validation fixes applied: {fixes_applied}")
    return df_validated, validation_results, fixes_applied


class DataCleaningPipeline:
    """Orchestrate the complete data cleaning process"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.cleaning_log = []
        self.df_original = None
        self.df_cleaned = None
        self.cleaning_stats = {}

    def log_step(self, step_name, details, fixes=0):
        """Log each cleaning step"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = {
            'timestamp': timestamp,
            'step': step_name,
            'details': details,
            'fixes_applied': fixes
        }
        self.cleaning_log.append(log_entry)
        print(f"[{timestamp}] {step_name}: {details} (Fixes: {fixes})")

    def run_pipeline(self, missing_strategy='auto', outlier_method='cap'):
        """Execute the complete cleaning pipeline"""
        print("🚀 STARTING DATA CLEANING PIPELINE")
        print("=" * 60)

        # Import modules
        from data_exploration import load_data, explore_data
        from data_cleaning import handle_missing_values, remove_duplicates, handle_outliers

        # Step 1: Load and explore data
        self.df_original = load_data(self.file_path)
        if self.df_original is None:
            return None

        exploration_results = explore_data(self.df_original)
        self.log_step("Data Exploration", f"Loaded {self.df_original.shape} dataset")

        # Step 2: Handle missing values
        self.df_cleaned, missing_fixes = handle_missing_values(
            self.df_original, strategy=missing_strategy
        )
        self.log_step("Missing Values", f"Applied {missing_strategy} strategy", missing_fixes)

        # Step 3: Remove duplicates
        self.df_cleaned, duplicate_fixes = remove_duplicates(self.df_cleaned)
        self.log_step("Duplicate Removal", "Removed exact duplicates", duplicate_fixes)

        # Step 4: Data validation
        self.df_cleaned, validation_results, validation_fixes = validate_building_data(self.df_cleaned)
        self.log_step("Data Validation", "Applied domain-specific rules", validation_fixes)

        # Step 5: Outlier handling
        self.df_cleaned, outlier_fixes = handle_outliers(self.df_cleaned, method=outlier_method)
        self.log_step("Outlier Handling", f"Applied {outlier_method} method", outlier_fixes)

        # Store cleaning statistics
        self.cleaning_stats = {
            'missing_fixes': missing_fixes,
            'duplicate_fixes': duplicate_fixes,
            'validation_fixes': validation_fixes,
            'outlier_fixes': outlier_fixes,
            'total_fixes': missing_fixes + duplicate_fixes + validation_fixes + outlier_fixes
        }

        # Final summary
        self.generate_summary_report()

        return self.df_cleaned

    def generate_summary_report(self):
        """Generate a comprehensive cleaning summary"""
        print("\n" + "=" * 60)
        print("📋 DATA CLEANING SUMMARY REPORT")
        print("=" * 60)

        if self.df_original is not None and self.df_cleaned is not None:
            print(f"Original dataset: {self.df_original.shape}")
            print(f"Cleaned dataset: {self.df_cleaned.shape}")
            print(f"Rows processed: {len(self.df_original)}")
            print(f"Rows in final dataset: {len(self.df_cleaned)}")

            # Calculate data quality metrics
            original_missing = self.df_original.isnull().sum().sum()
            cleaned_missing = self.df_cleaned.isnull().sum().sum()
            original_duplicates = self.df_original.duplicated().sum()
            cleaned_duplicates = self.df_cleaned.duplicated().sum()

            print(f"\n📊 QUALITY IMPROVEMENT:")
            print(f"Missing values: {original_missing} → {cleaned_missing}")
            print(f"Duplicate rows: {original_duplicates} → {cleaned_duplicates}")

            print(f"\n🔧 CLEANING STATISTICS:")
            print(f"Missing value fixes: {self.cleaning_stats['missing_fixes']}")
            print(f"Duplicate removals: {self.cleaning_stats['duplicate_fixes']}")
            print(f"Validation fixes: {self.cleaning_stats['validation_fixes']}")
            print(f"Outlier treatments: {self.cleaning_stats['outlier_fixes']}")
            print(f"TOTAL FIXES: {self.cleaning_stats['total_fixes']}")

        print(f"\n📝 CLEANING STEPS PERFORMED: {len(self.cleaning_log)}")
        for log in self.cleaning_log:
            print(f"  • {log['step']}: {log['details']} (Fixes: {log['fixes_applied']})")

    def save_cleaned_data(self, output_path='cleaned_building_data.csv'):
        """Save the cleaned dataset"""
        if self.df_cleaned is not None:
            self.df_cleaned.to_csv(output_path, index=False)
            print(f"\n💾 Cleaned data saved to: {output_path}")
            return True
        else:
            print("❌ No cleaned data to save. Run pipeline first.")
            return False

    def get_cleaning_report(self):
        """Return a dictionary with cleaning report"""
        return {
            'original_shape': self.df_original.shape if self.df_original is not None else None,
            'cleaned_shape': self.df_cleaned.shape if self.df_cleaned is not None else None,
            'cleaning_log': self.cleaning_log,
            'cleaning_stats': self.cleaning_stats
        }


if __name__ == "__main__":
    # Test the pipeline
    pipeline = DataCleaningPipeline('data.csv')
    cleaned_data = pipeline.run_pipeline()
    if cleaned_data is not None:
        pipeline.save_cleaned_data()
# main.py
"""
Main script to run the complete data cleaning pipeline
"""

from quality_validation import DataCleaningPipeline
from analysis_report import create_cleaning_report, generate_statistical_summary, create_feature_analysis
import config


def main():
    print("🏗️  BUILDING ENERGY EFFICIENCY DATA CLEANING")
    print("=" * 50)

    # Initialize and run the cleaning pipeline
    pipeline = DataCleaningPipeline(config.INPUT_FILE)

    # Execute the complete pipeline
    cleaned_data = pipeline.run_pipeline(
        missing_strategy=config.MISSING_VALUE_STRATEGY,
        outlier_method=config.OUTLIER_HANDLING_STRATEGY
    )

    if cleaned_data is not None:
        # Save the cleaned data
        pipeline.save_cleaned_data(config.OUTPUT_FILE)

        # Generate analysis and visualization
        print("\n📈 GENERATING ANALYSIS REPORTS...")
        create_cleaning_report(pipeline.df_original, cleaned_data, pipeline.cleaning_stats)
        generate_statistical_summary(cleaned_data)
        create_feature_analysis(cleaned_data)

        # Print final report
        report = pipeline.get_cleaning_report()
        print(f"\n🎉 DATA CLEANING COMPLETED SUCCESSFULLY!")
        print(f"📁 Cleaned data saved as: {config.OUTPUT_FILE}")
        print(f"📊 Report saved as: {config.REPORT_FILE}")
        print(f"🔧 Total fixes applied: {report['cleaning_stats']['total_fixes']}")

        # Show key improvements
        print(f"\n📊 KEY IMPROVEMENTS:")
        print(f"• Missing values: {pipeline.df_original.isnull().sum().sum()} → {cleaned_data.isnull().sum().sum()}")
        print(f"• Duplicate rows: {pipeline.df_original.duplicated().sum()} → {cleaned_data.duplicated().sum()}")
        print(f"• Data quality score: {calculate_quality_score(cleaned_data):.1f}%")

    else:
        print("❌ Data cleaning failed. Please check the input file.")


def calculate_quality_score(df):
    """Calculate a simple data quality score"""
    total_cells = len(df) * len(df.columns)
    missing_score = 100 - (df.isnull().sum().sum() / total_cells * 100)
    duplicate_score = 100 - (df.duplicated().sum() / len(df) * 100)
    return (missing_score + duplicate_score) / 2


if __name__ == "__main__":
    main()
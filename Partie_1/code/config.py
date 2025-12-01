# config.py
# Configuration settings for the data cleaning pipeline

# File paths
INPUT_FILE = 'data.csv'
OUTPUT_FILE = 'cleaned_building_data.csv'
REPORT_FILE = 'cleaning_report.png'

# Cleaning strategies
MISSING_VALUE_STRATEGY = 'auto'  # 'auto', 'median', 'mean', 'mode', 'drop'
OUTLIER_HANDLING_STRATEGY = 'cap'  # 'cap', 'remove', 'ignore'
DUPLICATE_HANDLING_STRATEGY = 'first'  # 'first', 'last', False

# Data validation rules
VALID_ORIENTATIONS = [2, 3, 4, 5]
VALID_GLAZING_DISTRIBUTION = [0, 1, 2, 3, 4, 5]

# Visualization settings
PLOT_STYLE = 'default'
COLOR_PALETTE = 'husl'
FIGURE_SIZE = (12, 8)

# Logging
LOG_LEVEL = 'INFO'
SAVE_REPORTS = True

# Feature specific constraints
FEATURE_CONSTRAINTS = {
    'Relative_Compactness': {'min': 0, 'max': 1},
    'Surface_Area': {'min': 0, 'max': None},
    'Wall_Area': {'min': 0, 'max': None},
    'Roof_Area': {'min': 0, 'max': None},
    'Overall_Height': {'min': 0, 'max': None},
    'Orientation': {'allowed': [2, 3, 4, 5]},
    'Glazing_Area': {'min': 0, 'max': 1},
    'Glazing_Area_Distribution': {'min': 0, 'max': 5},
    'Heating_Load': {'min': 0, 'max': None},
    'Cooling_Load': {'min': 0, 'max': None}
}
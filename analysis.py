import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker # For formatting ticks if needed
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import io
import base64
import os
import time 

# Set a professional plot style globally - keep whitegrid for structure
sns.set_theme(style="whitegrid", palette="muted") # Use muted palette as base
plt.rcParams['figure.dpi'] = 100 # Increase default DPI for potentially sharper images
plt.rcParams['grid.alpha'] = 0.3 # Make grid lines less prominent
plt.rcParams['axes.edgecolor'] = '#6c757d' # Match card border color slightly
plt.rcParams['xtick.color'] = '#adb5bd' # Light grey ticks
plt.rcParams['ytick.color'] = '#adb5bd'
plt.rcParams['axes.labelcolor'] = '#ced4da' # Light grey labels
plt.rcParams['axes.titlecolor'] = '#FFD700' # Gold title

# Update progress
def update_status(task_id, status_dict, status_message, progress_percent):
    if task_id in status_dict:
        status_dict[task_id]['status'] = status_message
        status_dict[task_id]['progress'] = progress_percent
        print(f"[{task_id}] Progress: {progress_percent}% - {status_message}")

# --- Main Analysis Function ---
def run_analysis_with_progress(task_id, file_path, status_dict):
    try:
        update_status(task_id, status_dict, 'Loading data', 5)
        processed_data = load_and_preprocess(task_id, file_path, status_dict) # Pass task_id and dict

        if processed_data is None:
            update_status(task_id, status_dict, 'Failed: Preprocessing Error', 100)
            status_dict[task_id]['error'] = status_dict[task_id].get('error', 'Failed during data loading or preprocessing.')
            status_dict[task_id]['status'] = 'Failed' 
            return 

        update_status(task_id, status_dict, 'Training model', 40)
        results_df, r2_score_val = train_and_predict(task_id, processed_data, status_dict) # Pass task_id and dict

        if results_df is None:
            update_status(task_id, status_dict, 'Failed: Training/Prediction Error', 100)
            status_dict[task_id]['error'] = status_dict[task_id].get('error', 'Failed during model training or prediction.')
            status_dict[task_id]['status'] = 'Failed'
            return

        update_status(task_id, status_dict, 'Generating plot', 85)
        # *** Pass the R2 score to the plot function for potential display ***
        plot_base64 = generate_plot(task_id, results_df, r2_score_val, status_dict) # Pass task_id, dict, and r2

        update_status(task_id, status_dict, 'Generating results table', 95)
        results_html = generate_html_table(results_df)

        # --- Store results in the status dict ---
        status_dict[task_id].update({
            'status': 'Complete',
            'progress': 100,
            'plot_url': plot_base64,
            'data_html': results_html,
            'r2': r2_score_val
        })
        print(f"[{task_id}] Analysis complete. Results stored.")

    except Exception as e:
        print(f"[{task_id}] Unhandled exception in run_analysis_with_progress: {e}")
        import traceback
        traceback.print_exc()
        update_status(task_id, status_dict, f'Failed: {str(e)}', 100)
        status_dict[task_id]['error'] = f"An unexpected error occurred: {str(e)}"
        status_dict[task_id]['status'] = 'Failed'


# --- (load_and_preprocess, train_and_predict) ---

def load_and_preprocess(task_id, file_path, status_dict):
    """Loads and preprocesses the CSV data, updating progress."""
    try:
        update_status(task_id, status_dict, 'Reading CSV', 10)
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"[{task_id}] Error: File not found at {file_path}")
        status_dict[task_id]['error'] = f"File not found: {os.path.basename(file_path)}"
        return None
    except Exception as e:
        print(f"[{task_id}] Error reading CSV: {e}")
        status_dict[task_id]['error'] = f"Error reading CSV: {str(e)}"
        return None

    update_status(task_id, status_dict, 'Cleaning and transforming data', 15)

    # --- Data Cleaning and Feature Engineering ---
    required_cols = ['År', 'År-Månad', 'A5_id', 'BefattningskodText', 'Åldersgrupp',
                     'Antal anställningar', 'Sjukfrånvaro total %', 'Sjukfrånvaro total tim']
    if not all(col in df.columns for col in required_cols):
        print(f"[{task_id}] Error: CSV missing one or more required columns.")
        status_dict[task_id]['error'] = "CSV missing required columns (e.g., År, År-Månad, Sjukfrånvaro total %, etc.)"
        return None

    if df['Sjukfrånvaro total %'].dtype == 'object':
        df['Sjukfrånvaro total %'] = df['Sjukfrånvaro total %'].str.replace('%', '', regex=False)
        df['Sjukfrånvaro total %'] = df['Sjukfrånvaro total %'].str.replace(',', '.', regex=False).str.strip()
    df['Sjukfrånvaro total %'] = pd.to_numeric(df['Sjukfrånvaro total %'], errors='coerce')

    # --- Add intermediate progress updates ---
    update_status(task_id, status_dict, 'Mapping features', 20)

    age_value_counts = df['Åldersgrupp'].value_counts()
    age_class_mapping = {value: idx + 1 for idx, value in enumerate(age_value_counts.index)}
    df['Åldersgrupp_mapped'] = df['Åldersgrupp'].map(age_class_mapping)

    month_mapping = {
        'JAN': 'Jan', 'FEB': 'Feb', 'MAR': 'Mar', 'APR': 'Apr', 'MAJ': 'May', 'JUN': 'Jun',
        'JUL': 'Jul', 'AUG': 'Aug', 'SEP': 'Sep', 'OKT': 'Oct', 'NOV': 'Nov', 'DEC': 'Dec'
    }
    df['År-Månad'] = df['År-Månad'].astype(str).replace(month_mapping, regex=True)

    try:
        df['År-Månad_dt'] = pd.to_datetime(df['År-Månad'], format='%Y-%b', errors='coerce')
    except ValueError:
        try:
             df['År-Månad_dt'] = pd.to_datetime(df['År-Månad'], format='%Y-%m', errors='coerce')
        except Exception as e:
             print(f"[{task_id}] Error parsing 'År-Månad': {e}. Setting to NaT.")
             df['År-Månad_dt'] = pd.NaT

    df['year'] = df['År-Månad_dt'].dt.year
    df['month'] = df['År-Månad_dt'].dt.month

    pos_value_counts = df['BefattningskodText'].value_counts()
    pos_class_mapping = {value: idx + 1 for idx, value in enumerate(pos_value_counts.index)}
    df['BefattningskodText_mapped'] = df['BefattningskodText'].map(pos_class_mapping)

    update_status(task_id, status_dict, 'Handling numerical data', 25)
    numerical_df = df.select_dtypes(include=['number']).copy()

    if 'Sjukfrånvaro total %' in numerical_df.columns:
         numerical_df['Sickness absence total %'] = numerical_df['Sjukfrånvaro total %'] / 100
         numerical_df = numerical_df.drop(columns=['Sjukfrånvaro total %']) 

    numerical_df = numerical_df.rename(columns={
        'Antal anställningar': 'Number of employments',
        'Sjukfrånvaro total tim': 'Sickness absence total hours',
        'Åldersgrupp_mapped': 'Age group mapped',
        'BefattningskodText_mapped': 'Position CodeText_mapped',
        'År': 'year_original'
    })

    if 'year' in df.columns and pd.api.types.is_numeric_dtype(df['year']):
         numerical_df['year'] = df['year']
    if 'month' in df.columns and pd.api.types.is_numeric_dtype(df['month']):
         numerical_df['month'] = df['month']

    if numerical_df['year'].isnull().any() or numerical_df['month'].isnull().any():
        print(f"[{task_id}] Warning: NaNs found in year/month columns after date parsing. Check 'År-Månad' format.")
        status_dict[task_id]['warning'] = "NaNs found in year/month, check 'År-Månad' format."

    update_status(task_id, status_dict, 'Imputing missing values', 30)
    cols_to_impute = numerical_df.columns
    imputer = SimpleImputer(strategy='mean')
    if not numerical_df.empty:
        numerical_df_imputed = pd.DataFrame(imputer.fit_transform(numerical_df), columns=cols_to_impute)
    else:
        numerical_df_imputed = pd.DataFrame(columns=cols_to_impute)

    update_status(task_id, status_dict, 'Adding cyclical features', 35)
    if 'month' in numerical_df_imputed.columns:
        numerical_df_imputed['month_sin'] = np.sin(2 * np.pi * numerical_df_imputed['month'] / 12)
        numerical_df_imputed['month_cos'] = np.cos(2 * np.pi * numerical_df_imputed['month'] / 12)
    else:
        print(f"[{task_id}] Warning: 'month' column not found for cyclical features.")
        numerical_df_imputed['month_sin'] = 0
        numerical_df_imputed['month_cos'] = 0

    if 'A5_id' in df.columns:
         numerical_df_imputed.index = df.index[numerical_df.index]
         numerical_df_imputed['A5_id'] = df['A5_id']

    return numerical_df_imputed


def train_and_predict(task_id, data, status_dict):
    """Trains XGBoost model and makes predictions, updating progress."""

    target_col_name = 'Sickness absence total %'
    potential_features = [
        'A5_id', 'Number of employments', 'Sickness absence total hours',
        'Age group mapped', 'Position CodeText_mapped',
        'year', 'month', 'month_sin', 'month_cos'
    ]
    feature_columns = [col for col in potential_features if col in data.columns and col != target_col_name]

    if target_col_name not in data.columns:
        print(f"[{task_id}] Error: Target column '{target_col_name}' not found.")
        status_dict[task_id]['error'] = f"Target column '{target_col_name}' not found after preprocessing."
        return None, None

    if not feature_columns or 'year' not in feature_columns:
         print(f"[{task_id}] Error: Feature columns (including 'year') are missing.")
         status_dict[task_id]['error'] = "Required feature columns (including 'year') missing after preprocessing."
         return None, None

    data['year'] = pd.to_numeric(data['year'], errors='coerce')
    data = data.dropna(subset=['year'])
    if data.empty:
        print(f"[{task_id}] Error: No valid data remaining after handling year column.")
        status_dict[task_id]['error'] = "No valid data with 'year' found after preprocessing."
        return None, None
    data['year'] = data['year'].astype(int)

    update_status(task_id, status_dict, 'Splitting data', 45)
    max_year = data['year'].max()
    if max_year <= data['year'].min():
        print(f"[{task_id}] Warning: Only one year ({max_year}) found. Using random split instead of year-based split.")
        if len(data) < 2:
             print(f"[{task_id}] Error: Not enough data (rows < 2) to perform train/test split.")
             status_dict[task_id]['error'] = "Not enough data rows to perform train/test split."
             return None, None

        train, test = train_test_split(data, test_size=0.2, random_state=42)
        test_year_label = f"Random 20% ({max_year})"
    else:
        print(f"[{task_id}] Using year {max_year} for testing.")
        train = data[data['year'] < max_year].copy()
        test = data[data['year'] == max_year].copy()
        test_year_label = str(max_year)

    if train.empty or test.empty:
        print(f"[{task_id}] Error: Train or test set is empty after splitting (Test Year: {test_year_label}).")
        status_dict[task_id]['error'] = f"Train or test set empty after split (Test Year: {test_year_label}). Check data distribution."
        return None, None

    X_train = train[feature_columns].copy()
    y_train = train[target_col_name].copy()
    X_test = test[feature_columns].copy()
    y_test_actual = test[target_col_name].copy()

    y_train_mean = np.nanmean(y_train.replace([np.inf, -np.inf], np.nan))
    y_train = y_train.fillna(y_train_mean).replace([np.inf, -np.inf], y_train_mean)

    y_test_actual_mean = np.nanmean(y_test_actual.replace([np.inf, -np.inf], np.nan))
    y_test_actual_filled = y_test_actual.fillna(y_test_actual_mean).replace([np.inf, -np.inf], y_test_actual_mean)

    update_status(task_id, status_dict, 'Fitting XGBoost model', 50)

    feature_imputer = SimpleImputer(strategy='mean')
    X_train = pd.DataFrame(feature_imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(feature_imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

    xgb_model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=42,
    )

    try:
        xgb_model.fit(X_train, y_train, verbose=False)
        update_status(task_id, status_dict, 'Model training complete', 75)
    except Exception as e:
        print(f"[{task_id}] Error during model training: {e}")
        status_dict[task_id]['error'] = f"Error during model training: {str(e)}"
        return None, None

    update_status(task_id, status_dict, 'Making predictions', 80)
    r2 = None
    try:
        predictions = xgb_model.predict(X_test)
        test = test.copy()
        test['predicted_absence_percentage'] = predictions
        test['actual_absence_percentage'] = y_test_actual

        if len(y_test_actual_filled) == len(predictions) and not np.isnan(predictions).any():
             r2 = r2_score(y_test_actual_filled, predictions)
             print(f"[{task_id}] R-squared (R²) on test set ({test_year_label}): {r2:.4f}")
        else:
             print(f"[{task_id}] Warning: Cannot calculate R² score due to length mismatch or NaNs in predictions.")
             r2 = None

    except Exception as e:
        print(f"[{task_id}] Error during prediction or R2 calculation: {e}")
        status_dict[task_id]['warning'] = f"Prediction/R² error: {str(e)}"
        test['predicted_absence_percentage'] = np.nan
        test['actual_absence_percentage'] = y_test_actual
        r2 = None

    if 'month' not in test.columns or 'year' not in test.columns:
         print(f"[{task_id}] Warning: 'month' or 'year' columns missing in test set. Cannot create monthly results.")
         monthly_results = test[['actual_absence_percentage', 'predicted_absence_percentage']].head(20).reset_index()
         monthly_results['Info'] = "Monthly aggregation failed."
    else:
        test['month_year'] = test['year'].astype(str) + '-' + test['month'].astype(int).astype(str).str.zfill(2)

        grouping_cols = []
        if 'A5_id' in test.columns:
            grouping_cols.append('A5_id')
        grouping_cols.append('month_year')

        agg_dict = {}
        if 'predicted_absence_percentage' in test.columns:
             agg_dict['predicted_absence_percentage'] = 'mean'
        if 'actual_absence_percentage' in test.columns:
             agg_dict['actual_absence_percentage'] = 'mean'

        if not agg_dict:
            print(f"[{task_id}] Error: No prediction/actual columns found to aggregate.")
            status_dict[task_id]['error'] = "Aggregation failed: Missing result columns."
            return None, r2

        monthly_results = test.groupby(grouping_cols).agg(agg_dict).reset_index()

        try:
            monthly_results['month_year_dt'] = pd.to_datetime(monthly_results['month_year'], format='%Y-%m')
            monthly_results = monthly_results.sort_values(by=(['A5_id'] if 'A5_id' in grouping_cols else []) + ['month_year_dt'])
            monthly_results = monthly_results.drop(columns=['month_year_dt'])
        except Exception as sort_e:
             print(f"[{task_id}] Warning: Could not sort monthly results by date: {sort_e}")

    monthly_results['test_year_label'] = test_year_label

    return monthly_results, r2


# --- generate_plot ---
def generate_plot(task_id, results_df, r2_score_val, status_dict):
    """Generates a more modern plot from results and returns base64 encoded string."""
    plot_url = None
    fig, ax = plt.subplots(figsize=(12, 6)) # Use object-oriented interface for more control

    # Set background color for the axes area (subtle contrast)
    ax.set_facecolor('#2e343b') # Slightly different dark grey
    fig.patch.set_facecolor('#212529') # Match body background (will be transparent on save)

    if results_df is None or results_df.empty or 'month_year' not in results_df.columns:
        print(f"[{task_id}] Cannot generate plot: Invalid or empty results data.")
        ax.text(0.5, 0.5, 'Plot generation failed.\nCheck data processing steps.',
                 ha='center', va='center', fontsize=12, color='#dc3545', wrap=True, transform=ax.transAxes) # Use axes transform
        ax.set_title("Plot Generation Error", color='#dc3545', fontsize=16, fontweight='bold')
        ax.set_xticks([]) # Hide ticks and labels for error plot
        ax.set_yticks([])
        ax.spines['top'].set_visible(False) # Hide spines
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

    elif 'predicted_absence_percentage' not in results_df.columns:
         print(f"[{task_id}] Cannot generate plot: Missing prediction column.")
         ax.text(0.5, 0.5, 'Plot generation failed.\nMissing prediction results.',
                  ha='center', va='center', fontsize=12, color='#dc3545', wrap=True, transform=ax.transAxes)
         ax.set_title("Plot Generation Error", color='#dc3545', fontsize=16, fontweight='bold')
         ax.set_xticks([])
         ax.set_yticks([])
         ax.spines['top'].set_visible(False)
         ax.spines['right'].set_visible(False)
         ax.spines['bottom'].set_visible(False)
         ax.spines['left'].set_visible(False)

    else:
        # --- Data Prep ---
        if 'A5_id' in results_df.columns and results_df['A5_id'].nunique() > 1:
            plot_data = results_df.groupby('month_year').agg({
                'predicted_absence_percentage': 'mean',
                'actual_absence_percentage': 'mean'
            }).reset_index()
            dept_id_text = "Avg Across Depts"
        else:
             plot_data = results_df
             dept_id_text = f"Dept: {results_df['A5_id'].iloc[0]}" if 'A5_id' in results_df.columns else "Overall"

        # Format month_year for better display if needed (keep original for sorting/plotting)
        try:
            plot_data['month_year_dt'] = pd.to_datetime(plot_data['month_year'], format='%Y-%m')
            plot_data = plot_data.sort_values(by='month_year_dt')
        except Exception:
            print(f"[{task_id}] Could not sort plot data by date.")
            # Continue without date sorting if error occurs


        # --- Melting Data ---
        id_vars = ['month_year']
        value_vars = ['predicted_absence_percentage']
        if 'actual_absence_percentage' in plot_data.columns:
            value_vars.append('actual_absence_percentage')

        # Check if data actually exists for melting
        plot_data_melt = None
        if all(col in plot_data.columns for col in value_vars):
             plot_data_melt = pd.melt(plot_data, id_vars=id_vars,
                                      value_vars=value_vars,
                                      var_name='Data Type', value_name='Absence Percentage')

             # --- Map to friendly names ---
             type_mapping = {
                 'predicted_absence_percentage': 'Predicerad',
                 'actual_absence_percentage': 'Aktuell'
             }
             plot_data_melt['Data Type'] = plot_data_melt['Data Type'].map(type_mapping)
        else:
             print(f"[{task_id}] Required columns for melting missing in plot_data.")


        # --- Plotting (Only if melt was successful) ---
        if plot_data_melt is not None:
            # --- Define modern palette ---
            # Gold for predicted, a nice blue/teal for actual
            palette = {'Predicerad': '#FFD700'} # Gold
            hue_order = ['Predicerad']
            if 'Aktuell' in plot_data_melt['Data Type'].values:
                 palette['Aktuell'] = '#4682B4' # SteelBlue
                 hue_order.append('Aktuell')


            sns.lineplot(
                data=plot_data_melt,
                x='month_year', y='Absence Percentage',
                hue='Data Type',
                hue_order=hue_order,
                palette=palette,
                linewidth=2.5, 
                marker='o',    # Use markers
                markersize=7,  # Slightly larger markers
                markeredgecolor='#212529', # Edge color matching dark background
                markeredgewidth=0.8, # Subtle edge
                ax=ax # Plot on the created axes
            )

            # --- Customization ---
            ax.set_title(f'Månatlig sjukfrånvaro', fontsize=16, fontweight='bold', pad=20)
            ax.set_xlabel('månad-år', fontsize=12, labelpad=10)
            ax.set_ylabel('Frånvaro i procent', fontsize=12, labelpad=10)

            # Format Y-axis as percentage
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

            # Rotate and selectively show X-axis labels
            xticks = sorted(plot_data['month_year'].unique())
            if len(xticks) > 12:
                 tick_indices = np.linspace(0, len(xticks) - 1, num=min(len(xticks), 12), dtype=int)
                 ax.set_xticks([xticks[i] for i in tick_indices])
                 ax.set_xticklabels([xticks[i] for i in tick_indices], rotation=45, ha='right')
            else:
                 ax.set_xticks(xticks)
                 ax.set_xticklabels(xticks, rotation=45, ha='right')

            # --- Legend ---
            legend = ax.legend(title='Data Type', frameon=True, loc='upper left', bbox_to_anchor=(1.02, 1))
            legend.get_frame().set_facecolor('#343a40') # Match card background
            legend.get_frame().set_edgecolor('#6c757d')
            for text in legend.get_texts():
                text.set_color('#f8f9fa') # Light text
            legend.get_title().set_color('#FFD700') # Gold title

            # --- R-squared annotation (optional) ---
            # if r2_score_val is not None:
            #     ax.text(1.02, 0.9, f'R²: {r2_score_val:.3f}', transform=ax.transAxes, fontsize=10,
            #             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='#343a40', ec='#6c757d', alpha=0.8), color='#f8f9fa')


            # --- Final Touches ---
            ax.grid(True, linestyle='--', alpha=0.3, color='#6c757d') # Subtle grid
            sns.despine(ax=ax, top=True, right=True, left=False, bottom=False) # Remove top/right spines
            plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to prevent legend cutoff, leave space right


        else: # Handle case where melting failed after checks
             print(f"[{task_id}] Could not create plot_data_melt.")
             ax.text(0.5, 0.5, 'Plot generation failed.\nData preparation error.',
                      ha='center', va='center', fontsize=12, color='#dc3545', wrap=True, transform=ax.transAxes)
             ax.set_title("Plot Generation Error", color='#dc3545', fontsize=16, fontweight='bold')
             # Hide axes elements like in other error cases
             ax.set_xticks([])
             ax.set_yticks([])
             ax.spines['top'].set_visible(False)
             ax.spines['right'].set_visible(False)
             ax.spines['bottom'].set_visible(False)
             ax.spines['left'].set_visible(False)


    # Save plot to memory with transparency
    try:
        img = io.BytesIO()
        # Save with transparency and tight bounding box
        fig.savefig(img, format='png', bbox_inches='tight', transparent=True, dpi=120)
        plt.close(fig) # Close the plot figure to free memory
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    except Exception as plot_err:
        print(f"[{task_id}] Error saving plot to memory: {plot_err}")
        plt.close(fig) # Ensure figure is closed even on error
        # If saving failed, plot_url remains None, handled by caller

    return plot_url


def generate_html_table(results_df):
    """Formats the results DataFrame into an HTML table string."""
    if results_df is None or results_df.empty:
        return "<p class='text-warning'>No results data to display.</p>"

    display_df = results_df.copy()
    percent_cols = ['actual_absence_percentage', 'predicted_absence_percentage']

    # Format percentage columns
    for col in percent_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
            display_df[col] = display_df[col].apply(lambda x: f"{x*100:.2f}%" if pd.notnull(x) else "N/A")

    # Create mapping from A5_id numbers to names
    a5_mapping = {
        1: "Serviceförvaltningen",
        2: "Utbildningsförvaltningen",
        3: "Omsorgsförvaltningen",
        4: "Samhällsbyggnadsförvaltningen",
        7: "Arbetsmarknad socialförvaltningen",
        8: "Kultur och fritid",
    }

    # Apply mapping if A5_id exists in the DataFrame
    if 'A5_id' in display_df.columns:
        display_df['A5_id'] = display_df['A5_id'].map(a5_mapping).fillna(display_df['A5_id'])

    # Rename columns for display
    column_map = {
        'A5_id': 'Förvaltning',
        'month_year': 'månad-år',
        'actual_absence_percentage': 'Aktuell frånvaro %',
        'predicted_absence_percentage': 'Prognostiserad frånvaro %',
        'test_year_label': 'Prognostiserad år'
    }

    display_cols_ordered = [col for col in column_map.keys() if col in display_df.columns]
    display_df = display_df[display_cols_ordered]
    display_df = display_df.rename(columns=column_map)

    # Generate HTML
    results_html = display_df.to_html(
        classes=["table", "table-striped", "table-hover", "table-bordered", "table-sm", "results-table-data", "table-dark-custom"],
        index=False,
        justify="center",
        border=0,
        na_rep="N/A",
        table_id="results-table-data"
    )
    return results_html

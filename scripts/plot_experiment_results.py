import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np # Import numpy for aggregation

# --- Configuration ---
RESULTS_FILE = 'sample_quality.txt'
TARGET_METRICS = ['atom_stable', 'valid']
PLOT_FILENAME = 'summary_metrics_plot.png' # Base name for the saved plot files

# --- 1. Data Parsing Function (Unchanged) ---
def parse_sample_quality_file(filepath: str) -> list[dict]:
    """
    Parses the concatenated sample_quality.txt file to extract metrics for all runs.
    Returns a list of dictionaries, where each dict represents one sample run's results.
    """
    all_runs_metrics = []
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except Exception:
        return []

    # Use the 'Summary of synthetic molecules...' line as a delimiter
    summary_blocks = re.split(r'(Summary of synthetic molecules from [^\n]*)', content, flags=re.MULTILINE)[1:]
    
    # Iterate over the blocks (Summary line and the subsequent content)
    for i in range(0, len(summary_blocks), 2):
        metrics_content = summary_blocks[i+1] if i+1 < len(summary_blocks) else ""
        
        metrics = {}
        found_all = True

        for metric_name in TARGET_METRICS:
            # Pattern to find the metric name followed by one or more spaces and the float value
            pattern = re.compile(rf'^\s*{metric_name}\s+([\d\.]+)', re.MULTILINE)
            match = pattern.search(metrics_content)
            
            if match:
                metrics[metric_name] = float(match.group(1))
            else:
                metrics[metric_name] = None
                found_all = False
        
        if found_all and metrics:
            all_runs_metrics.append(metrics)
            
    return all_runs_metrics

# --- 2. Helper Functions (Unchanged) ---
def clean_group_name(group_name: str) -> str:
    """Removes the 'test_' prefix from the group name for legend/axis use."""
    if group_name.startswith('test_'):
        return group_name[5:]
    return group_name

def extract_simplified_label(model_name: str, group_name: str) -> tuple[str, float]:
    """
    Extracts the numerical/variable part from the model name for the legend label 
    and a float key for sorting, using conditional division for weights.
    """
    if group_name == 'loss_weights_(h-x)' and model_name.count('_') >= 2:
        parts = model_name.split('_')
        try:
            val1 = float(parts[-2]) 
            val2 = float(parts[-1]) 
            
            total = val1 + val2
            divisor = 1.0
            
            if abs(total - 1000.0) < 1.0:
                divisor = 1000.0
            elif abs(total - 100.0) < 1.0:
                divisor = 100.0
            
            sort_key_decimal = val1 / divisor
            label = f"{int(val1)}/{int(val2)}"
            
            return label, sort_key_decimal
        except ValueError:
            return model_name, 0.0
        
    if model_name.count('_') >= 1:
        numerical_part = model_name.split('_')[-1]
        try:
            sort_key = float(numerical_part)
            return numerical_part, sort_key
        except ValueError:
            return model_name, 0.0
            
    return model_name, 0.0

# --- 3. Data Collection Function (Unchanged) ---
def collect_metrics_for_tests(base_dir: str) -> pd.DataFrame:
    """
    Collects metrics from the single, concatenated sample_quality.txt files.
    Each summary found within the file is treated as a separate row/sample run.
    """
    all_data = []

    for root, dirs, files in os.walk(base_dir):
        
        if RESULTS_FILE in files:
            filepath = os.path.join(root, RESULTS_FILE)
            
            all_runs_metrics = parse_sample_quality_file(filepath)
            
            if all_runs_metrics:
                model_name = os.path.basename(root)
                parent_dir_name = os.path.basename(os.path.dirname(root))
                
                if parent_dir_name.startswith('test_'):
                    raw_group_name = parent_dir_name
                    group_path = os.path.dirname(root) 
                else:
                    raw_group_name = 'Base_Models'
                    group_path = os.path.dirname(root)
                
                group_name = clean_group_name(raw_group_name)
                model_label, sort_key = extract_simplified_label(model_name, group_name)

                for metrics in all_runs_metrics:
                    data_row = {
                        'Group': group_name,
                        'ModelName': model_name,
                        'ModelLabel': model_label,
                        'SortKey': sort_key,
                        'GroupPath': group_path, 
                        **metrics
                    }
                    all_data.append(data_row)

    return pd.DataFrame(all_data)

# --- 4. Plotting Function (MODIFIED to add mean labels on bars) ---
def plot_grouped_bar_chart(df: pd.DataFrame, metrics: list):
    """
    Groups the DataFrame by model, calculates mean and std, and plots the results 
    with mean values labeled on top of the bars.
    """
    
    groups = df['Group'].unique()
    
    # Mapping for metric display names
    metric_display_map = {
        'atom_stable': 'atom stable',
        'valid': 'molecule valid' 
    }
    
    for group_name in groups:
        group_df = df[df['Group'] == group_name].copy()
        
        # 2. Aggregate the data: Calculate Mean and Std Dev for each model
        agg_df = group_df.groupby(['ModelLabel', 'SortKey', 'GroupPath']).agg(
            {metric: ['mean', 'std'] for metric in metrics}
        )
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        agg_df = agg_df.reset_index()

        if agg_df.empty:
            print(f"No aggregated results for Group: {group_name}")
            continue

        # 3. Sort the aggregated data
        agg_df = agg_df.sort_values(by='SortKey', ascending=True)
        
        # --- Reshape Data for Plotting ---
        # Melt Mean values
        plot_long = agg_df.melt(
            id_vars=['ModelLabel', 'SortKey', 'GroupPath'], 
            value_vars=[col for col in agg_df.columns if col.endswith('_mean')], 
            var_name='Metric_Mean_Name', 
            value_name='Mean'
        )
        # Melt Std Dev values
        std_long = agg_df.melt(
            id_vars=['ModelLabel', 'SortKey', 'GroupPath'], 
            value_vars=[col for col in agg_df.columns if col.endswith('_std')], 
            var_name='Metric_Std_Name', 
            value_name='StdDev'
        )
        
        plot_long['Metric'] = plot_long['Metric_Mean_Name'].str.replace('_mean', '')
        std_long['Metric'] = std_long['Metric_Std_Name'].str.replace('_std', '')
        
        # Merge Mean and StdDev
        final_plot_df = pd.merge(
            plot_long.drop(columns='Metric_Mean_Name'), 
            std_long[['Metric', 'ModelLabel', 'StdDev']], 
            on=['Metric', 'ModelLabel']
        )
        
        final_plot_df['StdDev'] = final_plot_df['StdDev'].fillna(0)


        # Determine the saving directory
        save_dir = agg_df['GroupPath'].iloc[0] 
        new_plot_filename = f"{group_name}_{PLOT_FILENAME}"
        save_path = os.path.join(save_dir, new_plot_filename)
        
        # Create figure and axes
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # 3. Plotting using the reshaped data
        model_labels = final_plot_df['ModelLabel'].unique()
        n_models = len(model_labels)
        n_metrics = len(metrics)
        bar_width = 0.8 / n_models
        x_indices = np.arange(n_metrics)

        # List to hold all bar containers for labeling
        all_bars = [] 
        
        for idx, model_label in enumerate(model_labels):
            model_data = final_plot_df[final_plot_df['ModelLabel'] == model_label].copy()
            model_data = model_data.set_index('Metric').reindex(metrics).reset_index()

            bar_positions = x_indices + (idx - n_models / 2) * bar_width + bar_width / 2

            bars = ax.bar(
                bar_positions, 
                model_data['Mean'], 
                yerr=model_data['StdDev'], 
                width=bar_width, 
                capsize=5,
                label=model_label 
            )
            all_bars.append((bars, model_data['Mean']))

        # --- Labeling the Mean Values ---
        for bars, mean_values in all_bars:
            for bar, mean_val in zip(bars, mean_values):
                # Get the height (mean) and position (x coordinate) of the bar
                height = bar.get_height()
                x_pos = bar.get_x() + bar.get_width() / 2.0
                
                # Add the text label
                ax.text(
                    x_pos, 
                    height + 0.015, # Offset label slightly above the bar
                    f'{mean_val:.3f}', # Format to 3 decimal places
                    ha='center', 
                    va='bottom', 
                    fontsize=8
                )


        # --- Formatting ---
        
        ax.set_ylim(0, 1.0)
        
        # Set central ticks for metric groups
        ax.set_xticks(x_indices)
        
        # Metric Display Name: Use the lookup map for display names
        ax.set_xticklabels([metric_display_map.get(m, m.replace('_', ' ')) for m in metrics])
        
        # Apply rotation and alignment fix
        ax.tick_params(axis='x', labelrotation=0)
        
        # Legend Location: Set to 'upper left'
        ax.legend(title=group_name.replace('_', ' ').title(), loc='upper right') 
        ax.grid(axis='y', linestyle='--', alpha=0.6)
    
        plt.tight_layout()
        
        # 4. Save the figure to the determined path
        try:
            fig.savefig(save_path)
            print(f"Plot saved successfully to: {save_path}")
        except Exception as e:
            print(f"Error saving plot to {save_path}: {e}")
        
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Aggregate and plot metrics from GNN sample analysis across test directories.")
    
    parser.add_argument(
        '--base_dir', 
        type=str, 
        default='./with_fixes/',
        help="The root directory (e.g., './for_hpc/trained_models/new/') containing the 'test_*' experiment folders."
    )
    
    args = parser.parse_args()
    
    results_df = collect_metrics_for_tests(args.base_dir)

    if results_df.empty:
        print(f"No results found in directory: {args.base_dir}. Ensure 'sample_quality.txt' exists in model folders with concatenated data.")
    else:
        plot_grouped_bar_chart(results_df, TARGET_METRICS)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        df = pd.read_csv("results.csv")
    except FileNotFoundError:
        print("Error: results.csv not found.")
        return

    # Debug: Print column names to understand the structure
    print("Available columns:", df.columns.tolist())
    print("DataFrame shape:", df.shape)
    
    # Handle duplicate column names by creating unique column names
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    
    print("Columns after deduplication:", df.columns.tolist())
    
    # Use the columns that actually exist in the CSV
    # Based on the CSV structure, we'll use the standalone columns
    accuracy_col = 'best_model_accuracy'
    loss_col = 'best_model_loss' 
    f1_col = 'best_model_f1'
    poison_rate_col = 'poison_rate'  # Use the standalone poison_rate column
    
    # Rename for consistency
    rename_dict = {}
    if accuracy_col in df.columns:
        rename_dict[accuracy_col] = "accuracy"
    if loss_col in df.columns:
        rename_dict[loss_col] = "loss"
    if f1_col in df.columns:
        rename_dict[f1_col] = "f1"
    
    df.rename(columns=rename_dict, inplace=True)

    # Convert to numeric, handling any conversion errors
    for col in ['accuracy', 'loss', 'f1']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Handle poison_rate column
    if poison_rate_col in df.columns:
        df['poison_rate_clean'] = pd.to_numeric(df[poison_rate_col], errors='coerce')
    else:
        print(f"Error: {poison_rate_col} column not found")
        return

    # Filter out rows that are not part of an experiment or have no metrics
    # Keep only rows with experiment names (not empty)
    df = df[df['Experiment'].notna() & (df['Experiment'] != '')]
    df = df.dropna(subset=['accuracy', 'loss', 'poison_rate_clean'])
    
    # Use the clean poison_rate column for sorting
    df = df.sort_values(by="poison_rate_clean").reset_index(drop=True)
    
    # Rename for plotting
    df['poison_rate'] = df['poison_rate_clean']

    # Save baseline accuracy for the report badge
    baseline_df = df[df['poison_rate'] == 0.0]
    if not baseline_df.empty:
        baseline_accuracy = baseline_df['accuracy'].iloc[0]
        with open("baseline_acc.txt", "w") as f:
            f.write(f"{baseline_accuracy:.3f}")
    else:
        with open("baseline_acc.txt", "w") as f:
            f.write("N/A")

    # --- Create the Plot ---
    sns.set_theme(style="whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle('Best Model Performance vs. Data Poisoning Rate', fontsize=16)

    sns.lineplot(data=df, x='poison_rate', y='accuracy', ax=ax1, marker='o', color='b')
    ax1.set_ylabel('Validation Accuracy')
    ax1.set_title('Accuracy degrades as more data is poisoned')

    sns.lineplot(data=df, x='poison_rate', y='loss', ax=ax2, marker='o', color='r')
    ax2.set_xlabel('Poison Rate')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Loss increases as more data is poisoned')
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x*100)}%'))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("summary_plot.png")
    print("\nSaved summary plot to summary_plot.png")

if __name__ == "__main__":
    main()

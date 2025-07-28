import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    try:
        df = pd.read_csv("results.csv")
    except FileNotFoundError:
        print("Error: results.csv not found.")
        return

    # Dynamically find and rename columns
    try:
        poison_col = next(col for col in df.columns if 'poison_rate' in col)
        acc_col = next(col for col in df.columns if 'best_model_accuracy' in col)
        loss_col = next(col for col in df.columns if 'best_model_loss' in col)
    except StopIteration:
        print("Error: Could not find expected metric/param columns in results.csv.")
        return

    df.rename(columns={
        poison_col: "poison_rate",
        acc_col: "accuracy",
        loss_col: "loss"
    }, inplace=True)
    
    df['accuracy'] = pd.to_numeric(df['accuracy'], errors='coerce')
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
    
    df = df[df['Experiment'].notna()].dropna(subset=['accuracy', 'loss'])
    df = df.sort_values(by="poison_rate").reset_index(drop=True)

    # ---- THIS IS THE FIX ----
    # Save baseline accuracy for the report badge, with a check
    baseline_df = df[df['poison_rate'] == 0.0]
    if not baseline_df.empty:
        baseline_accuracy = baseline_df['accuracy'].iloc[0]
        with open("baseline_acc.txt", "w") as f:
            f.write(f"{baseline_accuracy:.3f}")
    else:
        print("Warning: Baseline (poison_rate=0.0) experiment not found. Using 'N/A' for badge.")
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

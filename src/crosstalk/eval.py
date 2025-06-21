import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, precision_recall_curve
import pandas as pd
import numpy as np

def evaluate_and_save_results(model, X_val, y_val, output_dir, result_name='validation'):
    """
    Evaluates the model on a given dataset and saves metrics and plots.

    Args:
        model: The trained model object.
        X_val (scipy.sparse.csr_matrix): The features of the dataset to evaluate.
        y_val (np.ndarray): The labels of the dataset to evaluate.
        output_dir (str): The directory where results will be saved.
        result_name (str): The name for the result files (e.g., 'validation', 'test').
    """
    print(f"\nEvaluating model on the {result_name} set...")
    
    # --- Calculate Metrics ---
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    report_dict = classification_report(y_val, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_val, y_pred_proba)
    report_dict['roc_auc_score'] = auc_score
    
    print(f"\n--- {result_name.capitalize()} Metrics ---")
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC Score: {auc_score:.4f}")
    print("--------------------------\n")
    
    # --- Save Metrics to JSON ---
    metrics_path = os.path.join(output_dir, f'{result_name}_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"{result_name.capitalize()} metrics saved to {metrics_path}")

    # --- Generate and Save Plots ---
    plot_roc_curve(y_val, y_pred_proba, output_dir, result_name)
    plot_confusion_matrix(y_val, y_pred, output_dir, result_name)

def analyze_thresholds(y_true, y_pred_proba, output_dir):
    """
    Analyzes model performance across different thresholds to find the optimal balance
    between precision and recall.

    Args:
        y_true (np.ndarray): The true labels.
        y_pred_proba (np.ndarray): The predicted probabilities for the positive class.
        output_dir (str): The directory where results will be saved.
    """
    print("\nAnalyzing precision-recall trade-off for different thresholds...")
    
    # Calculate precision, recall, and thresholds
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    
    # Create a DataFrame for analysis
    # Note: precision and recall have one more element than thresholds
    pr_df = pd.DataFrame({
        'threshold': thresholds,
        'precision': precision[:-1],
        'recall': recall[:-1]
    })

    # --- Find Optimal Threshold based on F1-Score ---
    # Calculate F1-score for each threshold and add it to the DataFrame
    pr_df['f1_score'] = 2 * (pr_df['precision'] * pr_df['recall']) / (pr_df['precision'] + pr_df['recall'])
    pr_df = pr_df.fillna(0) # Handle cases where precision or recall are 0

    # Find the row with the best F1-score
    best_threshold_row = pr_df.loc[pr_df['f1_score'].idxmax()]
    
    # --- Report and Save the Optimal Threshold ---
    print("\n--- Optimal Threshold Analysis (Maximizing F1-Score) ---")
    print(f"Best Threshold: {best_threshold_row['threshold']:.4f}")
    print(f"Precision at Best Threshold: {best_threshold_row['precision']:.4f}")
    print(f"Recall at Best Threshold: {best_threshold_row['recall']:.4f}")
    print(f"F1-Score at Best Threshold: {best_threshold_row['f1_score']:.4f}")
    print("--------------------------------------------------------")

    # Save summary to JSON
    summary_path = os.path.join(output_dir, 'optimal_threshold_summary.json')
    # Convert numpy types to native Python types for JSON serialization
    best_threshold_row.apply(lambda x: x.item() if hasattr(x, 'item') else x).to_json(summary_path, indent=4)
    print(f"Optimal threshold summary saved to {summary_path}")

    # Save the full analysis to CSV
    csv_path = os.path.join(output_dir, 'threshold_analysis.csv')
    pr_df.to_csv(csv_path, index=False)
    print(f"Threshold analysis saved to {csv_path}")

    # Plot the precision-recall trade-off curve
    plt.figure(figsize=(10, 7))
    plt.plot(pr_df['threshold'], pr_df['precision'], label='Precision', color='blue')
    plt.plot(pr_df['threshold'], pr_df['recall'], label='Recall', color='green')
    plt.title('Precision and Recall vs. Classification Threshold')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plot_path = os.path.join(output_dir, 'precision_recall_tradeoff.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Precision-recall trade-off plot saved to {plot_path}")
    
    return best_threshold_row['threshold']

def plot_roc_curve(y_true, y_pred_proba, output_dir, result_name):
    """Generates and saves the ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve - {result_name.capitalize()} Set')
    plt.legend(loc="lower right")
    
    plot_path = os.path.join(output_dir, f'{result_name}_roc_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve plot saved to {plot_path}")

def plot_confusion_matrix(y_true, y_pred, output_dir, result_name):
    """Generates and saves the confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Enriched', 'Enriched'],
                yticklabels=['Not Enriched', 'Enriched'])
    plt.title(f'Confusion Matrix - {result_name.capitalize()} Set')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plot_path = os.path.join(output_dir, f'{result_name}_confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}")

def plot_explained_variance(svd_model, output_dir):
    """
    Generates and saves a scree plot for the explained variance of the SVD components.

    Args:
        svd_model: The fitted TruncatedSVD model object.
        output_dir (str): The directory where the plot will be saved.
    """
    print("\nPlotting explained variance for dimensionality reduction...")
    
    exp_var = svd_model.explained_variance_ratio_
    cum_exp_var = np.cumsum(exp_var)
    n_components = len(exp_var)
    
    plt.figure(figsize=(12, 7))
    plt.bar(range(1, n_components + 1), exp_var, alpha=0.6, align='center',
            label='Individual explained variance')
    plt.step(range(1, n_components + 1), cum_exp_var, where='mid',
             label='Cumulative explained variance', color='red')
    plt.ylabel('Explained Variance Ratio')
    plt.xlabel('Principal Component Index')
    plt.title('Explained Variance by SVD Components')
    plt.legend(loc='best')
    plt.grid(True)
    
    total_var = svd_model.explained_variance_ratio_.sum()
    print(f"Total variance explained by {n_components} components: {total_var:.4f}")

    plot_path = os.path.join(output_dir, 'svd_explained_variance.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"SVD explained variance plot saved to {plot_path}") 
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix

def evaluate_and_save_results(model, X_val, y_val, output_dir):
    """
    Evaluates the model on the validation set and saves metrics and plots.

    Args:
        model: The trained model object.
        X_val (scipy.sparse.csr_matrix): The validation features.
        y_val (np.ndarray): The validation labels.
        output_dir (str): The directory where results will be saved.
    """
    print("\n[4/4] Evaluating model on the validation set...")
    
    # --- Calculate Metrics ---
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    report_dict = classification_report(y_val, y_pred, output_dict=True)
    auc_score = roc_auc_score(y_val, y_pred_proba)
    report_dict['roc_auc_score'] = auc_score
    
    print("\n--- Validation Metrics ---")
    print(classification_report(y_val, y_pred))
    print(f"ROC AUC Score: {auc_score:.4f}")
    print("--------------------------\n")
    
    # --- Save Metrics to JSON ---
    metrics_path = os.path.join(output_dir, 'validation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"Validation metrics saved to {metrics_path}")

    # --- Generate and Save Plots ---
    plot_roc_curve(y_val, y_pred_proba, output_dir)
    plot_confusion_matrix(y_val, y_pred, output_dir)

def plot_roc_curve(y_true, y_pred_proba, output_dir):
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
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    plot_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"ROC curve plot saved to {plot_path}")

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Generates and saves the confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Enriched', 'Enriched'],
                yticklabels=['Not Enriched', 'Enriched'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    
    plot_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Confusion matrix plot saved to {plot_path}") 
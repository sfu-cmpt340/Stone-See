import json
import os
from datetime import datetime


def load_metrics():
    """
    Load evaluation metrics from JSON file.
    
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    metrics_path = "results/evaluation_metrics.json"
    if not os.path.exists(metrics_path):
        print(f"Error: Metrics file not found at {metrics_path}")
        print("Please run evaluation first: python src/evaluation/evaluate_model.py")
        return None
    
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def load_training_history():
    """
    Load training history from JSON file.
    
    Returns:
        history: Dictionary containing training history
    """
    history_path = "results/models/resnet18_kidney_history.json"
    if not os.path.exists(history_path):
        return None
    
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    return history


def format_number(value, decimals=2, percentage=False):
    """
    Format a number for display in report.
    
    Args:
        value: Number to format
        decimals: Number of decimal places
        percentage: If True, multiply by 100 and add % sign
    
    Returns:
        Formatted string
    """
    if percentage:
        return f"{value * 100:.{decimals}f}%"
    return f"{value:.{decimals}f}"


def generate_report_content(metrics, history=None):
    """
    Generate the report content (same for both text and LaTeX).
    
    Args:
        metrics: Dictionary containing evaluation metrics
        history: Dictionary containing training history (optional)
    
    Returns:
        report_data: Dictionary containing all report sections
    """
    # Get current date and time
    current_date = datetime.now().strftime("%B %d, %Y")
    current_time = datetime.now().strftime("%H:%M:%S")
    
    # Extract metrics
    class_names = metrics.get('class_names', ['Non-Stone', 'Stone'])
    
    # Report sections
    report_data = {
        'title': 'Kidney Stone Detection Model - Evaluation Report',
        'date': current_date,
        'time': current_time,
        'model_name': 'ResNet-18',
        'task': 'Binary Classification (Kidney Stone Detection)',
        'classes': class_names,
        
        # Overall metrics
        'overall_metrics': {
            'accuracy': format_number(metrics['accuracy'], percentage=True),
            'precision': format_number(metrics['precision'], percentage=True),
            'recall': format_number(metrics['recall'], percentage=True),
            'f1_score': format_number(metrics['f1_score'], percentage=True),
            'roc_auc': format_number(metrics['roc_auc'], decimals=3)
        },
        
        # Per-class metrics
        'per_class_metrics': []
    }
    
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        report_data['per_class_metrics'].append({
            'class_name': class_name,
            'precision': format_number(metrics['precision_per_class'][i], percentage=True),
            'recall': format_number(metrics['recall_per_class'][i], percentage=True),
            'f1_score': format_number(metrics['f1_per_class'][i], percentage=True)
        })
    
    # Confusion matrix
    cm = metrics['confusion_matrix']
    report_data['confusion_matrix'] = {
        'true_non_stone_pred_non_stone': cm[0][0],
        'true_non_stone_pred_stone': cm[0][1],
        'true_stone_pred_non_stone': cm[1][0],
        'true_stone_pred_stone': cm[1][1]
    }
    
    # Training information (if available)
    if history:
        best_val_acc = max(history['val_accuracy']) if history['val_accuracy'] else 0
        final_train_acc = history['train_accuracy'][-1] if history['train_accuracy'] else 0
        final_val_acc = history['val_accuracy'][-1] if history['val_accuracy'] else 0
        
        report_data['training_info'] = {
            'epochs': len(history['epochs']),
            'best_val_accuracy': format_number(best_val_acc / 100, percentage=True),
            'final_train_accuracy': format_number(final_train_acc / 100, percentage=True),
            'final_val_accuracy': format_number(final_val_acc / 100, percentage=True)
        }
    else:
        report_data['training_info'] = None
    
    return report_data


def generate_text_report(report_data):
    """
    Generate text format report.
    
    Args:
        report_data: Dictionary containing report content
    
    Returns:
        text: Complete report as text string
    """
    text = []
    
    # Title
    text.append("=" * 80)
    text.append(report_data['title'].center(80))
    text.append("=" * 80)
    text.append("")
    
    # Date and time
    text.append(f"Generated on: {report_data['date']} at {report_data['time']}")
    text.append("")
    
    # Model information
    text.append("MODEL INFORMATION")
    text.append("-" * 80)
    text.append(f"Model Architecture: {report_data['model_name']}")
    text.append(f"Task: {report_data['task']}")
    text.append(f"Classes: {', '.join(report_data['classes'])}")
    text.append("")
    
    # Training information (if available)
    if report_data['training_info']:
        text.append("TRAINING INFORMATION")
        text.append("-" * 80)
        text.append(f"Total Epochs: {report_data['training_info']['epochs']}")
        text.append(f"Best Validation Accuracy: {report_data['training_info']['best_val_accuracy']}")
        text.append(f"Final Training Accuracy: {report_data['training_info']['final_train_accuracy']}")
        text.append(f"Final Validation Accuracy: {report_data['training_info']['final_val_accuracy']}")
        text.append("")
    
    # Overall metrics
    text.append("OVERALL PERFORMANCE METRICS")
    text.append("-" * 80)
    text.append(f"Accuracy:  {report_data['overall_metrics']['accuracy']}")
    text.append(f"Precision: {report_data['overall_metrics']['precision']}")
    text.append(f"Recall:    {report_data['overall_metrics']['recall']}")
    text.append(f"F1-Score:  {report_data['overall_metrics']['f1_score']}")
    text.append(f"ROC-AUC:   {report_data['overall_metrics']['roc_auc']}")
    text.append("")
    
    # Per-class metrics
    text.append("PER-CLASS PERFORMANCE METRICS")
    text.append("-" * 80)
    for class_metrics in report_data['per_class_metrics']:
        text.append(f"\n{class_metrics['class_name']}:")
        text.append(f"  Precision: {class_metrics['precision']}")
        text.append(f"  Recall:    {class_metrics['recall']}")
        text.append(f"  F1-Score:  {class_metrics['f1_score']}")
    text.append("")
    
    # Confusion matrix
    text.append("CONFUSION MATRIX")
    text.append("-" * 80)
    text.append(f"                    Predicted")
    text.append(f"                  {report_data['classes'][0]:<15} {report_data['classes'][1]:<15}")
    text.append(f"True {report_data['classes'][0]:<15} {report_data['confusion_matrix']['true_non_stone_pred_non_stone']:<15} {report_data['confusion_matrix']['true_non_stone_pred_stone']:<15}")
    text.append(f"True {report_data['classes'][1]:<15} {report_data['confusion_matrix']['true_stone_pred_non_stone']:<15} {report_data['confusion_matrix']['true_stone_pred_stone']:<15}")
    text.append("")
    
    # Interpretation
    text.append("INTERPRETATION")
    text.append("-" * 80)
    text.append(f"The model achieved an overall accuracy of {report_data['overall_metrics']['accuracy']} on the test set.")
    text.append(f"The F1-score of {report_data['overall_metrics']['f1_score']} indicates a good balance between precision and recall.")
    text.append(f"The ROC-AUC score of {report_data['overall_metrics']['roc_auc']} demonstrates the model's ability to distinguish between classes.")
    text.append("")
    
    # Footer
    text.append("=" * 80)
    text.append("End of Report")
    text.append("=" * 80)
    
    return "\n".join(text)


def generate_latex_report(report_data):
    """
    Generate LaTeX format report (same content as text version).
    
    Args:
        report_data: Dictionary containing report content
    
    Returns:
        latex: Complete report as LaTeX string
    """
    latex = []
    
    # LaTeX document header
    latex.append("\\documentclass[11pt,a4paper]{article}")
    latex.append("\\usepackage[utf8]{inputenc}")
    latex.append("\\usepackage{geometry}")
    latex.append("\\geometry{margin=1in}")
    latex.append("\\usepackage{booktabs}")
    latex.append("\\usepackage{graphicx}")
    latex.append("\\usepackage{float}")
    latex.append("\\title{" + report_data['title'] + "}")
    latex.append("\\author{KidneyNet Project}")
    latex.append("\\date{" + report_data['date'] + "}")
    latex.append("")
    latex.append("\\begin{document}")
    latex.append("\\maketitle")
    latex.append("")
    
    # Model information section
    latex.append("\\section{Model Information}")
    latex.append("\\begin{itemize}")
    latex.append(f"\\item \\textbf{{Model Architecture:}} {report_data['model_name']}")
    latex.append(f"\\item \\textbf{{Task:}} {report_data['task']}")
    latex.append(f"\\item \\textbf{{Classes:}} {', '.join(report_data['classes'])}")
    latex.append("\\end{itemize}")
    latex.append("")
    
    # Training information (if available)
    if report_data['training_info']:
        latex.append("\\section{Training Information}")
        latex.append("\\begin{itemize}")
        latex.append(f"\\item \\textbf{{Total Epochs:}} {report_data['training_info']['epochs']}")
        latex.append(f"\\item \\textbf{{Best Validation Accuracy:}} {report_data['training_info']['best_val_accuracy']}")
        latex.append(f"\\item \\textbf{{Final Training Accuracy:}} {report_data['training_info']['final_train_accuracy']}")
        latex.append(f"\\item \\textbf{{Final Validation Accuracy:}} {report_data['training_info']['final_val_accuracy']}")
        latex.append("\\end{itemize}")
        latex.append("")
    
    # Overall metrics section
    latex.append("\\section{Overall Performance Metrics}")
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{lr}")
    latex.append("\\toprule")
    latex.append("Metric & Value \\\\")
    latex.append("\\midrule")
    latex.append(f"Accuracy & {report_data['overall_metrics']['accuracy']} \\\\")
    latex.append(f"Precision & {report_data['overall_metrics']['precision']} \\\\")
    latex.append(f"Recall & {report_data['overall_metrics']['recall']} \\\\")
    latex.append(f"F1-Score & {report_data['overall_metrics']['f1_score']} \\\\")
    latex.append(f"ROC-AUC & {report_data['overall_metrics']['roc_auc']} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Overall Performance Metrics}")
    latex.append("\\end{table}")
    latex.append("")
    
    # Per-class metrics section
    latex.append("\\section{Per-Class Performance Metrics}")
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append("Class & Precision & Recall & F1-Score \\\\")
    latex.append("\\midrule")
    for class_metrics in report_data['per_class_metrics']:
        class_name = class_metrics['class_name'].replace('_', '\\_')
        latex.append(f"{class_name} & {class_metrics['precision']} & {class_metrics['recall']} & {class_metrics['f1_score']} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Per-Class Performance Metrics}")
    latex.append("\\end{table}")
    latex.append("")
    
    # Confusion matrix section
    latex.append("\\section{Confusion Matrix}")
    latex.append("\\begin{table}[H]")
    latex.append("\\centering")
    latex.append("\\begin{tabular}{lcc}")
    latex.append("\\toprule")
    latex.append(" & \\multicolumn{2}{c}{Predicted} \\\\")
    latex.append("\\cmidrule(lr){2-3}")
    latex.append("True & " + report_data['classes'][0] + " & " + report_data['classes'][1] + " \\\\")
    latex.append("\\midrule")
    latex.append(f"{report_data['classes'][0]} & {report_data['confusion_matrix']['true_non_stone_pred_non_stone']} & {report_data['confusion_matrix']['true_non_stone_pred_stone']} \\\\")
    latex.append(f"{report_data['classes'][1]} & {report_data['confusion_matrix']['true_stone_pred_non_stone']} & {report_data['confusion_matrix']['true_stone_pred_stone']} \\\\")
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\caption{Confusion Matrix}")
    latex.append("\\end{table}")
    latex.append("")
    
    # Add figures if they exist
    if os.path.exists("results/figures/confusion_matrix.png"):
        latex.append("\\begin{figure}[H]")
        latex.append("\\centering")
        latex.append("\\includegraphics[width=0.6\\textwidth]{results/figures/confusion_matrix.png}")
        latex.append("\\caption{Confusion Matrix Visualization}")
        latex.append("\\end{figure}")
        latex.append("")
    
    if os.path.exists("results/figures/roc_curve.png"):
        latex.append("\\begin{figure}[H]")
        latex.append("\\centering")
        latex.append("\\includegraphics[width=0.6\\textwidth]{results/figures/roc_curve.png}")
        latex.append("\\caption{ROC Curve}")
        latex.append("\\end{figure}")
        latex.append("")
    
    # Interpretation section
    latex.append("\\section{Interpretation}")
    latex.append("\\begin{itemize}")
    latex.append(f"\\item The model achieved an overall accuracy of {report_data['overall_metrics']['accuracy']} on the test set.")
    latex.append(f"\\item The F1-score of {report_data['overall_metrics']['f1_score']} indicates a good balance between precision and recall.")
    latex.append(f"\\item The ROC-AUC score of {report_data['overall_metrics']['roc_auc']} demonstrates the model's ability to distinguish between classes.")
    latex.append("\\end{itemize}")
    latex.append("")
    
    # Footer
    latex.append("\\end{document}")
    
    return "\n".join(latex)


def main():
    """Main function to generate both text and LaTeX reports."""
    print("=" * 60)
    print("Generating Evaluation Report")
    print("=" * 60)
    
    # Load metrics
    metrics = load_metrics()
    if metrics is None:
        return
    
    # Load training history (optional)
    history = load_training_history()
    
    # Generate report content
    print("\nGenerating report content...")
    report_data = generate_report_content(metrics, history)
    
    # Generate text report
    print("Generating text report...")
    text_report = generate_text_report(report_data)
    
    # Generate LaTeX report
    print("Generating LaTeX report...")
    latex_report = generate_latex_report(report_data)
    
    # Save reports
    os.makedirs("results", exist_ok=True)
    
    text_path = "results/evaluation_report.txt"
    latex_path = "results/evaluation_report.tex"
    
    with open(text_path, 'w') as f:
        f.write(text_report)
    
    with open(latex_path, 'w') as f:
        f.write(latex_report)
    
    print("\n" + "=" * 60)
    print("Reports Generated Successfully!")
    print("=" * 60)
    print(f"\nText report saved to: {text_path}")
    print(f"LaTeX report saved to: {latex_path}")
    print("\nTo compile the LaTeX report, run:")
    print(f"  pdflatex {latex_path}")
    print("\nNote: Make sure to run pdflatex from the project root directory")


if __name__ == "__main__":
    main()


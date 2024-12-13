import matplotlib.pyplot as plt
import os

def load_labels(file_path):
    """Load labels from a file."""
    with open(file_path, 'r') as file:
        content = file.read()
    return list(map(int, content.split()))

def match_files(ground_truth_folder, test_folder):
    """Match test files with corresponding ground truth files."""
    test_files = [f for f in os.listdir(test_folder) if f.endswith('.txt')]
    matches = []
    for test_file in test_files:
        # Add `.regions` to find the corresponding ground truth file
        ground_truth_file = os.path.join(ground_truth_folder, test_file.replace('.txt', '.regions.txt'))
        test_file_path = os.path.join(test_folder, test_file)
        if os.path.exists(ground_truth_file):
            matches.append((ground_truth_file, test_file_path))
        else:
            print(f"Warning: No matching ground truth file for {test_file} in {ground_truth_folder}")
    return matches


def calculate_accuracy(ground_truth, predictions):
    """Calculate percentage of correct predictions for general accuracy."""
    correct = sum(gt == pred for gt, pred in zip(ground_truth, predictions))
    return (correct / len(ground_truth)) * 100

def calculate_metrics(ground_truth, predictions, num_labels=9):
    """Calculate TP, FP, FN, and TN for each label (0 to num_labels-1)."""
    tp = [0] * num_labels  # True Positives
    fp = [0] * num_labels  # False Positives
    fn = [0] * num_labels  # False Negatives
    tn = [0] * num_labels  # True Negatives

    for gt, pred in zip(ground_truth, predictions):
        for label in range(1,num_labels): # Skipping 'background' predictions as we want to use it as the negative value.
            # True Positive: both predicted and actual are the label
            if pred == label and gt == label:
                tp[label] += 1
            # False Positive: predicted is the label but actual is not
            elif pred == label and gt != label:
                fp[label] += 1
            # False Negative: predicted is not the label but actual is
            elif pred != label and gt == label:
                fn[label] += 1
            # True Negative: neither predicted nor actual is the label
            elif pred != label and gt != label:
                tn[label] += 1

    # Calculate accuracy, precision, recall
    label_metrics = []
    for i in range(num_labels):
        # Avoid division by zero
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0
        recall = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0

        # Append the metrics for each label
        label_metrics.append({
            'TP': tp[i],
            'FP': fp[i],
            'FN': fn[i],
            'TN': tn[i],
            'Precision': precision,
            'Recall': recall,
        })

    # Calculate the general metrics (TP, FP, FN, TN)
    total_tp = sum(tp)
    total_fp = sum(fp)
    total_fn = sum(fn)
    total_tn = sum(tn)

    general_metrics = {
        'Total TP': total_tp,
        'Total FP': total_fp,
        'Total FN': total_fn,
        'Total TN': total_tn,
        'Total Precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
        'Total Recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
    }

    return label_metrics, general_metrics

def compare_folders(ground_truth_folder, test_folders, num_labels=9):
    """Compare test folder files against ground truth files."""
    results = {}

    for test_folder in test_folders:
        general_accuracies = []
        all_label_metrics = []
        matches = match_files(ground_truth_folder, test_folder)

        # Process each file in the test folder
        for ground_truth_file, test_file in matches:
            ground_truth_labels = load_labels(ground_truth_file)
            test_labels = load_labels(test_file)

            min_length = min(len(ground_truth_labels), len(test_labels))
            ground_truth_labels = ground_truth_labels[:min_length]
            test_labels = test_labels[:min_length]

            # Calculate general accuracy for the file pair
            accuracy = calculate_accuracy(ground_truth_labels, test_labels)
            general_accuracies.append(accuracy)

            # Calculate label metrics (TP, FP, FN, TN) for the file pair
            label_metrics, general_metrics = calculate_metrics(ground_truth_labels, test_labels, num_labels)

            all_label_metrics.append(label_metrics)

        # Average general accuracy for the folder
        if general_accuracies:
            folder_accuracy = sum(general_accuracies) / len(general_accuracies)
        else:
            folder_accuracy = None
            print(f"No valid comparisons made in {test_folder}")

        results[test_folder] = {
            'total_accuracy': folder_accuracy,
            'label_metrics': all_label_metrics,
            'general_metrics': general_metrics
        }

    return results

if __name__ == "__main__":
    ground_truth_folder = "../dataset/test/labels"
    test_folders = ["../output_labels/noisy", "../output_labels/blurred", "../output_labels/sharpened","../output_labels/baseline"]
    
    # Perform comparison
    results = compare_folders(ground_truth_folder, test_folders)

    test_number = 0
    # Display general and label metrics
    for test_folder, result in results.items():
        print(f"\n{test_folder} :")
        print(f"  Total accuracy: {result['total_accuracy']:.2f}%")
        print(f"  General Metrics: {result['general_metrics']}")
        
        for label in range(9):
            label_metrics = result['label_metrics'][test_number][label]
            
            print(f"  Label {label} metrics:")
            print(f"    TP: {label_metrics['TP']}")
            print(f"    FP: {label_metrics['FP']}")
            print(f"    FN: {label_metrics['FN']}")
            print(f"    TN: {label_metrics['TN']}")
            print(f"    Precision: {label_metrics['Precision']:.2f}")
            print(f"    Recall: {label_metrics['Recall']:.2f}")
            print(f"    F1 Score: {label_metrics['F1 Score']:.2f}")


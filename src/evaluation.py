import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, top_k_accuracy_score
)
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime

class ModelEvaluator:
    """
    Comprehensive model evaluation class for CNN classification.
    """
    
    def __init__(self, model, class_to_idx: Dict[str, int], config):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained PyTorch model
            class_to_idx: Dictionary mapping class names to indices
            config: Configuration object
        """
        self.model = model
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.config = config
        self.num_classes = len(class_to_idx)
        
        # Results storage
        self.results = {}
        
    def evaluate(self, test_loader, device: torch.device) -> Dict:
        """
        Comprehensive evaluation of the model.
        
        Args:
            test_loader: Test data loader
            device: Device to run evaluation on
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        logging.info("Starting comprehensive model evaluation")
        
        # Get predictions
        predictions, targets, probabilities = self._get_predictions(test_loader, device)
        
        # Calculate metrics
        metrics = self._calculate_metrics(targets, predictions, probabilities)
        
        # Generate confusion matrix
        cm = self._generate_confusion_matrix(targets, predictions)
        
        # Generate classification report
        class_report = self._generate_classification_report(targets, predictions)
        
        # Store results
        self.results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': predictions.tolist(),
            'targets': targets.tolist(),
            'probabilities': probabilities.tolist(),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save results if configured
        if self.config.get('evaluation.save_metrics', True):
            self._save_results()
        
        logging.info("Evaluation completed")
        return self.results
    
    def _get_predictions(self, data_loader, device: torch.device) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions on the dataset."""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(device), targets.to(device)
                
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)
    
    def _calculate_metrics(self, targets: np.ndarray, predictions: np.ndarray, 
                          probabilities: np.ndarray) -> Dict:
        """Calculate comprehensive evaluation metrics."""
        metrics = {}
        
        # Basic accuracy metrics
        metrics['accuracy'] = accuracy_score(targets, predictions)
        metrics['top_5_accuracy'] = top_k_accuracy_score(targets, probabilities, k=5)
        
        # Precision, Recall, F1 (macro, weighted, and per class)
        metrics['precision_macro'] = precision_score(targets, predictions, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(targets, predictions, average='weighted', zero_division=0)
        
        metrics['recall_macro'] = recall_score(targets, predictions, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(targets, predictions, average='weighted', zero_division=0)
        
        metrics['f1_macro'] = f1_score(targets, predictions, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        precision_per_class = precision_score(targets, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(targets, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(targets, predictions, average=None, zero_division=0)
        
        metrics['precision_per_class'] = precision_per_class.tolist()
        metrics['recall_per_class'] = recall_per_class.tolist()
        metrics['f1_per_class'] = f1_per_class.tolist()
        
        # Class distribution
        unique_targets, counts_targets = np.unique(targets, return_counts=True)
        unique_preds, counts_preds = np.unique(predictions, return_counts=True)
        
        metrics['class_distribution_true'] = dict(zip(unique_targets.tolist(), counts_targets.tolist()))
        metrics['class_distribution_pred'] = dict(zip(unique_preds.tolist(), counts_preds.tolist()))
        
        # Error analysis
        metrics['total_samples'] = len(targets)
        metrics['correct_predictions'] = np.sum(targets == predictions)
        metrics['incorrect_predictions'] = np.sum(targets != predictions)
        metrics['error_rate'] = metrics['incorrect_predictions'] / metrics['total_samples']
        
        # Log key metrics
        logging.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logging.info(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
        logging.info(f"Macro F1 Score: {metrics['f1_macro']:.4f}")
        logging.info(f"Weighted F1 Score: {metrics['f1_weighted']:.4f}")
        logging.info(f"Error Rate: {metrics['error_rate']:.4f}")
        
        return metrics
    
    def _generate_confusion_matrix(self, targets: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        """Generate confusion matrix."""
        cm = confusion_matrix(targets, predictions)
        return cm
    
    def _generate_classification_report(self, targets: np.ndarray, predictions: np.ndarray) -> Dict:
        """Generate detailed classification report."""
        target_names = [self.idx_to_class[i] for i in range(self.num_classes)]
        report = classification_report(
            targets, predictions, 
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
        return report
    
    def _save_results(self) -> None:
        """Save evaluation results to file."""
        metrics_file = self.config.get('evaluation.metrics_file', 'results/metrics.json')
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        # Convert results to JSON-serializable format
        json_results = convert_numpy(self.results)
        
        # Save results
        with open(metrics_file, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        logging.info(f"Evaluation results saved to {metrics_file}")
    
    def get_top_misclassified_samples(self, data_loader, device: torch.device, 
                                    top_k: int = 10) -> List[Dict]:
        """
        Get the top misclassified samples with highest confidence.
        
        Args:
            data_loader: Data loader
            device: Device to run on
            top_k: Number of top misclassifications to return
            
        Returns:
            List of dictionaries containing misclassification info
        """
        self.model.eval()
        misclassifications = []
        
        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(device), targets.to(device)
                
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_classes = torch.argmax(probabilities, dim=1)
                
                # Find misclassifications
                incorrect_mask = predicted_classes != targets
                
                if incorrect_mask.any():
                    incorrect_indices = torch.where(incorrect_mask)[0]
                    
                    for idx in incorrect_indices:
                        true_class = targets[idx].item()
                        pred_class = predicted_classes[idx].item()
                        confidence = probabilities[idx, pred_class].item()
                        
                        misclassifications.append({
                            'batch_idx': batch_idx,
                            'sample_idx': idx.item(),
                            'true_class': self.idx_to_class[true_class],
                            'predicted_class': self.idx_to_class[pred_class],
                            'confidence': confidence,
                            'true_class_idx': true_class,
                            'predicted_class_idx': pred_class
                        })
        
        # Sort by confidence (highest first) and return top_k
        misclassifications.sort(key=lambda x: x['confidence'], reverse=True)
        return misclassifications[:top_k]
    
    def get_class_performance_summary(self) -> Dict:
        """
        Get a summary of performance per class.
        
        Returns:
            Dictionary with class-wise performance summary
        """
        if not self.results:
            logging.warning("No evaluation results available. Run evaluate() first.")
            return {}
        
        metrics = self.results['metrics']
        
        class_summary = {}
        for i in range(self.num_classes):
            class_name = self.idx_to_class[i]
            
            class_summary[class_name] = {
                'precision': metrics['precision_per_class'][i],
                'recall': metrics['recall_per_class'][i],
                'f1_score': metrics['f1_per_class'][i],
                'true_samples': metrics['class_distribution_true'].get(i, 0),
                'predicted_samples': metrics['class_distribution_pred'].get(i, 0)
            }
        
        # Sort by F1 score
        sorted_classes = sorted(class_summary.items(), key=lambda x: x[1]['f1_score'], reverse=True)
        
        return {
            'class_performance': dict(sorted_classes),
            'best_performing_class': sorted_classes[0] if sorted_classes else None,
            'worst_performing_class': sorted_classes[-1] if sorted_classes else None,
            'average_precision': np.mean(metrics['precision_per_class']),
            'average_recall': np.mean(metrics['recall_per_class']),
            'average_f1': np.mean(metrics['f1_per_class'])
        }
    
    def print_summary(self) -> None:
        """Print a summary of evaluation results."""
        if not self.results:
            logging.warning("No evaluation results available. Run evaluate() first.")
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print(f"\nOverall Performance:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
        print(f"  Error Rate: {metrics['error_rate']:.4f}")
        
        print(f"\nMacro-Averaged Metrics:")
        print(f"  Precision: {metrics['precision_macro']:.4f}")
        print(f"  Recall: {metrics['recall_macro']:.4f}")
        print(f"  F1-Score: {metrics['f1_macro']:.4f}")
        
        print(f"\nWeighted-Averaged Metrics:")
        print(f"  Precision: {metrics['precision_weighted']:.4f}")
        print(f"  Recall: {metrics['recall_weighted']:.4f}")
        print(f"  F1-Score: {metrics['f1_weighted']:.4f}")
        
        print(f"\nDataset Statistics:")
        print(f"  Total Samples: {metrics['total_samples']}")
        print(f"  Correct Predictions: {metrics['correct_predictions']}")
        print(f"  Incorrect Predictions: {metrics['incorrect_predictions']}")
        
        # Class performance summary
        class_summary = self.get_class_performance_summary()
        if class_summary:
            print(f"\nClass Performance Summary:")
            print(f"  Average Precision: {class_summary['average_precision']:.4f}")
            print(f"  Average Recall: {class_summary['average_recall']:.4f}")
            print(f"  Average F1-Score: {class_summary['average_f1']:.4f}")
            
            if class_summary['best_performing_class']:
                best_class, best_metrics = class_summary['best_performing_class']
                print(f"  Best Performing Class: {best_class} (F1: {best_metrics['f1_score']:.4f})")
            
            if class_summary['worst_performing_class']:
                worst_class, worst_metrics = class_summary['worst_performing_class']
                print(f"  Worst Performing Class: {worst_class} (F1: {worst_metrics['f1_score']:.4f})")
        
        print("="*60)

def compare_models(model_results: Dict[str, Dict], save_path: Optional[str] = None) -> Dict:
    """
    Compare multiple models and return comparison metrics.
    
    Args:
        model_results: Dictionary mapping model names to their evaluation results
        save_path: Optional path to save comparison results
        
    Returns:
        Dictionary containing comparison metrics
    """
    if not model_results:
        logging.warning("No model results provided for comparison")
        return {}
    
    comparison = {}
    
    # Extract key metrics for each model
    for model_name, results in model_results.items():
        if 'metrics' in results:
            metrics = results['metrics']
            comparison[model_name] = {
                'accuracy': metrics.get('accuracy', 0),
                'top_5_accuracy': metrics.get('top_5_accuracy', 0),
                'f1_macro': metrics.get('f1_macro', 0),
                'f1_weighted': metrics.get('f1_weighted', 0),
                'precision_macro': metrics.get('precision_macro', 0),
                'recall_macro': metrics.get('recall_macro', 0),
                'error_rate': metrics.get('error_rate', 0)
            }
    
    # Find best model for each metric
    best_models = {}
    for metric in ['accuracy', 'top_5_accuracy', 'f1_macro', 'f1_weighted']:
        best_model = max(comparison.items(), key=lambda x: x[1].get(metric, 0))
        best_models[f'best_{metric}'] = {
            'model': best_model[0],
            'value': best_model[1][metric]
        }
    
    comparison['best_models'] = best_models
    
    # Save comparison if path provided
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(comparison, f, indent=2)
        logging.info(f"Model comparison saved to {save_path}")
    
    return comparison

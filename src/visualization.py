import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import os
import logging
from sklearn.metrics import confusion_matrix
import json

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Visualizer:
    """
    Comprehensive visualization class for CNN training and evaluation results.
    """
    
    def __init__(self, config, save_dir: str = "results/plots"):
        """
        Initialize the visualizer.
        
        Args:
            config: Configuration object
            save_dir: Directory to save plots
        """
        self.config = config
        self.save_dir = save_dir
        self.save_plots = config.get('visualization.save_plots', True)
        
        # Create save directory
        if self.save_plots:
            os.makedirs(save_dir, exist_ok=True)
        
        # Set up matplotlib parameters
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
        logging.info(f"Visualizer initialized. Plots will be saved to: {save_dir}")
    
    def plot_training_curves(self, history: Dict, save_name: str = "training_curves") -> None:
        """
        Plot training and validation loss and accuracy curves.
        
        Args:
            history: Training history dictionary
            save_name: Name for saving the plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Loss curves
        ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, history['train_accuracy'], 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, history['val_accuracy'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate curve
        if 'learning_rates' in history:
            ax3.plot(epochs, history['learning_rates'], 'g-', linewidth=2)
            ax3.set_title('Learning Rate Schedule')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate')
            ax3.set_yscale('log')
            ax3.grid(True, alpha=0.3)
        
        # Combined metrics
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        line2 = ax4.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        line3 = ax4_twin.plot(epochs, history['train_accuracy'], 'g--', label='Train Acc', linewidth=2)
        line4 = ax4_twin.plot(epochs, history['val_accuracy'], 'm--', label='Val Acc', linewidth=2)
        
        ax4.set_title('Combined Metrics')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='b')
        ax4_twin.set_ylabel('Accuracy (%)', color='g')
        
        # Combine legends
        lines = line1 + line2 + line3 + line4
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='center right')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            save_name: str = "confusion_matrix", 
                            normalize: bool = True,
                            top_k: int = 20) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm: Confusion matrix
            class_names: List of class names
            save_name: Name for saving the plot
            normalize: Whether to normalize the confusion matrix
            top_k: Number of top classes to show (for large number of classes)
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            title = 'Normalized Confusion Matrix'
            fmt = '.2f'
        else:
            title = 'Confusion Matrix'
            fmt = 'd'
        
        # For large number of classes, show only top_k most frequent
        if len(class_names) > top_k:
            # Calculate class frequencies
            class_counts = cm.sum(axis=1)
            top_indices = np.argsort(class_counts)[-top_k:]
            
            cm = cm[np.ix_(top_indices, top_indices)]
            class_names = [class_names[i] for i in top_indices]
            title += f' (Top {top_k} classes)'
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Proportion' if normalize else 'Count'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_class_performance(self, metrics: Dict, class_names: List[str],
                             save_name: str = "class_performance") -> None:
        """
        Plot per-class performance metrics.
        
        Args:
            metrics: Metrics dictionary containing per-class scores
            class_names: List of class names
            save_name: Name for saving the plot
        """
        # Extract per-class metrics
        precision = np.array(metrics['precision_per_class'])
        recall = np.array(metrics['recall_per_class'])
        f1 = np.array(metrics['f1_per_class'])
        
        # Sort classes by F1 score for better visualization
        sort_indices = np.argsort(f1)
        class_names_sorted = [class_names[i] for i in sort_indices]
        precision_sorted = precision[sort_indices]
        recall_sorted = recall[sort_indices]
        f1_sorted = f1[sort_indices]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        
        # Plot 1: Bar chart of metrics
        x = np.arange(len(class_names_sorted))
        width = 0.25
        
        ax1.bar(x - width, precision_sorted, width, label='Precision', alpha=0.8)
        ax1.bar(x, recall_sorted, width, label='Recall', alpha=0.8)
        ax1.bar(x + width, f1_sorted, width, label='F1-Score', alpha=0.8)
        
        ax1.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Classes')
        ax1.set_ylabel('Score')
        ax1.set_xticks(x)
        ax1.set_xticklabels(class_names_sorted, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Plot 2: Distribution of F1 scores
        ax2.hist(f1_sorted, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(f1_sorted.mean(), color='red', linestyle='--', 
                   label=f'Mean: {f1_sorted.mean():.3f}')
        ax2.axvline(np.median(f1_sorted), color='green', linestyle='--',
                   label=f'Median: {np.median(f1_sorted):.3f}')
        
        ax2.set_title('Distribution of F1 Scores Across Classes', fontsize=14, fontweight='bold')
        ax2.set_xlabel('F1 Score')
        ax2.set_ylabel('Number of Classes')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Class performance plot saved to {save_path}")
        
        plt.show()
    
    def plot_sample_predictions(self, data_loader, model, device, class_names: List[str],
                              num_samples: int = 16, save_name: str = "sample_predictions") -> None:
        """
        Plot sample predictions with images.
        
        Args:
            data_loader: Data loader
            model: Trained model
            device: Device to run on
            class_names: List of class names
            num_samples: Number of samples to plot
            save_name: Name for saving the plot
        """
        model.eval()
        
        # Get a batch of data
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        
        # Move to device
        images, labels = images.to(device), labels.to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(images)
            probabilities = F.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
        
        # Move back to CPU for plotting
        images = images.cpu()
        labels = labels.cpu()
        predicted_classes = predicted_classes.cpu()
        confidences = confidences.cpu()
        
        # Denormalize images for display
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        images_denorm = images * std + mean
        images_denorm = torch.clamp(images_denorm, 0, 1)
        
        # Create subplot grid
        cols = 4
        rows = (num_samples + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(min(num_samples, len(images))):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Display image
            img = images_denorm[i].permute(1, 2, 0).numpy()
            ax.imshow(img)
            
            # Get true and predicted labels
            true_label = class_names[labels[i]]
            pred_label = class_names[predicted_classes[i]]
            confidence = confidences[i].item()
            is_correct = labels[i] == predicted_classes[i]
            
            # Set title color based on correctness
            title_color = 'green' if is_correct else 'red'
            
            ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}',
                        color=title_color, fontsize=10)
            ax.axis('off')
        
        # Hide unused subplots
        for i in range(num_samples, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Sample Predictions', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Sample predictions saved to {save_path}")
        
        plt.show()
    
    def plot_hyperparameter_analysis(self, analysis_results: Dict, 
                                   save_name: str = "hyperparameter_analysis") -> None:
        """
        Plot hyperparameter analysis results.
        
        Args:
            analysis_results: Results from hyperparameter analysis
            save_name: Name for saving the plot
        """
        if 'hyperparameter_effects' not in analysis_results:
            logging.warning("No hyperparameter effects found in analysis results")
            return
        
        effects = analysis_results['hyperparameter_effects']
        
        # Create subplots for each hyperparameter
        num_params = len(effects)
        cols = 2
        rows = (num_params + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (param_name, param_effects) in enumerate(effects.items()):
            row = i // cols
            col = i % cols
            ax = axes[row, col]
            
            # Extract data
            values = []
            mean_accuracies = []
            std_accuracies = []
            
            for value, stats in param_effects.items():
                values.append(str(value))
                mean_accuracies.append(stats['mean_accuracy'])
                std_accuracies.append(stats['std_accuracy'])
            
            # Create bar plot with error bars
            bars = ax.bar(range(len(values)), mean_accuracies, 
                         yerr=std_accuracies, capsize=5, alpha=0.7)
            
            # Color bars by performance
            colors = plt.cm.RdYlBu_r(np.array(mean_accuracies))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            ax.set_title(f'Effect of {param_name}', fontsize=12, fontweight='bold')
            ax.set_xlabel(param_name)
            ax.set_ylabel('Mean Validation Accuracy')
            ax.set_xticks(range(len(values)))
            ax.set_xticklabels(values, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(num_params, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.suptitle('Hyperparameter Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Hyperparameter analysis plot saved to {save_path}")
        
        plt.show()
    
    def plot_feature_maps(self, model, data_loader, device, layer_indices: List[int] = [0, 1],
                         save_name: str = "feature_maps") -> None:
        """
        Visualize feature maps from different layers.
        
        Args:
            model: Trained model
            data_loader: Data loader
            device: Device to run on
            layer_indices: Indices of layers to visualize
            save_name: Name for saving the plot
        """
        model.eval()
        
        # Get a sample image
        data_iter = iter(data_loader)
        images, _ = next(data_iter)
        sample_image = images[0:1].to(device)  # Take first image
        
        # Get feature maps
        feature_maps = model.get_feature_maps(sample_image)
        
        # Create subplot for each layer
        num_layers = min(len(layer_indices), len(feature_maps))
        fig, axes = plt.subplots(num_layers, 8, figsize=(20, 3 * num_layers))
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for layer_idx in range(num_layers):
            actual_layer_idx = layer_indices[layer_idx]
            if actual_layer_idx >= len(feature_maps):
                continue
            
            feature_map = feature_maps[actual_layer_idx].squeeze(0)  # Remove batch dimension
            num_channels = min(8, feature_map.size(0))  # Show first 8 channels
            
            for ch in range(num_channels):
                ax = axes[layer_idx, ch]
                
                # Get feature map for this channel
                channel_map = feature_map[ch].detach().cpu().numpy()
                
                # Display as heatmap
                im = ax.imshow(channel_map, cmap='viridis')
                ax.set_title(f'Layer {actual_layer_idx+1}, Ch {ch+1}', fontsize=8)
                ax.axis('off')
                
                # Add colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        plt.suptitle('Feature Maps Visualization', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if self.save_plots:
            save_path = os.path.join(self.save_dir, f"{save_name}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logging.info(f"Feature maps saved to {save_path}")
        
        plt.show()
    
    def create_comprehensive_report(self, history: Dict, evaluation_results: Dict,
                                  class_names: List[str], data_loader, model, device) -> None:
        """
        Create a comprehensive visualization report.
        
        Args:
            history: Training history
            evaluation_results: Evaluation results
            class_names: List of class names
            data_loader: Data loader
            model: Trained model
            device: Device to run on
        """
        logging.info("Creating comprehensive visualization report...")
        
        # 1. Training curves
        if self.config.get('visualization.plot_loss_curves', True):
            self.plot_training_curves(history)
        
        # 2. Confusion matrix
        if self.config.get('visualization.confusion_matrix', True):
            cm = np.array(evaluation_results['confusion_matrix'])
            self.plot_confusion_matrix(cm, class_names)
        
        # 3. Class performance
        if 'metrics' in evaluation_results:
            self.plot_class_performance(evaluation_results['metrics'], class_names)
        
        # 4. Sample predictions
        if self.config.get('visualization.plot_sample_predictions', True):
            num_samples = self.config.get('visualization.num_sample_predictions', 16)
            self.plot_sample_predictions(data_loader, model, device, class_names, num_samples)
        
        # 5. Feature maps
        self.plot_feature_maps(model, data_loader, device)
        
        logging.info("Comprehensive visualization report completed")
    
    def save_metrics_summary(self, history: Dict, evaluation_results: Dict) -> None:
        """
        Save a summary of key metrics as JSON.
        
        Args:
            history: Training history
            evaluation_results: Evaluation results
        """
        summary = {
            'training_summary': {
                'final_train_accuracy': history['train_accuracy'][-1],
                'final_val_accuracy': history['val_accuracy'][-1],
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'best_val_accuracy': max(history['val_accuracy']),
                'total_epochs': len(history['train_loss'])
            },
            'evaluation_summary': evaluation_results['metrics'] if 'metrics' in evaluation_results else {},
            'timestamp': str(np.datetime64('now'))
        }
        
        save_path = os.path.join(self.save_dir, 'metrics_summary.json')
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logging.info(f"Metrics summary saved to {save_path}")

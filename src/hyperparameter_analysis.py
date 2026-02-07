import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, List, Tuple, Any, Optional
import logging
from itertools import product
from datetime import datetime
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

from .model import create_model_from_config
from .trainer import Trainer, DeviceManager
from .data_loader import create_train_val_loaders
from .evaluation import ModelEvaluator

class HyperparameterAnalyzer:
    """
    Comprehensive hyperparameter analysis and optimization tool.
    """
    
    def __init__(self, base_config, results_dir: str = "results/hyperparameter_analysis", exclude_optimal: bool = True):
        """
        Initialize the hyperparameter analyzer.
        
        Args:
            base_config: Base configuration object
            results_dir: Directory to save analysis results
            exclude_optimal: Whether to exclude the current optimal config from experiments
        """
        self.base_config = base_config
        self.results_dir = results_dir
        self.exclude_optimal = exclude_optimal
        self.experiment_results = []
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load hyperparameter ranges from config
        self.hyperparameter_ranges = self._load_hyperparameter_ranges()
        
        logging.info(f"Hyperparameter analyzer initialized")
        logging.info(f"Results will be saved to: {results_dir}")
        if self.exclude_optimal:
            logging.info("Excluding current optimal configuration from experiments")
    
    def _load_hyperparameter_ranges(self) -> Dict[str, List[Any]]:
        """Load hyperparameter ranges from configuration."""
        if not self.base_config.get('hyperparameter_analysis.enabled', False):
            logging.warning("Hyperparameter analysis is disabled in config")
            return {}
        
        ranges = {
            'learning_rate': self.base_config.get('hyperparameter_analysis.learning_rates', [0.001, 0.0001, 0.01]),
            'batch_size': self.base_config.get('hyperparameter_analysis.batch_sizes', [16, 32, 64]),
            'num_conv_blocks': self.base_config.get('hyperparameter_analysis.num_conv_blocks', [2, 3, 4]),
            'conv_filters': self.base_config.get('hyperparameter_analysis.filter_sizes', [[32, 64], [64, 128], [128, 256]])
        }
        
        logging.info("Hyperparameter ranges loaded:")
        for param, values in ranges.items():
            logging.info(f"  {param}: {values}")
        
        return ranges
    
    def _is_optimal_config(self, combination: List) -> bool:
        """
        Check if a combination matches the current optimal configuration.
        
        Args:
            combination: Parameter combination to check
            
        Returns:
            True if matches optimal config, False otherwise
        """
        if not self.exclude_optimal:
            return False
            
        # Get optimal values from base config
        optimal_lr = self.base_config.get('training.learning_rate', 0.001)
        optimal_batch_size = self.base_config.get('dataset.batch_size', 32)
        optimal_conv_blocks = self.base_config.get('model.num_conv_blocks', 3)
        optimal_conv_filters = self.base_config.get('model.conv_filters', [64, 128, 256])
        optimal_dropout = self.base_config.get('model.dropout_rate', 0.3)
        
        # Check if combination matches all optimal values
        for j, param_name in enumerate(self.original_param_names):
            if param_name == 'learning_rate' and combination[j] != optimal_lr:
                return False
            elif param_name == 'batch_size' and combination[j] != optimal_batch_size:
                return False
            elif param_name == 'num_conv_blocks' and combination[j] != optimal_conv_blocks:
                return False
            elif param_name == 'conv_filters' and combination[j] != optimal_conv_filters:
                return False
            elif param_name == 'dropout_rate' and combination[j] != optimal_dropout:
                return False
        
        return True
    
    def generate_experiment_configs(self) -> List[Dict]:
        """
        Generate all experiment configurations based on hyperparameter ranges.
        
        Returns:
            List of configuration dictionaries for each experiment
        """
        if not self.hyperparameter_ranges:
            logging.warning("No hyperparameter ranges defined")
            return []
        
        # Generate all combinations
        param_names = list(self.hyperparameter_ranges.keys())
        param_values = list(self.hyperparameter_ranges.values())
        self.original_param_names = param_names.copy()  # Store original
        
        # Handle conv_filters specially since it's a list of lists
        if 'conv_filters' in param_names:
            # Store the original index
            original_conv_filters_idx = param_names.index('conv_filters')
            
            # Remove conv_filters from regular grid search
            param_names = [p for p in param_names if p != 'conv_filters']
            param_values = [v for p, v in zip(param_names, param_values) if p != 'conv_filters']
            
            # Generate base combinations
            base_combinations = list(product(*param_values))
            
            # For each base combination, try each conv_filter configuration
            all_combinations = []
            for base_combo in base_combinations:
                for filter_config in self.hyperparameter_ranges['conv_filters']:
                    # Create full parameter combination
                    full_combo = list(base_combo)
                    full_combo.insert(original_conv_filters_idx, filter_config)
                    
                    # Validate that conv_filters length matches num_conv_blocks
                    temp_config = copy.deepcopy(self.base_config.config)
                    for j, param_name in enumerate(self.original_param_names):
                        if param_name == 'conv_filters':
                            temp_config['model']['conv_filters'] = full_combo[j]
                        elif param_name == 'num_conv_blocks':
                            temp_config['model']['num_conv_blocks'] = full_combo[j]
                    
                    if len(temp_config['model']['conv_filters']) == temp_config['model']['num_conv_blocks']:
                        # Check if this is the optimal configuration and should be excluded
                        if not self._is_optimal_config(full_combo):
                            all_combinations.append(full_combo)
                        else:
                            logging.info(f"Excluding optimal configuration: {full_combo}")
            
            combinations = all_combinations
        else:
            combinations = list(product(*param_values))
            # Filter out optimal configuration if conv_filters is not in ranges
            if self.exclude_optimal:
                filtered_combinations = []
                for combo in combinations:
                    if not self._is_optimal_config(list(combo)):
                        filtered_combinations.append(combo)
                combinations = filtered_combinations
        
        # Convert to configuration dictionaries
        experiment_configs = []
        for i, combination in enumerate(combinations):
            config = copy.deepcopy(self.base_config.config)
            
            # Apply hyperparameter values - override base config
            for j, param_name in enumerate(self.original_param_names):
                if param_name == 'conv_filters':
                    # Special handling for conv_filters
                    config['model']['conv_filters'] = combination[j]
                elif param_name == 'learning_rate':
                    config['training']['learning_rate'] = combination[j]
                elif param_name == 'batch_size':
                    config['dataset']['batch_size'] = combination[j]
                elif param_name == 'num_conv_blocks':
                    config['model']['num_conv_blocks'] = combination[j]
                elif param_name == 'dropout_rate':
                    config['model']['dropout_rate'] = combination[j]
                # Add other parameters as needed
            
            # Add experiment metadata
            config['experiment'] = {
                'id': i,
                'hyperparameters': {
                    'learning_rate': config['training'].get('learning_rate'),
                    'batch_size': config['dataset'].get('batch_size'),
                    'num_conv_blocks': config['model'].get('num_conv_blocks'),
                    'conv_filters': config['model'].get('conv_filters'),
                    'dropout_rate': config['model'].get('dropout_rate')
                }
            }
            
            experiment_configs.append(config)
        
        logging.info(f"Generated {len(experiment_configs)} experiment configurations")
        return experiment_configs
    
    def run_single_experiment(self, experiment_config: Dict, experiment_id: int) -> Dict:
        """
        Run a single hyperparameter experiment.
        
        Args:
            experiment_config: Configuration for this experiment
            experiment_id: ID of the experiment
            
        Returns:
            Dictionary containing experiment results
        """
        logging.info(f"Running experiment {experiment_id}: {experiment_config['experiment']['hyperparameters']}")
        
        try:
            # Create temporary config object
            from .config_manager import ConfigManager
            temp_config = ConfigManager.__new__(ConfigManager)
            temp_config.config = experiment_config
            
            # Setup device
            device = DeviceManager.get_device(temp_config)
            
            # Create data loaders
            train_loader, val_loader, class_to_idx = create_train_val_loaders(temp_config)
            
            # Create model
            model = create_model_from_config(temp_config)
            
            # Create trainer
            trainer = Trainer(model, temp_config, device)
            
            # Train model (reduced epochs for hyperparameter search)
            epochs = min(temp_config.get('training.epochs', 100), 30)  # Cap at 30 for efficiency
            history = trainer.train(train_loader, val_loader, epochs=epochs)
            
            # Evaluate model
            evaluator = ModelEvaluator(model, class_to_idx, temp_config)
            results = evaluator.evaluate(val_loader, device)
            
            # Compile experiment results
            experiment_results = {
                'experiment_id': experiment_id,
                'hyperparameters': experiment_config['experiment']['hyperparameters'],
                'final_train_accuracy': history['train_accuracy'][-1],
                'final_val_accuracy': history['val_accuracy'][-1],
                'best_val_accuracy': trainer.best_val_accuracy,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1],
                'training_history': history,
                'evaluation_metrics': results['metrics'],
                'total_parameters': model.count_parameters()[0],
                'training_time': len(history['train_loss']),  # Approximate
                'status': 'completed'
            }
            
            logging.info(f"Experiment {experiment_id} completed. Best val accuracy: {trainer.best_val_accuracy:.4f}")
            return experiment_results
            
        except Exception as e:
            logging.error(f"Experiment {experiment_id} failed: {str(e)}")
            return {
                'experiment_id': experiment_id,
                'hyperparameters': experiment_config['experiment']['hyperparameters'],
                'status': 'failed',
                'error': str(e)
            }
    
    def run_grid_search(self, max_experiments: Optional[int] = None) -> List[Dict]:
        """
        Run a complete grid search over all hyperparameter combinations.
        
        Args:
            max_experiments: Maximum number of experiments to run (None for all)
            
        Returns:
            List of experiment results
        """
        experiment_configs = self.generate_experiment_configs()
        
        if max_experiments:
            experiment_configs = experiment_configs[:max_experiments]
            logging.info(f"Limiting to first {max_experiments} experiments")
        
        logging.info(f"Starting grid search with {len(experiment_configs)} experiments")
        
        # Run experiments sequentially (can be parallelized with ThreadPoolExecutor if needed)
        for i, config in enumerate(experiment_configs):
            result = self.run_single_experiment(config, i)
            self.experiment_results.append(result)
            
            # Save intermediate results
            self._save_experiment_results()
        
        # Final save
        self._save_experiment_results()
        
        logging.info(f"Grid search completed. Total experiments: {len(self.experiment_results)}")
        return self.experiment_results
    
    def analyze_results(self) -> Dict:
        """
        Analyze the results of hyperparameter experiments.
        
        Returns:
            Dictionary containing analysis results
        """
        if not self.experiment_results:
            logging.warning("No experiment results to analyze")
            return {}
        
        # Filter successful experiments
        successful_results = [r for r in self.experiment_results if r.get('status') == 'completed']
        
        if not successful_results:
            logging.warning("No successful experiments to analyze")
            return {}
        
        analysis = {
            'total_experiments': len(self.experiment_results),
            'successful_experiments': len(successful_results),
            'failed_experiments': len(self.experiment_results) - len(successful_results)
        }
        
        # Find best experiments for different metrics
        best_val_acc = max(successful_results, key=lambda x: x['best_val_accuracy'])
        best_train_acc = max(successful_results, key=lambda x: x['final_train_accuracy'])
        best_loss = min(successful_results, key=lambda x: x['final_val_loss'])
        
        analysis['best_experiments'] = {
            'best_validation_accuracy': {
                'experiment_id': best_val_acc['experiment_id'],
                'hyperparameters': best_val_acc['hyperparameters'],
                'accuracy': best_val_acc['best_val_accuracy']
            },
            'best_training_accuracy': {
                'experiment_id': best_train_acc['experiment_id'],
                'hyperparameters': best_train_acc['hyperparameters'],
                'accuracy': best_train_acc['final_train_accuracy']
            },
            'best_validation_loss': {
                'experiment_id': best_loss['experiment_id'],
                'hyperparameters': best_loss['hyperparameters'],
                'loss': best_loss['final_val_loss']
            }
        }
        
        # Analyze individual hyperparameter effects
        analysis['hyperparameter_effects'] = self._analyze_hyperparameter_effects(successful_results)
        
        # Create summary table
        analysis['summary_table'] = self._create_summary_table(successful_results)
        
        # Save analysis
        self._save_analysis(analysis)
        
        return analysis
    
    def _analyze_hyperparameter_effects(self, results: List[Dict]) -> Dict:
        """Analyze the effect of individual hyperparameters."""
        effects = {}
        
        # Get all hyperparameter names
        hyperparams = set()
        for result in results:
            hyperparams.update(result['hyperparameters'].keys())
        
        for param in hyperparams:
            param_results = {}
            
            for result in results:
                param_value = result['hyperparameters'].get(param)
                # Convert list to tuple for hashability
                if isinstance(param_value, list):
                    param_value = tuple(param_value)
                if param_value not in param_results:
                    param_results[param_value] = []
                param_results[param_value].append(result['best_val_accuracy'])
            
            # Calculate statistics for each parameter value
            effects[param] = {}
            for value, accuracies in param_results.items():
                effects[param][str(value)] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies),
                    'num_experiments': len(accuracies)
                }
        
        return effects
    
    def _create_summary_table(self, results: List[Dict]) -> List[Dict]:
        """Create a summary table of all experiments."""
        table = []
        
        for result in results:
            row = {
                'experiment_id': result['experiment_id'],
                'best_val_accuracy': result['best_val_accuracy'],
                'final_train_accuracy': result['final_train_accuracy'],
                'final_val_loss': result['final_val_loss'],
                'total_parameters': result['total_parameters']
            }
            
            # Add hyperparameters
            row.update(result['hyperparameters'])
            table.append(row)
        
        # Sort by validation accuracy
        table.sort(key=lambda x: x['best_val_accuracy'], reverse=True)
        
        return table
    
    def _save_experiment_results(self) -> None:
        """Save experiment results to file."""
        results_file = os.path.join(self.results_dir, 'experiment_results.json')
        
        with open(results_file, 'w') as f:
            json.dump(self.experiment_results, f, indent=2, default=str)
        
        logging.info(f"Experiment results saved to {results_file}")
    
    def _save_analysis(self, analysis: Dict) -> None:
        """Save analysis results to file."""
        analysis_file = os.path.join(self.results_dir, 'hyperparameter_analysis.json')
        
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logging.info(f"Hyperparameter analysis saved to {analysis_file}")
    
    def get_best_hyperparameters(self, metric: str = 'best_val_accuracy') -> Dict:
        """
        Get the best hyperparameters based on a specific metric.
        
        Args:
            metric: Metric to optimize ('best_val_accuracy', 'final_train_accuracy', 'final_val_loss')
            
        Returns:
            Dictionary of best hyperparameters
        """
        if not self.experiment_results:
            logging.warning("No experiment results available")
            return {}
        
        successful_results = [r for r in self.experiment_results if r.get('status') == 'completed']
        
        if not successful_results:
            logging.warning("No successful experiments available")
            return {}
        
        if metric == 'final_val_loss':
            best_result = min(successful_results, key=lambda x: x[metric])
        else:
            best_result = max(successful_results, key=lambda x: x[metric])
        
        logging.info(f"Best hyperparameters based on {metric}:")
        for param, value in best_result['hyperparameters'].items():
            logging.info(f"  {param}: {value}")
        
        return best_result['hyperparameters']
    
    def create_hyperparameter_report(self) -> str:
        """
        Create a comprehensive hyperparameter analysis report.
        
        Returns:
            Report as a string
        """
        if not self.experiment_results:
            return "No experiment results available for reporting."
        
        analysis = self.analyze_results()
        
        report = []
        report.append("=" * 80)
        report.append("HYPERPARAMETER ANALYSIS REPORT")
        report.append("=" * 80)
        
        # Summary statistics
        report.append(f"\nSUMMARY:")
        report.append(f"Total Experiments: {analysis['total_experiments']}")
        report.append(f"Successful Experiments: {analysis['successful_experiments']}")
        report.append(f"Failed Experiments: {analysis['failed_experiments']}")
        
        # Best experiments
        best_exps = analysis['best_experiments']
        report.append(f"\nBEST EXPERIMENTS:")
        
        for metric, exp_info in best_exps.items():
            report.append(f"\n{metric.upper().replace('_', ' ')}:")
            report.append(f"  Experiment ID: {exp_info['experiment_id']}")
            report.append(f"  Value: {exp_info.get('accuracy', exp_info.get('loss', 'N/A'))}")
            report.append(f"  Hyperparameters:")
            for param, value in exp_info['hyperparameters'].items():
                report.append(f"    {param}: {value}")
        
        # Hyperparameter effects
        report.append(f"\nHYPERPARAMETER EFFECTS:")
        for param, effects in analysis['hyperparameter_effects'].items():
            report.append(f"\n{param.upper()}:")
            
            # Sort by mean accuracy
            sorted_effects = sorted(effects.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True)
            
            for value, stats in sorted_effects:
                report.append(f"  {value}: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} "
                             f"(n={stats['num_experiments']})")
        
        # Top 5 experiments
        report.append(f"\nTOP 5 EXPERIMENTS (by validation accuracy):")
        for i, exp in enumerate(analysis['summary_table'][:5]):
            report.append(f"\n{i+1}. Experiment {exp['experiment_id']}:")
            report.append(f"   Validation Accuracy: {exp['best_val_accuracy']:.4f}")
            report.append(f"   Training Accuracy: {exp['final_train_accuracy']:.4f}")
            report.append(f"   Validation Loss: {exp['final_val_loss']:.4f}")
            report.append(f"   Parameters: {exp['total_parameters']:,}")
            
            # Show key hyperparameters
            key_params = ['learning_rate', 'batch_size', 'num_conv_blocks']
            for param in key_params:
                if param in exp:
                    report.append(f"   {param}: {exp[param]}")
        
        report.append("\n" + "=" * 80)
        
        return "\n".join(report)

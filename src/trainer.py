import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import time
import os
from tqdm import tqdm
import json

class DeviceManager:
    """Manages device selection with support for AMD Radeon GPUs."""
    
    @staticmethod
    def get_device(config) -> torch.device:
        """
        Get the appropriate device for training.
        
        Args:
            config: Configuration object
            
        Returns:
            PyTorch device object
        """
        use_gpu = config.get('device.use_gpu', True)
        gpu_type = config.get('device.gpu_type', 'amd').lower()
        
        if not use_gpu:
            logging.info("GPU disabled in configuration. Using CPU.")
            return torch.device('cpu')
        
        # Try DirectML first for AMD GPUs on Windows
        if gpu_type == 'amd':
            try:
                import torch_directml
                device = torch_directml.device()
                logging.info(f"Using DirectML device for AMD GPU: {device}")
                return device
            except ImportError:
                logging.warning("torch-directml not available. Falling back to CUDA/CPU.")
            except Exception as e:
                logging.warning(f"DirectML initialization failed: {e}. Falling back to CUDA/CPU.")
        
        # Check for CUDA availability (works with AMD ROCm)
        if torch.cuda.is_available():
            device = torch.device('cuda')
            gpu_count = torch.cuda.device_count()
            current_device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(current_device)
            
            logging.info(f"CUDA available. Using device: {device}")
            logging.info(f"GPU Name: {device_name}")
            logging.info(f"GPU Count: {gpu_count}")
            logging.info(f"Current GPU: {current_device}")
            
            # Check if it's an AMD GPU
            if 'amd' in device_name.lower() or 'radeon' in device_name.lower():
                logging.info("AMD Radeon GPU detected via CUDA/ROCm")
            
            return device
        
        else:
            logging.warning("No GPU acceleration available. Falling back to CPU.")
            logging.info("Note: For AMD Radeon GPU support, ensure DirectML or ROCm is properly installed.")
            return torch.device('cpu')
    
    @staticmethod
    def get_device_info(device: torch.device) -> Dict[str, str]:
        """
        Get information about the current device.
        
        Args:
            device: PyTorch device
            
        Returns:
            Dictionary with device information
        """
        info = {
            'device_type': str(device),
            'device_available': str(torch.cuda.is_available() if device.type == 'cuda' else True)
        }
        
        if device.type == 'cuda' and torch.cuda.is_available():
            info.update({
                'gpu_name': torch.cuda.get_device_name(device),
                'gpu_memory': f"{torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB",
                'gpu_count': str(torch.cuda.device_count()),
                'cuda_version': torch.version.cuda
            })
        
        return info

class Trainer:
    """
    Training class for the CompactCNN model with support for AMD Radeon GPUs.
    """
    
    def __init__(self, model: nn.Module, config, device: Optional[torch.device] = None):
        """
        Initialize the trainer.
        
        Args:
            model: PyTorch model to train
            config: Configuration object
            device: Device to use for training (auto-detected if None)
        """
        self.model = model
        self.config = config
        self.device = device or DeviceManager.get_device(config)
        
        # Move model to device
        self.model.to(self.device)
        
        # Setup training components
        self.criterion = self._setup_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler()
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_accuracy = 0.0
        self.best_model_state = None
        
        logging.info(f"Trainer initialized on device: {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    def _setup_criterion(self) -> nn.Module:
        """Setup the loss function."""
        loss_name = self.config.get('training.loss_function', 'cross_entropy')
        
        if loss_name.lower() == 'cross_entropy':
            criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_name}")
        
        logging.info(f"Loss function: {loss_name}")
        return criterion
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup the optimizer."""
        optimizer_name = self.config.get('training.optimizer', 'adam')
        learning_rate = self.config.get('training.learning_rate', 0.001)
        weight_decay = self.config.get('training.weight_decay', 1e-4)
        
        if optimizer_name.lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_name.lower() == 'sgd':
            momentum = self.config.get('training.momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:  # Default to Adam
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        logging.info(f"Optimizer: {optimizer_name} (lr={learning_rate}, weight_decay={weight_decay})")
        return optimizer
    
    def _setup_scheduler(self) -> Optional[object]:
        """Setup the learning rate scheduler."""
        scheduler_config = self.config.get('training.scheduler', {})
        
        if not scheduler_config:
            return None
        
        scheduler_name = scheduler_config.get('name', 'step_lr')
        
        if scheduler_name == 'step_lr':
            scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_name == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=scheduler_config.get('gamma', 0.1),
                patience=scheduler_config.get('patience', 10)
            )
        elif scheduler_name == 'cosine_annealing':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', 50),
                eta_min=scheduler_config.get('eta_min', 0.00001)
            )
        else:
            logging.warning(f"Unsupported scheduler: {scheduler_name}")
            return None
        
        logging.info(f"Scheduler: {scheduler_name}")
        return scheduler
    
    def train_epoch(self, train_loader) -> Tuple[float, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        
        # Pre-allocate GPU memory to avoid initialization delays
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        progress_bar = tqdm(train_loader, desc="Training", leave=False, mininterval=1.0)
        
        # Warm-up: preload first batch to avoid initial delay
        data_iter = iter(train_loader)
        try:
            first_batch = next(data_iter)
            del first_batch  # Don't process, just warm up the pipeline
        except StopIteration:
            pass
        
        for batch_idx, (data, targets) in enumerate(progress_bar):
            # Move data to device (non-blocking if possible)
            data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update progress bar with more detailed information
            if batch_idx % 5 == 0:  # Update more frequently
                current_loss = running_loss / (batch_idx + 1)
                current_acc = 100. * correct / total
                current_lr = self.optimizer.param_groups[0]['lr']
                batch_time = (time.time() - epoch_start_time) / (batch_idx + 1) if batch_idx > 0 else 0
                eta = batch_time * (len(train_loader) - batch_idx - 1)
                
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'Acc': f'{current_acc:.2f}%',
                    'LR': f'{current_lr:.6f}',
                    'Batch': f'{batch_idx+1}/{len(train_loader)}',
                    'ETA': f'{eta:.0f}s'
                })
        
        # Final GPU sync
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def validate_epoch(self, val_loader) -> Tuple[float, float]:
        """
        Validate the model for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        import time
        start_time = time.time()
        
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Pre-allocate GPU memory to avoid initialization delays
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        setup_time = time.time()
        print(f"Validation setup time: {setup_time - start_time:.3f}s")
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validating", leave=False, mininterval=1.0)
            
            # Warm-up: preload first batch to avoid initial delay
            data_iter = iter(val_loader)
            warmup_start = time.time()
            try:
                first_batch = next(data_iter)
                del first_batch  # Don't process, just warm up the pipeline
            except StopIteration:
                pass
            warmup_time = time.time()
            print(f"Validation warmup time: {warmup_time - warmup_start:.3f}s")
            
            for batch_idx, (data, targets) in enumerate(progress_bar):
                # Move data to device (non-blocking if possible)
                data, targets = data.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
                
                # Update progress bar less frequently
                if batch_idx % 10 == 0:
                    current_loss = running_loss / (batch_idx + 1)
                    current_acc = 100. * correct / total
                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Acc': f'{current_acc:.2f}%'
                    })
        
        # Final GPU sync
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = 100. * correct / total
        
        return epoch_loss, epoch_accuracy
    
    def train(self, train_loader, val_loader=None, epochs: Optional[int] = None) -> Dict[str, List[float]]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            epochs: Number of epochs to train (overrides config if provided)
            
        Returns:
            Training history dictionary
        """
        epochs = epochs or self.config.get('training.epochs', 100)
        
        logging.info(f"Starting training for {epochs} epochs")
        logging.info(f"Device: {self.device}")
        
        # Print device info
        device_info = DeviceManager.get_device_info(self.device)
        for key, value in device_info.items():
            logging.info(f"  {key}: {value}")
        
        start_time = time.time()
        logging.info(f"Starting training for {epochs} epochs on {len(train_loader.dataset)} samples")
        logging.info(f"Batch size: {train_loader.batch_size}, Device: {self.device}")
        logging.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logging.info(f"Expected training time: ~{epochs * len(train_loader) * 0.1/60:.1f} minutes (estimate)")
        logging.info("=" * 80)
        logging.info("")  # Add blank line like in the example
        
        # Overall progress bar for entire training
        from tqdm import tqdm
        overall_progress = tqdm(range(epochs), desc="Training Progress", unit="epoch", position=0, leave=True, ncols=100)  # Increase width
        
        for epoch in overall_progress:
            epoch_start_time = time.time()
            
            # Training
            train_loss, train_accuracy = self.train_epoch(train_loader)
            
            # Validation
            if val_loader:
                val_loss, val_accuracy = self.validate_epoch(val_loader)
            else:
                val_loss, val_accuracy = 0.0, 0.0
            
            # Update learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_accuracy)
                else:
                    self.scheduler.step()
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(val_accuracy)
            self.history['learning_rates'].append(current_lr)
            
            # Track best model
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                self.best_model_state = self.model.state_dict().copy()
            
            # Enhanced epoch logging with more detailed state information
            epoch_time = time.time() - epoch_start_time
            total_time = time.time() - start_time
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Calculate gradient statistics
            grad_norm = 0.0
            for param in self.model.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            
            # Calculate learning rate change
            lr_change = "STABLE"
            if len(self.history['learning_rates']) > 0:
                prev_lr = self.history['learning_rates'][-1]
                if current_lr > prev_lr * 1.01:
                    lr_change = "INCREASING"
                elif current_lr < prev_lr * 0.99:
                    lr_change = "REDUCED"
            
            # Update overall progress bar with enhanced information
            overall_progress.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Acc': f'{val_accuracy:.1f}%',
                'Best': f'{self.best_val_accuracy:.1f}%',
                'Grad Norm': f'{grad_norm:.3f}',
                'LR': f'{current_lr:.6f} ({lr_change})'
            })
            
            # Enhanced detailed logging with comprehensive training state
            if (epoch + 1) % 1 == 0 or val_accuracy > self.best_val_accuracy - 0.01:  # Log every epoch
                # Calculate additional metrics
                samples_per_sec = len(train_loader.dataset) / epoch_time if epoch_time > 0 else 0
                memory_usage = 0
                if self.device.type == 'cuda':
                    memory_usage = torch.cuda.memory_allocated() / 1024**3  # GB
                
                # Create detailed epoch summary in the requested format
                epoch_summary = [
                    f"\n{'='*60}",
                    f"EPOCH {epoch+1}/{epochs} - {epoch_time:.1f}s | Total: {total_time/60:.1f}min | ETA: {((epochs-epoch-1)*epoch_time/60):.0f}min",
                    f"Train: Loss {train_loss:.4f} (initial) | Acc {train_accuracy:.2f}% (initial)",
                    f"Val:   Loss {val_loss:.4f} (initial) | Acc {val_accuracy:.2f}% (initial)",
                    f"LR: {current_lr:.6f} ({lr_change}) | NEW BEST!" if val_accuracy > self.best_val_accuracy - 0.01 else f"LR: {current_lr:.6f} ({lr_change}) | Best: {self.best_val_accuracy:.2f}%",
                    f"Batches: {len(train_loader)} train, {len(val_loader) if val_loader else 0} val | Samples: {len(train_loader.dataset)}",
                    f"{'='*60}"
                ]
                
                for line in epoch_summary:
                    logging.info(line)
        
        # Close overall progress bar
        overall_progress.close()
        
        total_time = time.time() - start_time
        hours = total_time // 3600
        minutes = (total_time % 3600) // 60
        seconds = total_time % 60
        
        logging.info("\n" + "="*80)
        logging.info("TRAINING COMPLETED!")
        logging.info("="*80)
        logging.info(f"Total Time: {int(hours)}h {int(minutes)}m {seconds:.1f}s ({total_time:.1f} seconds)")
        logging.info(f"Average per epoch: {total_time/epochs:.1f}s")
        logging.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
        logging.info(f"Final train accuracy: {self.history['train_accuracy'][-1]:.2f}%")
        logging.info(f"Final validation accuracy: {self.history['val_accuracy'][-1]:.2f}%")
        logging.info(f"Total samples processed: {epochs * len(train_loader.dataset):,}")
        logging.info(f"Samples per second: {epochs * len(train_loader.dataset) / total_time:.1f}")
        logging.info("="*80)
        
        return self.history
    
    def save_model(self, path: str, save_best: bool = True) -> None:
        """
        Save the model.
        
        Args:
            path: Path to save the model
            save_best: Whether to save the best model or current model
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if save_best and self.best_model_state is not None:
            torch.save({
                'model_state_dict': self.best_model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_accuracy': self.best_val_accuracy,
                'config': self.config.config,
                'history': self.history
            }, path)
            logging.info(f"Best model saved to {path}")
        else:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'best_val_accuracy': self.best_val_accuracy,
                'config': self.config.config,
                'history': self.history
            }, path)
            logging.info(f"Current model saved to {path}")
    
    def load_model(self, path: str, load_optimizer: bool = True) -> None:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
            load_optimizer: Whether to load optimizer state
        """
        # Load to CPU first to avoid device issues
        checkpoint = torch.load(path, map_location='cpu')
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if load_optimizer and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.history = checkpoint.get('history', self.history)
        
        # Move model to current device
        self.model.to(self.device)
        
        logging.info(f"Model loaded from {path}")
        logging.info(f"Best validation accuracy: {self.best_val_accuracy:.2f}%")
    
    def get_predictions(self, data_loader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get predictions from the model.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Tuple of (predictions, targets, probabilities)
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                probabilities = torch.softmax(outputs, dim=1)
                _, predictions = torch.max(outputs, 1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_targets), np.array(all_probabilities)

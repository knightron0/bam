"""Simple config-driven trainer."""

from math import ceil
import torch
import torch.nn.functional as F
import wandb
import time
import matplotlib.pyplot as plt

from .param_selectors import select_parameters
from .lr_schedulers import get_lr_scheduler
from optim import get_optimizer
from utils import print_columns, print_training_details, logging_columns_list


class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.training_config = config['training']
        
        # Setup optimizer groups
        self.optimizers = []
        self.lr_schedulers = []
        self._setup_optimizers()
        
        # Loss config
        loss_config = self.training_config.get('loss', {'type': 'cross_entropy'})
        self.label_smoothing = loss_config.get('label_smoothing', 0.0)
        
        # Timing
        self.starter = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
        self.ender = torch.cuda.Event(enable_timing=True) if device == 'cuda' else None
        self.time_seconds = 0.0
        
        self.optimizer_step_times = []
        
        self.step = 0
    
    def _setup_optimizers(self):
        """Setup optimizers from config."""
        groups = self.training_config.get('optimizer_groups', [])
        
        if not groups:
            # Default: all params with SGD
            groups = [{
                'name': 'default',
                'param_selector': {'type': 'all'},
                'optimizer': {'type': 'sgd', 'lr': 0.01, 'momentum': 0.9},
                'lr_schedule': {'type': 'linear_decay'}
            }]
        
        selected_param_ids = set()
        total_optimized_params = 0
        
        for group_config in groups:
            params = select_parameters(self.model, group_config['param_selector'], exclude_params=selected_param_ids)
            if len(params) == 0:
                continue
            
            selected_param_ids.update(id(p) for p in params)
            total_optimized_params += sum(p.numel() for p in params)
            
            opt_config = group_config['optimizer']
            opt_type = opt_config['type'].lower()
            
            if opt_type == 'sgd':
                optimizer = torch.optim.SGD(
                    params,
                    lr=opt_config['lr'],
                    momentum=opt_config.get('momentum', 0.9),
                    weight_decay=opt_config.get('weight_decay', 0.0),
                    nesterov=opt_config.get('nesterov', False),
                    fused=opt_config.get('fused', False)
                )
            elif opt_type == 'adamw':
                optimizer = torch.optim.AdamW(
                    params,
                    lr=opt_config['lr'],
                    weight_decay=opt_config.get('weight_decay', 0.0),
                    eps=opt_config.get('eps', 1e-8),
                    betas=opt_config.get('betas', (0.9, 0.999))
                )
            else:
                optimizer = get_optimizer(opt_type)(
                    params,
                    lr=opt_config['lr'],
                    **{k: v for k, v in opt_config.items() if k not in ['type', 'lr']}
                )
            
            try:
                selected_ids = {id(p) for p in params}
                optimizer.param_id_to_name = {id(p): n for n, p in self.model.named_parameters() if id(p) in selected_ids}
            except Exception:
                optimizer.param_id_to_name = {}
            
            for param_group in optimizer.param_groups:
                param_group['initial_lr'] = param_group['lr']
            
            # Print parameter count for this optimizer group
            num_params = sum(p.numel() for p in params)
            num_tensors = len(params)
            opt_name = group_config['name']
            print(f"Optimizer group '{opt_name}' ({opt_type}): {num_params:,} parameters in {num_tensors} tensors")
            
            self.optimizers.append({
                'optimizer': optimizer,
                'schedule_config': group_config.get('lr_schedule', {'type': 'linear_decay'}),
                'name': group_config['name']
            })
        
        # Print summary
        total_model_params = sum(p.numel() for p in self.model.parameters())
        total_trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total optimized parameters: {total_optimized_params:,} / {total_trainable_params:,} trainable / {total_model_params:,} total model parameters")
        
        unoptimized_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad and id(param) not in selected_param_ids:
                unoptimized_params.append((name, param.numel(), param.shape))
        
        if unoptimized_params:
            total_unoptimized = sum(num for _, num, _ in unoptimized_params)
            print(f"\nWARNING: {total_unoptimized:,} trainable parameters are NOT assigned to any optimizer group:")
            for name, num, shape in unoptimized_params:
                print(f"  {name}: {num:,} params, shape {shape}")
    
    def train(self, train_loader, test_loader=None, wandb_enabled=False, wandb_config=None, warmup=False):
        """Main training loop."""
        if warmup:
            wandb_enabled = False

        run_name = "warmup" if warmup else getattr(self, 'run_number', '')
            
        # Setup schedulers (now we know train_loader size)
        total_epochs = self.training_config.get('epochs', 10)
        total_train_steps = ceil(total_epochs * len(train_loader))
        special_config = self.training_config.get('special', {})
        
        for opt_info in self.optimizers:
            schedule_config = opt_info['schedule_config']
            total_steps_ref = schedule_config.get('total_steps', 'total_epochs')
            
            if total_steps_ref == 'total_epochs':
                steps = total_train_steps
            elif total_steps_ref in special_config:
                steps = ceil(special_config[total_steps_ref] * len(train_loader))
            else:
                steps = int(total_steps_ref)
            
            opt_info['scheduler_fn'] = get_lr_scheduler(schedule_config, steps)
        
        if hasattr(self.model, 'reset'):
            self.model.reset()
        self.step = 0
        
        if hasattr(self.model, 'pre_training_init'):
            self._start_timer()
            self.model.pre_training_init(train_loader, special_config)
            self._stop_timer()
        
        if wandb_enabled and wandb_config:
            wandb.init(**wandb_config)
        
        # actual training loop
        for epoch in range(ceil(total_train_steps / len(train_loader))):
            self._start_timer()
            self.model.train()
            
            train_loss = 0
            num_samples = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                if self.training_config.get('whiten_epochs', 0) and self.step < self.training_config.get('whiten_epochs', 0) * len(train_loader):
                    outputs = self.model(inputs, whiten_bias_grad=False)
                else:
                    outputs = self.model(inputs)
                
                loss = F.cross_entropy(
                    outputs.float(), 
                    labels, 
                    label_smoothing=self.label_smoothing,
                    reduction='sum'
                )
                
                # Check for NaN in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    raise RuntimeError(
                        f"NaN/Inf detected in loss at step {self.step}. "
                        f"Loss value: {loss.item()}. "
                        f"Output stats: min={outputs.min().item():.4f}, max={outputs.max().item():.4f}, "
                        f"mean={outputs.mean().item():.4f}, std={outputs.std().item():.4f}"
                    )
                
                loss.backward()
                
                train_loss += loss.detach().item()
                num_samples += len(inputs)
                
                for opt_info in self.optimizers:
                    optimizer = opt_info['optimizer']
                    scheduler_fn = opt_info['scheduler_fn']
                    opt_name = opt_info['name']
                    
                    for param_group in optimizer.param_groups:
                        initial_lr = param_group['initial_lr']
                        param_group['lr'] = scheduler_fn(initial_lr, self.step)
                    
                    # Track optimizer step time
                    step_start_time = time.perf_counter()
                    optimizer.step()
                    step_end_time = time.perf_counter()
                    
                    step_time = step_end_time - step_start_time
                    
                    self.optimizer_step_times.append({
                        'step': self.step,
                        'optimizer_name': opt_name,
                        'optimizer_type': type(optimizer).__name__,
                        'step_time_seconds': step_time
                    })

                    if wandb_enabled and hasattr(optimizer, 'last_svs_by_name'):
                        if optimizer.last_svs_by_name:
                            for pname, svs in optimizer.last_svs_by_name.items():
                                log_dict = {'step': self.step}
                                if 'weight' in svs and svs['weight'].numel() > 0:
                                    w = svs['weight']
                                    indices = torch.arange(len(w), dtype=torch.float32).numpy()
                                    w_numpy = w.numpy()
                                    
                                    # Create matplotlib figure
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(indices, w_numpy, 'b-', linewidth=2)
                                    ax.set_xlabel('Index', fontsize=12)
                                    ax.set_ylabel('Singular Value', fontsize=12)
                                    ax.set_title(f'Singular Values: {opt_name}/{pname} (weights)', fontsize=14)
                                    ax.grid(True, alpha=0.3)
                                    
                                    log_dict[f'singular/weight/{opt_name}/{pname}'] = wandb.Image(fig)
                                    plt.close(fig)
                                    
                                    log_dict[f'singular/weight_max/{opt_name}/{pname}'] = float(w.max().item())
                                    log_dict[f'singular/weight_mean/{opt_name}/{pname}'] = float(w.mean().item())
                                if 'update' in svs and svs['update'].numel() > 0:
                                    u = svs['update']
                                    indices = torch.arange(len(u), dtype=torch.float32).numpy()
                                    u_numpy = u.numpy()
                                    
                                    # Create matplotlib figure
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.plot(indices, u_numpy, 'r-', linewidth=2)
                                    ax.set_xlabel('Index', fontsize=12)
                                    ax.set_ylabel('Singular Value', fontsize=12)
                                    ax.set_title(f'Singular Values: {opt_name}/{pname} (updates)', fontsize=14)
                                    ax.grid(True, alpha=0.3)
                                    
                                    log_dict[f'singular/update/{opt_name}/{pname}'] = wandb.Image(fig)
                                    plt.close(fig)
                                    
                                    log_dict[f'singular/update_max/{opt_name}/{pname}'] = float(u.max().item())
                                    log_dict[f'singular/update_mean/{opt_name}/{pname}'] = float(u.mean().item())
                                if len(log_dict) > 1:
                                    wandb.log(log_dict)
                            optimizer.last_svs_by_name = {}
                
                self.model.zero_grad(set_to_none=True)
                self.step += 1
                
                if self.step >= total_train_steps:
                    break

            self._stop_timer()
            
            # metrics
            train_acc = (outputs.detach().argmax(1) == labels).float().mean().item()
            train_loss = train_loss / num_samples
            
            # Evaluate with config-driven args
            eval_config = self.training_config.get('evaluation', {})
            eval_kwargs = {k: v for k, v in eval_config.items() if k != 'frequency'}
            val_acc = self.evaluate(test_loader, **eval_kwargs) if test_loader else None
            
            # logging 
            metrics = {
                'epoch': epoch,
                'step': self.step,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'time_seconds': self.time_seconds
            }
            if val_acc is not None:
                metrics['eval_accuracy'] = val_acc
        
            table_vars = {
                'run': run_name,
                'epoch': epoch,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'final_val_accuracy': '',
                'train_loss': train_loss,
                'time_seconds': self.time_seconds
            }
            print_training_details(table_vars, is_final_entry=False)
            run_name = None
            
            if wandb_enabled:
                current_epoch_start_step = epoch * len(train_loader)
                current_epoch_end_step = (epoch + 1) * len(train_loader)
                
                for timing_data in self.optimizer_step_times:
                    if current_epoch_start_step <= timing_data['step'] < current_epoch_end_step:
                        metric_name = f"optimizer_step_time/{timing_data['optimizer_name']}_{timing_data['optimizer_type']}"
                        wandb.log({
                            metric_name: timing_data['step_time_seconds'],
                            'step': timing_data['step']
                        })
                
                wandb.log(metrics)
                
                self.optimizer_step_times = [data for data in self.optimizer_step_times 
                                           if not (current_epoch_start_step <= data['step'] < current_epoch_end_step)]
            
            if self.step >= total_train_steps:
                break
        
        eval_config = self.training_config.get('evaluation', {})
        eval_accuracy = None
        if test_loader:
            self._start_timer()
            eval_accuracy = self.evaluate(test_loader, **eval_config)
            self._stop_timer()
            
            if wandb_enabled:
                wandb.log({
                    "final_val_accuracy": eval_accuracy,
                    "eval_time_seconds": self.time_seconds
                })
            
            table_vars = {
                'run': run_name,
                'epoch': 'eval',
                'train_acc': train_acc,
                'val_acc': val_acc,
                'final_val_accuracy': eval_accuracy,
                'train_loss': train_loss,
                'time_seconds': self.time_seconds
            }
            print_training_details(table_vars, is_final_entry=True)
            
            if wandb_enabled:
                wandb.finish()
            
            return eval_accuracy
        
        if wandb_enabled:
            wandb.finish()
        
        self.optimizer_step_times = []
        
        return eval_accuracy if eval_accuracy is not None else val_acc
    
    def evaluate(self, test_loader, **kwargs):
        """Evaluate model."""
        if hasattr(self.model, 'evaluate'):
            return self.model.evaluate(test_loader, **kwargs)
        
        # Fallback for models without evaluate method
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += len(labels)
        
        return correct / total if total > 0 else 0.0
    
    def _start_timer(self):
        if self.starter:
            self.starter.record()
    
    def _stop_timer(self):
        if self.ender:
            self.ender.record()
            torch.cuda.synchronize()
            self.time_seconds += 1e-3 * self.starter.elapsed_time(self.ender)


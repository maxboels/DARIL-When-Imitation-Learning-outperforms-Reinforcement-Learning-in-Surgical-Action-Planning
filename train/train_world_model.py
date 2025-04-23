import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

def train_world_model(cfg, logger, model, train_loader, test_video_loaders, device='cuda'):
    """
    Train a world model with proper temporal relationships and evaluation metrics.
    This approach follows the Era of Experience principles by learning from
    continuous streams of experience.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        model: WorldModel instance
        train_loader: DataLoader for training data
        test_video_loaders: List of DataLoaders for test videos
        device: Device to train on
        
    Returns:
        Path to the best saved model
    """
    # Extract configuration
    config = cfg['models']['world_model']
    train_config = cfg['training']
    epochs = train_config['epochs']
    lr = train_config['learning_rate']
    weight_decay = train_config['weight_decay']
    warmup_steps = train_config['scheduler']['warmup_steps']
    clip_grad = train_config.get('clip_grad', 1.0)
    eval_batch_interval = train_config['log_every_n_steps']
    eval_epoch_interval = train_config.get('eval_epoch_interval', 1)

    # Create log directory
    log_dir = logger.log_dir
    os.makedirs(log_dir, exist_ok=True)

    # Create checkpoint directory
    checkpoint_dir = os.path.join(logger.log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tensorboard for logging
    tensorboard_dir = os.path.join(logger.log_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    tb_writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=epochs * len(train_loader)
    )
    
    # Track best model and metrics
    best_model_path = None
    best_val_loss = float('inf')
    metrics_history = defaultdict(list)
    
    # Start training
    logger.info(f"[TRAINING] Starting world model training for {epochs} epochs")
    model.train()
    
    for epoch in range(epochs):
        # Training loop
        model.train()
        train_losses = defaultdict(float)
        start_time = time.time()
        
        # Use tqdm for progress bar
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            # Move batch to device
            current_states = batch['current_states'].to(device)
            next_states = batch['next_states'].to(device)
            next_actions = batch['next_actions'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            outputs = model(
                current_state=current_states,
                next_state=next_states,
                next_actions=next_actions,
                attention_mask=attention_mask
            )
            
            # Get losses
            loss = outputs['loss']
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            if clip_grad > 0:
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            
            # Log losses
            for key, value in outputs.items():
                if key.endswith('loss') and isinstance(value, torch.Tensor):
                    train_losses[key] += value.item()
                    tb_writer.add_scalar(f"train/{key}_per_batch", value.item(), epoch * len(train_loader) + batch_idx)

                    if batch_idx % eval_batch_interval == 0 and key == 'loss':
                        logger.info(f"[TRAINING] Batch {batch_idx}/{len(train_loader)} | "
                                    f"{key}: {value.item():.4f}")
        
        # Calculate average losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
            metrics_history[f"train_{key}"].append(train_losses[key])
            tb_writer.add_scalar(f"train/{key}_per_epoch", train_losses[key], epoch)
        
        # Log training progress
        epoch_time = time.time() - start_time
        logger.info(f"[TRAINING] Epoch {epoch+1}/{epochs} completed in {epoch_time:.2f}s | "
                   f"Train loss: {train_losses['loss']:.4f}")
        
        # Evaluation
        if (epoch + 1) % eval_epoch_interval == 0 or epoch == epochs - 1:
            val_metrics = evaluate_world_model(cfg, logger, model, test_video_loaders, device, epoch=epoch)
            
            # Log validation metrics
            for key, value in val_metrics.items():
                metrics_history[f"val_{key}"].append(value)
                tb_writer.add_scalar(f"val/{key}_per_epoch", value, epoch)
            
            logger.info(f"[EVAL] Validation | Loss: {val_metrics['loss']:.4f} | "
                       f"State Pred MSE: {val_metrics['state_pred_error']:.4f} | "
                       f"Rollout MSE Mean: {val_metrics['rollout_error_mean']:.4f} | "
                       f"Rollout MSE Growth: {val_metrics['rollout_error_growth']:.4f} | "
                       f"Action Pred Accuracy: {val_metrics['action_pred_accuracy']:.4f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                model_path = os.path.join(log_dir, f"world_model_best_epoch_{epoch+1}.pt")
                model.save(model_path)
                best_model_path = model_path
                logger.info(f"New best model saved at {model_path}")
        
        # Always save the latest model
        latest_path = os.path.join(log_dir, "world_model_latest.pt")
        model.save(latest_path)
    
    # Save training history
    history_path = os.path.join(log_dir, "training_history.pt")
    torch.save(metrics_history, history_path)
    
    # Plot training curves
    plot_training_curves(metrics_history, log_dir)
    
    logger.info(f"Training completed. Best model saved at {best_model_path}")
    return best_model_path

def evaluate_world_model(cfg, logger, model, test_video_loaders, device='cuda', epoch=0):
    """
    Evaluate the world model on test videos.
    
    This function evaluates both state prediction accuracy and action prediction accuracy,
    as well as performing multi-step rollouts to assess the model's ability to predict
    future surgical states over time.
    
    Args:
        cfg: Configuration dictionary
        model: WorldModel instance
        test_video_loaders: List of DataLoaders for test videos
        device: Device to evaluate on
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    metrics = defaultdict(float)
    num_videos = len(test_video_loaders)
    video_ids = list(test_video_loaders.keys())
    logger.info(f"[EVAL] Evaluating world model on {num_videos} test videos")
    logger.info(f"[EVAL] Video IDs: {video_ids}")

    # Extract evaluation config
    eval_config = cfg['evaluation']['world_model']
    rollout_horizon = eval_config.get('rollout_horizon', 10)
    
    with torch.no_grad():
        # Evaluate on each test video
        for video_id, video_loader in test_video_loaders.items():
            video_metrics = defaultdict(float)
            num_batches = 0
            
            # Process each batch in the video
            for batch_idx, batch in enumerate(tqdm(video_loader, desc=f"Evaluating video {video_id}")):
                num_batches += 1
                
                # Move batch to device
                current_states = batch['current_states'].to(device)
                next_states = batch['next_states'].to(device)
                next_actions = batch['next_actions'].to(device)
                future_states = batch['future_states'].to(device) # [batch_size, max_horizon, embedding_dim]
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # Single-step next state prediction for evaluation
                outputs = model(
                    current_state=current_states,
                    next_state=next_states,
                    next_actions=next_actions,
                    attention_mask=attention_mask
                )
                
                # Calculate metrics for this batch
                # State prediction error (MSE)
                state_pred_error = ((outputs['_z_hat'] - next_states) ** 2).mean().item()
                video_metrics['state_pred_error'] += state_pred_error
                
                # Action prediction accuracy (if applicable)
                if model.imitation_learning and '_a' in model.heads:
                    if 'head_outputs' in outputs and '_a' in outputs['head_outputs']:
                        action_logits = outputs['head_outputs']['_a']
                        pred_actions = (torch.sigmoid(action_logits) > 0.5).float()
                        action_accuracy = (pred_actions == next_actions).float().mean().item()
                        video_metrics['action_pred_accuracy'] += action_accuracy
                
                # Add loss
                if 'loss' in outputs:
                    video_metrics['loss'] += outputs['loss'].item()
                
                # Generate rollout trajectory
                rollout = model.generate_conditional_future_states(
                    input_embeddings=current_states,
                    input_actions=next_actions, # check if sample or use true next actions from expert
                    horizon=rollout_horizon,
                    temperature=0.7,
                    use_past=True
                )
                
                # Extract generated trajectory
                gen_states = rollout['full_embeddings'][:, -rollout_horizon:]  # [batch_size, rollout_horizon, embedding_dim]
                gen_actions = rollout['full_actions'] if 'full_actions' in rollout else None
                
                # Calculate rollout error over timestep
                rollout_errors = []
                for t in range(gen_states.size(1)):
                    # Only calculate error for timesteps that exist in the future states
                    if t < gen_states.size(1):
                        error_t = ((future_states[:, t] - gen_states[:, t]) ** 2).mean().item()
                        rollout_errors.append(error_t)
                
                # Aggregate rollout metrics
                video_metrics['rollout_error_mean'] = sum(rollout_errors) / len(rollout_errors) if rollout_errors else 0
                video_metrics['rollout_error_final'] = rollout_errors[-1] if rollout_errors else 0
                
                # Calculate error growth rate (how quickly errors accumulate)
                if len(rollout_errors) > 1:
                    error_growth = (rollout_errors[-1] / (rollout_errors[0] + 1e-8))
                    video_metrics['rollout_error_growth'] = error_growth
            
            # Calculate average metrics for this video
            for key in video_metrics:
                if key not in ['rollout_error_mean', 'rollout_error_final', 'rollout_error_growth']:
                    video_metrics[key] /= num_batches
            
            # Add to overall metrics (average across videos)
            for key, value in video_metrics.items():
                metrics[key] += value / num_videos
            
            # Log video metrics
            logger.info(f"[EVAL] Video {video_id} | "
                       f"State Pred MSE: {video_metrics['state_pred_error']:.4f} | "
                       f"Rollout MSE Mean: {video_metrics['rollout_error_mean']:.4f} | "
                       f"Rollout MSE Growth: {video_metrics['rollout_error_growth']:.4f}")
    
    # Set default action prediction accuracy if not calculated
    if 'action_pred_accuracy' not in metrics:
        metrics['action_pred_accuracy'] = 0.0
    
    return dict(metrics)

def run_generation_inference(cfg, logger, model, test_video_loaders, device='cuda'):
    """
    Run inference with the world model to generate trajectories.
    
    This function takes the trained world model and generates multi-step predictions
    to evaluate its ability to predict future surgical states and actions.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        model: WorldModel instance
        test_video_loaders: List of DataLoaders for test videos
        device: Device to evaluate on
        
    Returns:
        Dictionary of results containing generated trajectories and metrics
    """
    model.eval()
    
    # Extract evaluation config
    eval_config = cfg['evaluation']
    num_trajectories = eval_config.get('num_trajectories', 5)
    trajectory_length = eval_config.get('trajectory_length', 20)
    log_dir = eval_config.get('log_dir', 'generation_results')
    
    # Create directory for saving results
    os.makedirs(log_dir, exist_ok=True)
    
    # List to store all results
    all_results = []
    
    # Generate trajectories for selected videos/frames
    logger.info(f"Generating {num_trajectories} trajectories of length {trajectory_length}")
    
    with torch.no_grad():
        trajectory_count = 0
        
        # Iterate through test videos
        for video_idx, video_loader in enumerate(test_video_loaders):
            # Process only some videos to limit the total trajectories
            if trajectory_count >= num_trajectories:
                break
            
            # Get one batch from this video
            for batch in video_loader:
                # Move batch to device
                current_states = batch['current_states'].to(device)
                next_states = batch.get('next_states', None)
                if next_states is not None:
                    next_states = next_states.to(device)
                next_actions = batch.get('next_actions', None)
                if next_actions is not None:
                    next_actions = next_actions.to(device)
                
                # Generate trajectories for first few examples in batch
                batch_size = min(current_states.size(0), num_trajectories - trajectory_count)
                
                for i in range(batch_size):
                    # Extract initial state and action
                    initial_state = current_states[i:i+1, 0:1].clone()  # [1, 1, embedding_dim]
                    
                    # Extract or create initial action
                    if next_actions is not None:
                        initial_action = next_actions[i:i+1, 0:1].clone()  # [1, 1, action_classes]
                    else:
                        # If no action provided, sample a random one
                        initial_action = torch.zeros(1, 1, model.num_action_classes, device=device)
                        initial_action = initial_action.bernoulli(0.5)  # Random binary actions
                    
                    # Generate trajectory
                    generated = model.generate_conditional_future_states(
                        input_embeddings=initial_state,
                        input_actions=initial_action,
                        horizon=trajectory_length,
                        temperature=0.7,
                        use_past=True
                    )
                    
                    # Extract ground truth if available (for comparison)
                    gt_trajectory = None
                    if next_states is not None:
                        max_gt_length = min(trajectory_length + 1, current_states.size(1))
                        gt_trajectory = {
                            'states': current_states[i:i+1, :max_gt_length].cpu(),
                            'actions': next_actions[i:i+1, :max_gt_length].cpu() if next_actions is not None else None
                        }
                    
                    # Calculate metrics comparing generated to ground truth
                    metrics = {}
                    if gt_trajectory is not None:
                        # State prediction error at each timestep
                        gen_states = generated['full_embeddings']
                        gt_states = gt_trajectory['states']
                        
                        # Calculate error for overlapping timesteps
                        min_length = min(gen_states.size(1), gt_states.size(1))
                        state_errors = []
                        
                        for t in range(1, min_length):  # Skip initial state
                            error_t = ((gt_states[:, t] - gen_states[:, t]) ** 2).mean().item()
                            state_errors.append(error_t)
                        
                        # Calculate state prediction metrics
                        metrics['state_error_mean'] = sum(state_errors) / len(state_errors) if state_errors else 0
                        metrics['state_error_final'] = state_errors[-1] if state_errors else 0
                        metrics['state_error_growth'] = state_errors[-1] / (state_errors[0] + 1e-8) if len(state_errors) > 1 else 0
                    
                    # Store result
                    result = {
                        'video_idx': video_idx,
                        'example_idx': i,
                        'generated': {
                            'states': generated['full_embeddings'].cpu(),
                            'actions': generated['full_actions'].cpu() if 'full_actions' in generated else None,
                            'head_outputs': {k: v.cpu() for k, v in generated['head_outputs'].items()} if 'head_outputs' in generated else None
                        },
                        'ground_truth': gt_trajectory,
                        'metrics': metrics
                    }
                    
                    all_results.append(result)
                    trajectory_count += 1
                
                # Only process one batch per video
                break
    
    # Calculate overall metrics
    overall_metrics = defaultdict(float)
    for result in all_results:
        for key, value in result['metrics'].items():
            overall_metrics[key] += value
    
    # Average metrics
    for key in overall_metrics:
        overall_metrics[key] /= len(all_results)
    
    # Log overall metrics
    logger.info(f"Generation metrics | State Error Mean: {overall_metrics.get('state_error_mean', 0):.4f} | "
               f"State Error Growth: {overall_metrics.get('state_error_growth', 0):.4f}")
    
    # Save results
    results_path = os.path.join(log_dir, "generation_results.pt")
    torch.save({
        'results': all_results,
        'overall_metrics': overall_metrics
    }, results_path)
    
    return {
        'results': all_results,
        'overall_metrics': dict(overall_metrics)
    }

def plot_training_curves(metrics_history, log_dir):
    """
    Plot training and validation curves.
    
    Args:
        metrics_history: Dictionary of metrics history
        log_dir: Directory to save plots
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Plot loss curves
    plt.figure(figsize=(10, 6))
    if 'train_loss' in metrics_history:
        plt.plot(metrics_history['train_loss'], label='Train Loss')
    if 'val_loss' in metrics_history:
        plt.plot(metrics_history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'loss_curves.png'))
    # plt.close()
    
    # Plot state prediction error
    plt.figure(figsize=(10, 6))
    if 'train_state_pred_error' in metrics_history:
        plt.plot(metrics_history['train_state_pred_error'], label='Train State Error')
    if 'val_state_pred_error' in metrics_history:
        plt.plot(metrics_history['val_state_pred_error'], label='Validation State Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('State Prediction Error')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'state_error_curves.png'))
    plt.close()
    
    # Plot action prediction accuracy (if available)
    if 'train_action_pred_accuracy' in metrics_history or 'val_action_pred_accuracy' in metrics_history:
        plt.figure(figsize=(10, 6))
        if 'train_action_pred_accuracy' in metrics_history:
            plt.plot(metrics_history['train_action_pred_accuracy'], label='Train Action Accuracy')
        if 'val_action_pred_accuracy' in metrics_history:
            plt.plot(metrics_history['val_action_pred_accuracy'], label='Validation Action Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Action Prediction Accuracy')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(log_dir, 'action_accuracy_curves.png'))
        plt.close()

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Create a learning rate scheduler with linear warmup and decay.
    
    Args:
        optimizer: Optimizer
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total number of training steps
        last_epoch: Last epoch (-1 for start from scratch)
        
    Returns:
        Scheduler
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)
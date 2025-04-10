import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
import os
import glob
import json
import pandas as pd
import re
from tqdm import tqdm
import yaml
import os
from datetime import datetime


# Training function for next frame predictor
def train_next_frame_model(cfg, logger, model, train_loader, val_loader=None,
                            device='cuda'):

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg['training']['learning_rate'])

    with open(cfg['data']['paths']['class_labels_file_path'], 'r') as f:
        class_labels_names = json.load(f)
    action_labels_name = [class_name for class_id, class_name in class_labels_names['action'].items()]


    save_logs_dir = logger.log_dir
    
    # Create checkpoint directory if it doesn't exist
    checkpoint_dir = os.path.join(save_logs_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # For tracking best model
    best_val_loss = float('inf')

    best_model_path = cfg['best_model_path']
    logger.info(f"Best model path: {best_model_path}")
    
    # For logging
    tensorboard_dir = os.path.join(save_logs_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    train_losses = []
    val_losses = []
    epochs = cfg['training']['epochs']

    # Save results to text and json file
    results_all_file_json = os.path.join(save_logs_dir, 'results_dict.json')
    results_all_file_txt = os.path.join(save_logs_dir, 'results_text.txt')

    for epoch in range(epochs):
        # Skip training if best model is given and exists
        if best_model_path and os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_val_loss = checkpoint['val_loss']
            logger.info(f"Loaded best model from {best_model_path}")
            logger.info(f"Skip training and start validation")
        else:
            # Training
            model.train()
            train_loss = 0.0
            logger.info(f"Epoch {epoch+1}/{epochs} Training")

            for batch_idx, (z_seq, _z_seq, _a_seq, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")):
                z_seq, _z_seq, _a_seq = z_seq.to(device), _z_seq.to(device), _a_seq.to(device)

                assert z_seq[:, -1, :].equal(_z_seq[:, -2, :]), "Last frame of z must equal second-to-last frame of _z"
                
                # Forward pass
                outputs = model(z_seq, next_frame=_z_seq, next_actions=_a_seq)

                # Get loss
                _z_loss = outputs["_z_loss"]
                _a_loss = outputs['_a_loss'] if '_a_loss' in outputs else 0.0
                loss = _z_loss + _a_loss
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * z_seq.size(0)

                # Calculate global step using epoch and batch index
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/World_Model_Train', loss.item(), global_step)
                writer.add_scalar('Loss/World_Model_Train_Z', _z_loss.item(), global_step)
                writer.add_scalar('Loss/World_Model_Train_A', _a_loss.item(), global_step)

                # Logging every
                if batch_idx % cfg['training']['log_every_n_steps'] == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                    
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            logger.info(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
        
        # Inference on validation set
        logger.info(f"Epoch {epoch+1}/{epochs} Validation")
        val_loss = 0.0
        model.eval()
        logger.info(f"Starting validation")

        # Initialize results dictionary
        results = {}
        eval_horizons = cfg['eval']['world_model']['eval_horizons']
        top_ks = cfg['eval']['world_model']['top_ks']
        for horizon in eval_horizons:
            for k in top_ks:
                results[f"horizon_{horizon}_top_{k}"] = []
        
        use_memory = cfg['eval']['world_model']['use_memory']
        max_horizon = cfg['eval']['world_model']['max_horizon']
        action_accuracies = []
        with torch.no_grad():
            for batch_idx, (z_seq, _z_seq, _a_seq, f_a_seq) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} Validation")):
                z_seq, _z_seq, _a_seq, f_a_seq = z_seq.to(device), _z_seq.to(device), _a_seq.to(device), f_a_seq.to(device)
                global_step = epoch * len(val_loader) + batch_idx

                # Reset batch loss
                val_batch_loss = 0.0

                # New frame observed
                # We should already have seen the previous frames and stored them memory
                if use_memory:
                    z_seq = z_seq[:, -1:, :] # select current frame
                
                # Generate next frame predictions
                outputs = model.generate(z_seq, horizon=max_horizon, use_memory=use_memory)
                
                # Calculate loss between predictions and targets
                # We didnt pass the future embeddings in out dataloader
                _z_loss = F.mse_loss(outputs['_zs_hat'][:, 0, :], _z_seq[:, -1, :]) # Check if first frame isn't the input frame
                val_batch_loss += _z_loss.item() * z_seq.size(0)

                if 'f_a_seq_hat' in outputs:
                    f_a_loss = F.binary_cross_entropy_with_logits(outputs['f_a_seq_hat'], f_a_seq)
                    val_batch_loss += f_a_loss.item() * z_seq.size(0)

                    # Convert logits to probabilities
                    f_a_probs = torch.sigmoid(outputs['f_a_seq_hat'])

                    # Select intervals for evaluation
                    for h in eval_horizons:
                        if h <= outputs['f_a_seq_hat'].shape[1]:  # Make sure horizon is within prediction range
                            
                            f_a_h_probs = f_a_probs[:, :h, :]
                            f_a_h_targets = f_a_seq[:, :h, :]
                            
                            # Accuracy, Recall, Precision, F1 (top-k)
                            true_indices = torch.where(f_a_h_targets > 0.5)[0]
                            if len(true_indices) == 0:
                                continue # we don't compute false positives and true negatives

                            horizon_metrics = evaluate_multi_label_predictions(f_a_h_probs, f_a_h_targets, top_ks)
                            
                            # Store results with horizon prefix
                            for metric_name, value in horizon_metrics.items():
                                results[f"horizon_{h}_{metric_name}"] = value

                            # mean Average Precision (mAP) for each horizon
                            map_scores = calculate_map(f_a_h_probs, f_a_h_targets, class_names=action_labels_name)
                            results[f"horizon_{h}_mAP"] = map_scores['mAP']

                            # Log and visualize results
                            writer.add_scalar(f'Metrics/mAP_Horizon_{h}', map_scores['mAP'], global_step)
                    
                    
                    writer.add_scalar('Loss/World_Model_Val_A', f_a_loss.item(), global_step)

                # Accumulate batch loss   
                val_loss += val_batch_loss
                writer.add_scalar('Loss/World_Model_Val', val_batch_loss, global_step)
                writer.add_scalar('Loss/World_Model_Val_Z', _z_loss.item(), global_step)
                    
                # Evaluation loop logging
                if batch_idx % cfg['training']['log_every_n_steps'] == 0:
                    logger.info(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(val_loader)}, Loss: {val_batch_loss:.4f}")

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        logger.info(f"Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}")
        
        # Log metrics
        log_comprehensive_metrics(results, writer, epoch, logger)

        # Create comparison plots
        plots_dir = os.path.join(save_logs_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)

        metrics_plot = create_metrics_comparison_plot(
            results, 
            os.path.join(plots_dir, f'metrics_comparison_epoch{epoch}.png')
        )

        topk_plot = create_metric_breakdown_by_topk(
            results,
            os.path.join(plots_dir, f'topk_breakdown_epoch{epoch}.png')
        )

        # Calculate and log average accuracies for each horizon and top-k
        logger.info("\nAction Prediction Accuracy:")
        logger.info("-" * 50)
        logger.info(f"{'Horizon':<10} {'Top-k':<10} {'Accuracy':<10}")
        logger.info("-" * 50)
        
        for horizon in sorted(eval_horizons):
            map_key = f"horizon_{horizon}_mAP"
            if map_key in results:
                logger.info(f"{horizon:<10} {'mAP':<15} {results[map_key]:.4f}")
            
            for k in sorted(top_ks):
                key = f"horizon_{horizon}_top_{k}"
                if results[key]:
                    avg_accuracy = sum(results[key]) / len(results[key])
                    logger.info(f"{horizon:<10} {k:<10} {avg_accuracy:.4f}")
                    writer.add_scalar(f'Accuracy/Avg_Horizon_{horizon}_Top_{k}', avg_accuracy, epoch)
        # print("-" * 50)
        logger.info("-" * 50)
        
        # Save model if it's the best so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save plots
            plots_save_dir = os.path.join(save_logs_dir, 'plots')
            os.makedirs(plots_save_dir, exist_ok=True)
            plot_title = f"World Model Evaluation - Epoch {epoch+1}"
            
            # mAP plots
            # Generate all plots at once using the comprehensive function
            plot_files = generate_map_vs_accuracy_plots(
                results, 
                plots_save_dir,
                experiment_name=plot_title
            )

            # Accuracy plots
            plot_files = plot_action_prediction_results(
                results, 
                save_dir=plots_save_dir,
                experiment_name=plot_title
            )

            # Create descriptive filename
            checkpoint_filename = f"best_model_epoch{epoch+1}_valloss{val_loss:.6f}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            
            # Save model state
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': cfg,
                'action_accuracies': {key: (sum(results[key]) / len(results[key]) if isinstance(results[key], list) else results[key]) 
                                     for key in results}
            }, checkpoint_path)

            logger.info(f"Model saved to: {checkpoint_path}")
            logger.info(f"Saved new best model at epoch {epoch+1} with validation loss: {val_loss:.6f}")
            
            best_model_path = checkpoint_path
        
        # Optionally save periodic checkpoints (every N epochs)
        save_frequency = cfg.get('training', {}).get('save_checkpoint_every_n_epochs', 1)
        if save_frequency > 0 and (epoch + 1) % save_frequency == 0:
            checkpoint_filename = f"model_epoch{epoch+1}_valloss{val_loss:.6f}.pt"
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': cfg,
            }, checkpoint_path)
            
            logger.info(f"Saved periodic checkpoint at epoch {epoch+1}")
    
    # Return training statistics and best model path
    return best_model_path
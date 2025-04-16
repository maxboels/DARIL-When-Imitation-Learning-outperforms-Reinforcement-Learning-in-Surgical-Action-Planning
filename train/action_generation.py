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
    best_model_path = cfg['experiment']['pretrain_next_frame']['best_model_path']
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

            # Iterate over training batches
            for batch_idx, (z_seq, _z_seq, _a_seq, f_a, c_a, c_i) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} Training")):
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


        # Validation phase
        if val_loader is not None:
            model.eval()

            # Eval loop
            eval_results = run_world_model_inference(cfg, logger, model, val_loader, 
                device=device, epoch=epoch, tb_writer=writer, run_type='validation'
            )

            # Main metric for model selection
            val_map = eval_results['sklearn_mAP_a_from_vid_ap']

            # Save best model
            if val_map > best_val_map:
                best_val_map = val_map
                # Save model checkpoint
                checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch{epoch+1}_map{val_map:.4f}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_map': val_map,
                    'config': cfg
                }, checkpoint_path)
                
                best_model_path = checkpoint_path
                logger.info(f"Saved new best model with mAP: {val_map:.4f}")
    
        # Load best model
        if best_model_path is not None and os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded best model with mAP: {best_val_map:.4f}")
        
        # returns the best model for inference
        return model

def run_world_model_inference(cfg, logger, model, test_video_loaders, device='cuda', epoch=None, tb_writer=None, run_type='inference'):
    """
    Run inference and evaluation on the test set.
    
    Args:
        model: Trained recognition head model
        test_video_loaders: DataLoaders for test videos
        cfg: Configuration dictionary
        device: Device to run inference on
        log_dir: Directory to save outputs
        
    Returns:
        Output directory with results
    """
    run_string = "[VALIDATION]" if epoch is not None else "[TEST]"
    # Initialize recognition metrics
    horizons = [1, 2, 3, 5, 10]  # Time horizons in seconds
    max_horizon = cfg['eval']['world_model']['max_horizon']
    recognize = {}

    for h in range(1, max_horizon + 1):
        # Initialize recognition metrics
        recognize[f"{h}"] = ivtmetrics.Recognition(num_class=100)
        recognize[f"{h}"].reset_global()

    # Log directory
    log_dir = os.path.join(logger.log_dir, run_type)
    os.makedirs(log_dir, exist_ok=True)
    output_dir = os.path.join(log_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Tensorboard for logging
    if tb_writer is not None:
        writer = tb_writer
    else:
        tensorboard_dir = os.path.join(log_dir, 'tensorboard')
        os.makedirs(tensorboard_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Load class labels
    with open(cfg['data']['paths']['class_labels_file_path'], 'r') as f:
        class_labels = json.load(f)
    action_labels = [class_name for class_id, class_name in class_labels['action'].items()]
    instrument_labels = [class_name for class_id, class_name in class_labels['instrument'].items()]
    
    # Set model to evaluation mode
    model.eval()
    metrics = ['sklearn_vids_mAP_i', 'sklearn_vids_mAP_a']
    videos_scores = {metric: [] for metric in metrics}
    per_class_ap_scores = {}
    per_video_sklearn_per_class_map_a = {}
    per_video_sklearn_per_class_map_i = {}
    overall_results = {} 
    vids = []   
    # Run inference
    with torch.no_grad():
        for vid_id, video_loader in test_video_loaders.items():

            vids.append(vid_id)

            # Initialize containers for predictions and targets
            video_preds_actions = []
            video_targets_actions = []
            video_preds_instruments = []
            video_targets_instruments = []

            for batch_idx, (z_seq, _, _, f_a_seq, c_a, c_i) in enumerate(tqdm(video_loader, desc="Running inference")):
                z_seq, f_a_seq, c_a, c_i = z_seq.to(device), f_a_seq.to(device), c_a.to(device), c_i.to(device)                

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

                    # add max horizon to f_a_seq
                    f_a_targets = f_a_seq[:, :, :]

                    # Store predictions and targets
                    video_preds_actions.append(f_a_probs.cpu().numpy())
                    video_targets_actions.append(f_a_targets.cpu().numpy())
                    # video_preds_instruments.append(instrument_probs.cpu().numpy())
                    # video_targets_instruments.append(target_instruments.cpu().numpy())

                    for h in range(1, max_horizon + 1):
                        # Update recognition metrics
                        f_a_h_probs = f_a_probs[:, h-1, :].unsqueeze(1)
                        f_a_h_targets = f_a_seq[:, h-1, :].unsqueeze(1)
                        recognize[f"{h}"].update(f_a_h_targets.cpu().numpy(), f_a_h_probs.cpu().numpy())
                    
                    writer.add_scalar('Loss/World_Model_Val_A', f_a_loss.item(), global_step)

                # Accumulate batch loss   
                val_loss += val_batch_loss
                writer.add_scalar('Loss/World_Model_Val', val_batch_loss, global_step)
                writer.add_scalar('Loss/World_Model_Val_Z', _z_loss.item(), global_step)
                    
                # Evaluation loop logging
                if batch_idx % cfg['training']['log_every_n_steps'] == 0:
                    logger.info(f"{run_string} Video {vid_id} | Batch {batch_idx}/{len(video_loader)}, Loss: {val_batch_loss:.4f}")

            # End video and add video and init new lists
            for h in range(1, max_horizon + 1):
                recognize[f"{h}"].video_end()

            # Mean Average Precision (mAP) for actions (keep temporal dimension)
            video_targets_actions = np.concatenate(video_targets_actions, axis=0)
            video_preds_actions = np.concatenate(video_preds_actions, axis=0)
            
            for h in range(1, max_horizon + 1):
                # Select intervals for evaluation
                video_targets_actions_h = video_targets_actions[:, h-1, :]
                video_preds_actions_h = video_preds_actions[:, h-1, :]
                # Calculate mAP using sklearn
                action_map_scores = calculate_map_recognition(video_targets_actions_h, video_preds_actions_h, action_labels)
                action_overall_map = action_map_scores['mAP']
                action_per_class_ap = action_map_scores['AP_scores']
                # Log mAP to console
                logger.info(f"{run_string} h={h} | Video {vid_id}: Test mAP (sklearn) Action: {action_overall_map:.4f}")
                # Add to overall results
                videos_scores[f'sklearn_vids_mAP_a_{h}'].append(action_overall_map)
                # Add the video mAP to the per class mAP
                per_video_sklearn_per_class_map_a[f'video_{vid_id}_sklearn_vids_mAP_a_{h}'] = action_per_class_ap

        # END ALL VIDEOS LOOP
        results_ivt = {}
        for h in range(1, max_horizon + 1):

            # sklearn metrics
            action_overall_map = np.mean(videos_scores[f'sklearn_vids_mAP_a_{h}'])
            # instrument_overall_map = np.mean(videos_scores[f'sklearn_vids_mAP_i'])

            # Aggregate all the map scores per class on all videos in per_video_sklearn_per_class_map
            # for each video, add the per class map scores to the per class map scores
            # add the per class map scores to the per class map scores (there are None values)
            vals_h = np.array(list(per_video_sklearn_per_class_map_a[f'video_{vid_id}_sklearn_vids_mAP_a_{h}']))[:, h-1, :]
            per_class_ap_scores[f'sklearn_vids_mAP_a_{h}'] = np.nanmean(vals_h, axis=0).tolist()
            # per_class_ap_scores['sklearn_vids_mAP_i'] = np.nanmean(np.array(list(per_video_sklearn_per_class_map_i.values())), axis=0).tolist()

            # Add the overall nan means to the per class map scores based on the intermediate results from the video AP
            per_class_ap_scores['sklearn_mAP_a_from_vid_ap'] = np.nanmean(np.array(per_class_ap_scores[f'sklearn_vids_mAP_a_{h}']), axis=0).tolist()
            # per_class_ap_scores['sklearn_mAP_i_from_vid_ap'] = np.nanmean(np.array(per_class_ap_scores['sklearn_vids_mAP_i']), axis=0).tolist()

            # ivt metrics
            results_ivt[f"{h}"] = recognize[f"{h}"].compute_video_AP('ivt', ignore_null=False) # try with ignore null classes
            # results_i = recognize.compute_video_AP('i', ignore_null=False) # try with ignore null classes

            # Per class scores (all videos aggragated)
            per_class_ap_scores[f'ivt_map_a_{h}'] = results_ivt[f"{h}"]['AP'].tolist()
            # per_class_ap_scores['ivt_map_i'] = results_i['AP'].tolist()
            per_class_ap_scores[f'ivt_mAP_a_from_vid_ap_{h}'] = np.nanmean(np.array(per_class_ap_scores[f'ivt_map_a_{h}']), axis=0).tolist()
            # per_class_ap_scores['ivt_mAP_i_from_vid_ap'] = np.nanmean(np.array(per_class_ap_scores['ivt_map_i']), axis=0).tolist()

            overall_results[f'ivt_mAP_a_from_vid_ap_{h}'] = per_class_ap_scores[f'ivt_mAP_a_from_vid_ap_{h}']
            overall_results[f'ivt_mAP_a_{h}'] = np.round(results_ivt[f"{h}"]["mAP"], 4)
            # overall_results['ivt_mAP_i'] = np.round(results_i["mAP"], 4)   

            # Save per video mAP results
            overall_results['sklearn_mAP_a_from_vid_ap'] = per_class_ap_scores[f'sklearn_mAP_a_from_vid_ap_{h}']
            overall_results['sklearn_mAP_a'] = np.round(action_overall_map, 4)
            # overall_results['sklearn_mAP_i'] = np.round(instrument_overall_map, 4)
        
            logger.info(f"{run_string} h={h}  Overall action mAP (sklearn): {overall_results['sklearn_mAP_a']:.4f}")
            logger.info(f"{run_string} h={h}  Overall instrument mAP (sklearn): {overall_results['sklearn_mAP_i']:.4f}")
            logger.info(f"{run_string} h={h}  Overall action mAP (ivt): {overall_results['ivt_mAP_a']:.4f}")
            logger.info(f"{run_string} h={h}  Overall instrument mAP (ivt): {overall_results['ivt_mAP_i']:.4f}")

    if epoch is not None:
        # Save mAP results to Tensorboard
        writer.add_scalar('Metrics/Validation_mAP_Action_sklean', overall_results['sklearn_mAP_a'], epoch)
        writer.add_scalar('Metrics/Validation_mAP_Instrument_sklean', overall_results['sklearn_mAP_i'], epoch)
        writer.add_scalar('Metrics/Validation_mAP_Action_ivt', overall_results['ivt_mAP_a'], epoch)
        writer.add_scalar('Metrics/Validation_mAP_Instrument_ivt', overall_results['ivt_mAP_i'], epoch)
        # Save mAP results to console
        logger.info(f"{run_string} Epoch {epoch+1}: mAP Action (sklearn): {overall_results['sklearn_mAP_a']:.4f}")
        logger.info(f"{run_string} Epoch {epoch+1}: mAP Instrument (sklearn): {overall_results['sklearn_mAP_i']:.4f}")
        logger.info(f"{run_string} Epoch {epoch+1}: mAP Action (ivt): {overall_results['ivt_mAP_a']:.4f}")
        logger.info(f"{run_string} Epoch {epoch+1}: mAP Instrument (ivt): {overall_results['ivt_mAP_i']:.4f}")
    else:
        # Save mAP results to Tensorboard
        writer.add_scalar('Metrics/Test_mAP_Action_sklean', overall_results['sklearn_mAP_a'], 0)
        writer.add_scalar('Metrics/Test_mAP_Instrument_sklean', overall_results['sklearn_mAP_i'], 0)
        # Save mAP results to console
        logger.info(f"{run_string} Test mAP Action (sklearn): {overall_results['sklearn_mAP_a']:.4f}")
        logger.info(f"{run_string} Test mAP Instrument (sklearn): {overall_results['sklearn_mAP_i']:.4f}")
        logger.info(f"{run_string} Test mAP Action (ivt): {overall_results['ivt_mAP_a']:.4f}")
        logger.info(f"{run_string} Test mAP Instrument (ivt): {overall_results['ivt_mAP_i']:.4f}")
    # Save mAP results to JSON
    with open(os.path.join(output_dir, 'mAP_results.json'), 'w') as f:
        json.dump(overall_results, f, indent=4)
    # Save predictions to JSON
    with open(os.path.join(output_dir, 'predictions.json'), 'w') as f:
        json.dump(videos_scores, f, indent=4)
    
    # Save per_class_ap_scores 
    with open(os.path.join(output_dir, 'per_class_ap_scores.json'), 'w') as f:
        json.dump(per_class_ap_scores, f, indent=4)
    
    return overall_results
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
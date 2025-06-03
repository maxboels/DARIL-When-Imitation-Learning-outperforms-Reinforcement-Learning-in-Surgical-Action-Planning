import torch
import torch.nn as nn
import numpy as np
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support, average_precision_score
from sklearn.decomposition import PCA
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Union

def evaluate_world_model(
    cfg: Dict[str, Any],
    logger: Any,
    model: nn.Module,
    test_video_loaders: Dict[str, Any],
    device: str = 'cuda',
    eval_mode: str = 'full',
    save_results: bool = True,
    epoch: int = None,
    render_visualizations: bool = True
) -> Dict[str, Any]:
    """
    Comprehensive evaluation function for world models that combines functionality from
    evaluate_world_model, run_generation_inference, and enhanced_inference_evaluation.
    
    This unified function can be used both during training and for final model evaluation.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger instance
        model: WorldModel instance
        test_video_loaders: Dictionary of DataLoaders for test videos
        device: Device to evaluate on
        eval_mode: Evaluation mode - 'basic', 'generation', or 'full'
                   - 'basic': Simple metrics used during training (faster)
                   - 'generation': Include trajectory generation and basic metrics
                   - 'full': Comprehensive evaluation with all metrics and visualizations
        save_results: Whether to save results to disk
        epoch: Current epoch (if called during training)
        render_visualizations: Whether to generate visualizations
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Create results directory if saving
    results_dir = None
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_str = f"_epoch_{epoch}" if epoch is not None else ""
        results_dir = os.path.join("results", f"eval_{eval_mode}{epoch_str}_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)
        logger.info(f"Running evaluation in '{eval_mode}' mode, saving to {results_dir}")
    
    # Extract evaluation config
    eval_config = cfg['evaluation']['world_model']
    rollout_horizon = eval_config.get('rollout_horizon', 10)
    horizons = eval_config.get('eval_horizons', [1, 3, 5, 10, 15])
    overall_h = eval_config.get('overall_horizon', 1)
    
    # Initialize metrics structure
    metrics = {
        'action': {
            'overall': {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'mAP': 0.0
            },
            'per_class': [],
            'per_video': {},
            'horizon': {h: {} for h in horizons}
        },
        'state': {
            'overall': {
                'mse': 0.0
            },
            'per_video': {},
            'horizon': {h: {} for h in horizons}
        },
        'rollout': {
            'overall': {
                'mean_error': 0.0,
                'growth_rate': 0.0
            },
            'per_video': {}
        },
        'total_loss': 0.0
    }
    
    # Also track generated trajectories if needed
    generated_trajectories = [] if eval_mode in ['generation', 'full'] else None
    
    # Process each video
    num_videos = len(test_video_loaders)
    video_ids = list(test_video_loaders.keys())
    logger.info(f"[EVAL] Evaluating world model on {num_videos} test videos")
    logger.info(f"[EVAL] Video IDs: {video_ids}")

    # Init per frame metrics
    metrics_per_frame = {video_id: {} for video_id in test_video_loaders}
    
    with torch.no_grad():
        # Evaluate on each test video
        for video_id, video_loader in test_video_loaders.items():
            video_metrics = defaultdict(float)
            num_batches = 0
            
            # Initialize video-specific metrics
            video_action_preds = []
            video_action_gts = []
            video_state_mse = []
            video_horizon_metrics = {h: {'action_preds': [], 'action_gts': [], 'state_mse': []} for h in horizons}
            
            # Process each batch in the video
            for batch_idx, batch in enumerate(tqdm(video_loader, desc=f"Evaluating video {video_id}")):
                num_batches += 1
                
                # Move batch to device
                current_states = batch['current_states'].to(device)
                future_states = batch.get('future_states', None)
                if future_states is not None:
                    future_states = future_states.to(device)
                next_states = batch.get('next_states', None)
                if next_states is not None:
                    next_states = next_states.to(device)
                next_actions = batch.get('next_actions', None)
                if next_actions is not None:
                    next_actions = next_actions.to(device)
                next_phases = batch.get('next_phases', None) 
                if next_phases is not None:
                    next_phases = next_phases.to(device)
                next_rewards = batch.get('next_rewards', None)
                if next_rewards is not None:
                    next_rewards = {k: v.to(device) for k, v in next_rewards.items()}
                future_actions = batch.get('future_actions', None)
                if future_actions is not None:
                    future_actions = future_actions.to(device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # 1. Basic Evaluation - Single-step next state prediction
                outputs = model(
                    current_state=current_states,
                    next_state=next_states,
                    next_rewards=next_rewards,
                    next_actions=next_actions,
                    next_phases=next_phases,
                    attention_mask=attention_mask
                )
                
                # Calculate metrics for this batch
                # State prediction error (MSE)
                if '_z_hat' in outputs and next_states is not None:
                    state_pred_error = ((outputs['_z_hat'] - next_states) ** 2).mean().item()
                    video_metrics['state_pred_error'] += state_pred_error
                    video_state_mse.append(state_pred_error)
                
                # Action prediction accuracy (if applicable)
                if model.imitation_learning and '_a' in model.heads:
                    if 'head_outputs' in outputs and '_a' in outputs['head_outputs']:
                        action_logits = outputs['head_outputs']['_a']
                        action_probs = torch.sigmoid(action_logits)
                        pred_actions = (action_probs > 0.5).float()
                        
                        if next_actions is not None:
                            action_accuracy = (pred_actions == next_actions).float().mean().item()
                            video_metrics['action_pred_accuracy'] += action_accuracy
                            
                            # Store for more detailed metrics
                            video_action_preds.append(pred_actions.cpu().numpy())
                            video_action_gts.append(next_actions.cpu().numpy())
                
                # Add total loss
                if 'total_loss' in outputs:
                    video_metrics['total_loss'] += outputs['total_loss'].item()
                
                # For generation and full evaluation modes, perform additional evaluations
                if eval_mode in ['generation', 'full']:

                    # Get frame indices
                    frame_indices = batch.get('frame_indices', None)
                    if frame_indices is None:
                        start_idx = batch_idx * batch['current_states'].size(0)
                        frame_indices = torch.arange(start_idx, start_idx + batch['current_states'].size(0))
                    
                    # Generate once with maximum horizon
                    max_horizon = max(horizons)
                    rollout = model.generate_conditional_future_states(
                        input_embeddings=current_states,
                        input_actions=next_actions if next_actions is not None else None,
                        horizon=max_horizon,
                        temperature=0.7,
                        use_past=True
                    )
    
                    # Extract predictions for all horizons at once
                    if 'head_outputs' in rollout and '_a' in rollout['head_outputs']:
                        action_probs_all = torch.sigmoid(rollout['head_outputs']['_a'])  # [batch_size, seq_length, num_classes]
                        
                        # Get ground truth
                        if 'future_actions' in batch:
                            gt_actions_all = batch['future_actions']  # [batch_size, seq_length, num_classes]
                            
                            # For each frame in the batch
                            for i, frame_idx in enumerate(frame_indices):
                                frame_idx = frame_idx.item()
                                if frame_idx not in metrics_per_frame[video_id]:
                                    metrics_per_frame[video_id][frame_idx] = {}
                                
                                # For each horizon, compute metrics
                                for h_idx, h in enumerate(horizons):
                                    if h <= action_probs_all.size(1) and h <= gt_actions_all.size(1):
                                        try:
                                            # Get prediction and ground truth for this frame at this horizon
                                            pred_probs = action_probs_all[i, h-1]
                                            gt = gt_actions_all[i, h-1]
                                            
                                            # Calculate mAP only for classes with positive examples
                                            ap_scores = []
                                            for c in range(gt.size(0)):
                                                if gt[c].sum() > 0:  # Check if there are positive examples
                                                    try:
                                                        class_ap = average_precision_score(
                                                            gt[c].cpu().unsqueeze(0).numpy(),
                                                            pred_probs[c].cpu().unsqueeze(0).numpy()
                                                        )
                                                        ap_scores.append(class_ap)
                                                    except:
                                                        continue
                                            
                                            # Store mAP
                                            map_score = np.mean(ap_scores) if ap_scores else np.nan  # Use NaN for visualization if no scores
                                            metrics_per_frame[video_id][frame_idx][h] = {'mAP': map_score}
                                        except Exception as e:
                                            logger.warning(f"Error calculating metrics for frame {frame_idx}, horizon {h}: {str(e)}")
                                            # Set to NaN to indicate missing data in visualization
                                            metrics_per_frame[video_id][frame_idx][h] = {'mAP': np.nan}

                    # # 2. Generate rollout trajectory for evaluation
                    # rollout = model.generate_conditional_future_states(
                    #     input_embeddings=current_states,
                    #     input_actions=next_actions if next_actions is not None else None,
                    #     horizon=rollout_horizon,
                    #     temperature=0.7,
                    #     use_past=True
                    # )
                    
                    # Extract generated trajectory
                    gen_states = rollout['full_embeddings'][:, -rollout_horizon:]  # [batch_size, rollout_horizon, embedding_dim]
                    gen_actions = rollout.get('full_actions', None)
                    
                    # Calculate rollout error over timestep (if future states available)
                    if future_states is not None:
                        rollout_errors = []
                        for t in range(min(gen_states.size(1), future_states.size(1))):
                            error_t = ((future_states[:, t] - gen_states[:, t]) ** 2).mean().item()
                            rollout_errors.append(error_t)
                        
                        # Aggregate rollout metrics
                        if rollout_errors:
                            video_metrics['rollout_error_mean'] = sum(rollout_errors) / len(rollout_errors)
                            video_metrics['rollout_error_final'] = rollout_errors[-1]
                            
                            # Calculate error growth rate (how quickly errors accumulate)
                            if len(rollout_errors) > 1:
                                error_growth = (rollout_errors[-1] / (rollout_errors[0] + 1e-8))
                                video_metrics['rollout_error_growth'] = error_growth
                    
                    # Store a few generated trajectories for later analysis
                    if batch_idx < 2:  # Only store from first 2 batches
                        for i in range(min(2, current_states.size(0))):  # Only store 2 examples per batch
                            trajectory_data = {
                                'video_id': video_id,
                                'batch_idx': batch_idx,
                                'example_idx': i,
                                'generated': {
                                    'states': gen_states[i].cpu(),
                                    'actions': gen_actions[i].cpu() if gen_actions is not None else None
                                },
                                'ground_truth': {
                                    'states': future_states[i].cpu() if future_states is not None else None,
                                    'actions': future_actions[i].cpu() if future_actions is not None else None
                                }
                            }
                            generated_trajectories.append(trajectory_data)
                
                # For full evaluation mode, perform auto-regressive multi-horizon evaluation
                if eval_mode == 'full' and future_states is not None and future_actions is not None:
                    # 3. Auto-regressive multi-horizon prediction
                    horizon_outputs = evaluate_auto_regressive_horizons(
                        model, current_states, future_states, future_actions, 
                        horizons, device
                    )
                    
                    # Store horizon metrics
                    for h in horizons:
                        if h in horizon_outputs:
                            video_horizon_metrics[h]['action_preds'].append(
                                horizon_outputs[h]['action_preds'].cpu().numpy())
                            video_horizon_metrics[h]['action_gts'].append(
                                future_actions[:, min(h-1, future_actions.size(1)-1)].cpu().numpy())
                            video_horizon_metrics[h]['state_mse'].append(
                                horizon_outputs[h]['state_mse'].cpu().numpy())    
            
            # Calculate average metrics for this video
            for key in video_metrics:
                if key not in ['rollout_error_mean', 'rollout_error_final', 'rollout_error_growth']:
                    video_metrics[key] /= max(num_batches, 1)
            
            # Calculate detailed action metrics if we have predictions
            if video_action_preds and video_action_gts:
                # Concatenate all predictions and ground truth for this video
                video_action_preds_concat = np.vstack(video_action_preds)
                video_action_gts_concat = np.vstack(video_action_gts)

                # Make sure preds have the same shape as gts
                # If preds has 3 dimensions and dim[1] == 1, squeeze it if gts has 2 dimensions
                if len(video_action_preds_concat.shape) == 3 and video_action_preds_concat.shape[1] == 1:
                    video_action_preds_concat = video_action_preds_concat.squeeze(1)
                
                # Calculate action metrics
                action_metrics = calculate_action_metrics(video_action_preds_concat, video_action_gts_concat)
                metrics['action']['per_video'][video_id] = action_metrics
            
            # Store state prediction metrics
            if video_state_mse:
                metrics['state']['per_video'][video_id] = {
                    'mse': float(np.mean(video_state_mse)),
                    'num_samples': len(video_state_mse)
                }
            
            # Store rollout metrics
            if 'rollout_error_mean' in video_metrics:
                metrics['rollout']['per_video'][video_id] = {
                    'mean_error': float(video_metrics['rollout_error_mean']),
                    'growth_rate': float(video_metrics['rollout_error_growth']) if 'rollout_error_growth' in video_metrics else 0.0,
                    'final_error': float(video_metrics['rollout_error_final']) if 'rollout_error_final' in video_metrics else 0.0
                }
            
            # Process horizon metrics for full evaluation
            if eval_mode == 'full':
                for h in horizons:
                    if (video_horizon_metrics[h]['action_preds'] and 
                        video_horizon_metrics[h]['action_gts'] and 
                        video_horizon_metrics[h]['state_mse']):
                        
                        # Concatenate horizon metrics
                        h_action_preds = np.vstack(video_horizon_metrics[h]['action_preds'])
                        h_action_gts = np.vstack(video_horizon_metrics[h]['action_gts'])
                        h_state_mse = np.concatenate(video_horizon_metrics[h]['state_mse'])
                        
                        # Make sure preds have the same shape as gts
                        # If preds has 3 dimensions and dim[1] == 1, squeeze it if gts has 2 dimensions
                        if len(h_action_preds.shape) == 3 and h_action_preds.shape[1] == 1:
                            h_action_preds = h_action_preds.squeeze(1)

                        # Calculate action metrics for this horizon
                        h_action_metrics = calculate_action_metrics(h_action_preds, h_action_gts)
                        metrics['action']['horizon'][h][video_id] = h_action_metrics
                        
                        # Calculate state MSE for this horizon
                        h_avg_mse = np.mean(h_state_mse)
                        metrics['state']['horizon'][h][video_id] = {
                            'mse': float(h_avg_mse),
                            'num_samples': len(h_state_mse)
                        }
            
            # Log video metrics
            logger.info(f"[EVAL] Video {video_id} | "
                      f"State Pred MSE: {video_metrics.get('state_pred_error', 0):.4f} | "
                      f"Action Pred Accuracy: {video_metrics.get('action_pred_accuracy', 0):.4f} | "
                      f"Total Loss: {video_metrics.get('total_loss', 0):.4f}")
            
            if eval_mode in ['generation', 'full']:
                logger.info(f"[EVAL] Video {video_id} Rollout | "
                          f"Mean Error: {video_metrics.get('rollout_error_mean', 0):.4f} | "
                          f"Growth Rate: {video_metrics.get('rollout_error_growth', 0):.4f}")
    
    # Calculate overall metrics
    
    # Action metrics
    if metrics['action']['per_video']:
        calculate_overall_metrics(metrics['action'])
        logger.info(f"[EVAL] Overall Action - Acc: {metrics['action']['overall']['accuracy']:.4f}, "
                  f"mAP: {metrics['action']['overall']['mAP']:.4f}")
    
    # State metrics
    if metrics['state']['per_video']:
        metrics['state']['overall']['mse'] = weighted_average(
            [metrics['state']['per_video'][v]['mse'] for v in metrics['state']['per_video']],
            [metrics['state']['per_video'][v]['num_samples'] for v in metrics['state']['per_video']]
        )
        logger.info(f"[EVAL] Overall State MSE: {metrics['state']['overall']['mse']:.4f}")
    
    # Rollout metrics
    if metrics['rollout']['per_video']:
        metrics['rollout']['overall']['mean_error'] = np.mean([
            metrics['rollout']['per_video'][v]['mean_error'] for v in metrics['rollout']['per_video']
        ])
        metrics['rollout']['overall']['growth_rate'] = np.mean([
            metrics['rollout']['per_video'][v]['growth_rate'] for v in metrics['rollout']['per_video']
        ])
        logger.info(f"[EVAL] Overall Rollout - Mean Error: {metrics['rollout']['overall']['mean_error']:.4f}, "
                  f"Growth Rate: {metrics['rollout']['overall']['growth_rate']:.4f}")
    
    # Horizon metrics for full evaluation
    if eval_mode == 'full':
        for h in horizons:
            if metrics['action']['horizon'][h]:
                action_h_overall = {}
                calculate_overall_metrics({'per_video': metrics['action']['horizon'][h], 'overall': action_h_overall})
                metrics['action']['horizon'][h]['overall'] = action_h_overall
                
                state_h_overall = {'mse': weighted_average(
                    [metrics['state']['horizon'][h][v]['mse'] for v in metrics['state']['horizon'][h]],
                    [metrics['state']['horizon'][h][v]['num_samples'] for v in metrics['state']['horizon'][h]]
                )}
                metrics['state']['horizon'][h]['overall'] = state_h_overall
                
                logger.info(f"[EVAL] Horizon {h} - Action Acc: {action_h_overall['accuracy']:.4f}, "
                          f"mAP: {action_h_overall['mAP']:.4f}, State MSE: {state_h_overall['mse']:.4f}")

    # Add overall metrics to the main metrics dictionary
    if overall_h in metrics['action']['horizon'] and 'overall' in metrics['action']['horizon'][overall_h]:
        metrics['action']['overall'] = metrics['action']['horizon'][overall_h]['overall']


    # Generate visualizations and save results if specified
    if save_results and results_dir:
        # Save metrics
        metrics_path = os.path.join(results_dir, "metrics.json")
        save_json(metrics, metrics_path)
        logger.info(f"Saved metrics to {metrics_path}")
        
        # Save generated trajectories if any
        if generated_trajectories:
            trajectories_path = os.path.join(results_dir, "generated_trajectories.pt")
            torch.save(generated_trajectories, trajectories_path)
            logger.info(f"Saved generated trajectories to {trajectories_path}")
        
        # Generate visualizations if specified
        if render_visualizations:
            if eval_mode == 'full':
                visualize_results(metrics, os.path.join(results_dir, "visualizations"), logger)
            
            # Generate the heatmap plots
            if save_results and eval_mode == 'full':
                plot_map_anticipation_heatmap(
                    metrics_per_frame, 
                    video_ids=list(test_video_loaders.keys()),
                    horizons=horizons,
                    save_dir=os.path.join(results_dir, "visualizations/anticipation_heatmaps"),
                    logger=logger
                )
                logger.info(f"Generated anticipation heatmaps for {len(metrics_per_frame)} videos")
                
                # If you have position-specific metrics (would need to modify the evaluation logic)
                # plot_map_heatmap_per_video(metrics_per_position, video_lengths, horizons, 
                #                            save_dir=os.path.join(results_dir, "visualizations/heatmaps"),
                #                            logger=logger)
                # logger.info(f"Generated mAP heatmaps for {len(metrics_per_position)} videos")
            
            # Visualize sample trajectories
            if generated_trajectories:
                for i, traj in enumerate(generated_trajectories[:5]):  # Only visualize first 5
                    if traj['generated']['states'] is not None and traj['ground_truth']['states'] is not None:
                        visualize_sample_rollout(
                            traj['generated'], traj['ground_truth']['states'],
                            os.path.join(results_dir, f"{traj['video_id']}_sample_{i}.png"),
                            logger, title=f"Video {traj['video_id']} - Sample {i}"
                        )
            
            # Generate summary report for full evaluation
            if eval_mode == 'full':
                generate_summary_report(metrics, os.path.join(results_dir, "summary.md"), logger)
    
    return metrics

def evaluate_auto_regressive_horizons(
    model: nn.Module, 
    current_states: torch.Tensor,
    future_states: torch.Tensor, 
    future_actions: torch.Tensor, 
    horizons: List[int],
    device: str
) -> Dict[int, Dict[str, torch.Tensor]]:
    """
    Evaluate model predictions at different horizons using auto-regressive prediction.
    
    At each step, the model uses its own previous predictions as input.

    Optionally uses the first future action as input for the first step.
    
    Args:
        model: WorldModel instance
        current_states: Current state tensor [batch_size, context_length, embedding_dim]
        future_states: Future state tensor [batch_size, future_length, embedding_dim]
        future_actions: Future action tensor [batch_size, future_length, action_dim]
        horizons: List of horizons to evaluate at
        device: Device to evaluate on
        
    Returns:
        Dictionary of horizon-specific metrics
    """
    max_horizon = max(horizons)
    batch_size = current_states.size(0)
    
    # Initialize output dictionary
    horizon_outputs = {h: {} for h in horizons}
    
    # Initial state is the last frame in the context window
    state = current_states[:, -1:].clone()  # [batch_size, 1, embedding_dim]
    
    # Predictions and ground truth for each horizon
    all_action_preds = []
    all_state_preds = []
    
    # Auto-regressive rollout
    for step in range(max_horizon):
        # Forward pass
        with torch.no_grad():
            outputs = model(
                current_state=state, 
                next_state=future_states[:, step:step+1],
                next_actions=future_actions[:, step:step+1] if step < future_actions.size(1) else None,
                attention_mask=None)
        
            # outputs = model(
            #     current_state=current_states,
            #     next_state=next_states,
            #     next_rewards=next_rewards,
            #     next_actions=next_actions,
            #     next_phases=next_phases,
            #     attention_mask=attention_mask
            # )

        # Extract predictions
        if '_a_hat' in outputs:
            action_probs = torch.sigmoid(outputs['_a_hat'])
            action_preds = (action_probs > 0.5).float()
        else:
            action_probs = outputs.get('head_outputs', {}).get('_a', None)
            if action_probs is not None:
                action_preds = (torch.sigmoid(action_probs) > 0.5).float()
            else:
                action_preds = None
        
        # Extract state predictions
        state_preds = outputs.get('_z_hat', None)
        
        # Store predictions
        if action_preds is not None:
            all_action_preds.append(action_preds)
        if state_preds is not None:
            all_state_preds.append(state_preds)
            # Use predictions as next input (auto-regressive)
            # Add time dimension if not already present
            if len(state_preds.size()) == 2:
                state = state_preds.unsqueeze(1)
            else:
                state = state_preds  
        
        # Store metrics for horizons we care about
        horizon = step + 1
        if horizon in horizons:
            # Store action predictions
            if action_preds is not None:
                horizon_outputs[horizon]['action_preds'] = action_preds
            
            # Calculate state prediction MSE
            if state_preds is not None and horizon < future_states.size(1)+1:
                state_mse = torch.mean((state_preds - future_states[:, horizon-1]) ** 2, dim=1)
                horizon_outputs[horizon]['state_mse'] = state_mse
    
    return horizon_outputs

def calculate_action_metrics(predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Any]:
    """
    Calculate comprehensive metrics for action prediction.
    
    Args:
        predictions: Binary predictions [num_samples, num_classes]
        ground_truth: Binary ground truth [num_samples, num_classes]
        
    Returns:
        Dictionary of metrics
    """
    # Calculate accuracy (exact match across all classes)
    exact_match = np.all(predictions == ground_truth, axis=1)
    accuracy = np.mean(exact_match)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth.flatten(), predictions.flatten(), average='binary'
    )
    
    # Calculate mAP for each action class
    ap_scores = []
    per_class_metrics = []
    
    for i in range(ground_truth.shape[1]):
        # Only calculate AP if class is present
        class_metrics = {}
        
        if len(np.unique(ground_truth[:, i])) > 1:
            class_precision, class_recall, class_f1, _ = precision_recall_fscore_support(
                ground_truth[:, i], predictions[:, i], average='binary'
            )
            
            try:
                class_ap = average_precision_score(ground_truth[:, i], predictions[:, i])
                ap_scores.append(class_ap)
                
                class_metrics = {
                    'precision': float(class_precision),
                    'recall': float(class_recall),
                    'f1': float(class_f1),
                    'ap': float(class_ap),
                    'support': int(np.sum(ground_truth[:, i]))
                }
            except:
                # If calculation fails (e.g., only one class present)
                class_metrics = {
                    'precision': float(class_precision),
                    'recall': float(class_recall),
                    'f1': float(class_f1),
                    'ap': 0.0,
                    'support': int(np.sum(ground_truth[:, i]))
                }
        else:
            # If class is not present or only one class value exists
            class_metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'ap': 0.0,
                'support': int(np.sum(ground_truth[:, i]))
            }
        
        per_class_metrics.append(class_metrics)
    
    # Overall mAP
    mAP = np.mean(ap_scores) if ap_scores else 0.0
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'mAP': float(mAP),
        'per_class': per_class_metrics,
        'num_samples': int(predictions.shape[0])
    }

def calculate_overall_metrics(metrics_dict: Dict[str, Any]) -> None:
    """
    Calculate overall metrics from per-video metrics, weighted by sample count.
    
    Args:
        metrics_dict: Dictionary with 'per_video' and 'overall' keys
        
    Updates the 'overall' dictionary in place.
    """
    if not metrics_dict['per_video']:
        return
    
    # Get all video IDs
    video_ids = list(metrics_dict['per_video'].keys())
    
    # Calculate total number of samples
    total_samples = sum(metrics_dict['per_video'][v]['num_samples'] for v in video_ids)
    
    if total_samples == 0:
        return
    
    # Initialize overall metrics
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'mAP']:
        if metric not in metrics_dict['overall']:
            metrics_dict['overall'][metric] = 0.0
    
    # Calculate weighted averages
    for metric in metrics_dict['overall'].keys():
        if all(metric in metrics_dict['per_video'][v] for v in video_ids):
            metrics_dict['overall'][metric] = weighted_average(
                [metrics_dict['per_video'][v][metric] for v in video_ids],
                [metrics_dict['per_video'][v]['num_samples'] for v in video_ids]
            )

def weighted_average(values: List[float], weights: List[int]) -> float:
    """
    Calculate weighted average.
    
    Args:
        values: List of values
        weights: List of weights (same length as values)
        
    Returns:
        Weighted average as a float
    """
    if not values or not weights or sum(weights) == 0:
        return 0.0
    
    return float(np.average(values, weights=weights))

def visualize_sample_rollout(
    generated: Dict[str, torch.Tensor],
    ground_truth: torch.Tensor,
    save_path: str,
    logger: Any,
    title: str = None
) -> None:
    """
    Visualize a comparison between predicted and ground truth trajectories.
    
    Args:
        generated: Dictionary with model generated outputs
        ground_truth: Ground truth future states tensor
        save_path: Path to save visualization
        logger: Logger instance
        title: Optional title for the plot
    """
    # Extract predicted trajectory
    if 'states' in generated:
        pred_trajectory = generated['states'].cpu().numpy()
        if len(pred_trajectory.shape) == 3:
            pred_trajectory = pred_trajectory.squeeze(0)
        
        # Extract ground truth trajectory
        gt_trajectory = ground_truth.cpu().numpy()
        if len(gt_trajectory.shape) == 3:
            gt_trajectory = gt_trajectory.squeeze(0)
        
        # Only compare up to the minimum length
        min_length = min(pred_trajectory.shape[0], gt_trajectory.shape[0])
        pred_trajectory = pred_trajectory[:min_length]
        gt_trajectory = gt_trajectory[:min_length]
        
        # Apply PCA for visualization
        combined = np.vstack([pred_trajectory, gt_trajectory])
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(combined)
        
        # Split back into predicted and ground truth
        pred_pca = pca_result[:len(pred_trajectory)]
        gt_pca = pca_result[len(pred_trajectory):]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Plot predicted trajectory
        plt.scatter(pred_pca[:, 0], pred_pca[:, 1], 
                   c=range(len(pred_pca)), cmap='viridis', 
                   s=100, marker='o', label='Predicted')
        
        # Connect with arrows
        for i in range(len(pred_pca) - 1):
            plt.arrow(pred_pca[i, 0], pred_pca[i, 1], 
                     pred_pca[i+1, 0] - pred_pca[i, 0], 
                     pred_pca[i+1, 1] - pred_pca[i, 1],
                     head_width=0.05, head_length=0.08, fc='blue', ec='blue', alpha=0.6)
        
        # Plot ground truth trajectory
        plt.scatter(gt_pca[:, 0], gt_pca[:, 1], 
                   c=range(len(gt_pca)), cmap='viridis', 
                   s=100, marker='x', label='Ground Truth')
        
        # Connect with arrows
        for i in range(len(gt_pca) - 1):
            plt.arrow(gt_pca[i, 0], gt_pca[i, 1], 
                     gt_pca[i+1, 0] - gt_pca[i, 0], 
                     gt_pca[i+1, 1] - gt_pca[i, 1],
                     head_width=0.05, head_length=0.08, fc='green', ec='green', alpha=0.6)
        
        if title:
            plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        
        logger.info(f"Saved rollout visualization to {save_path}")

def visualize_results(metrics: Dict[str, Any], save_dir: str, logger: Any) -> None:
    """
    Create visualizations for the evaluation results.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_dir: Directory to save visualizations
        logger: Logger instance
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Action metrics summary
    if 'action' in metrics and 'overall' in metrics['action']:
        plt.figure(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'mAP']
        metric_values = [
            metrics['action']['overall']['accuracy'],
            metrics['action']['overall']['precision'],
            metrics['action']['overall']['recall'],
            metrics['action']['overall']['f1'],
            metrics['action']['overall']['mAP']
        ]
        
        plt.bar(metric_names, metric_values)
        plt.ylim(0, 1)
        plt.title('Overall Action Prediction Metrics')
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(save_dir, 'action_metrics.png'))
        plt.close()
    
    # 2. Action metrics across horizons
    if 'action' in metrics and 'horizon' in metrics['action']:
        horizons = sorted(metrics['action']['horizon'].keys())
        
        metrics_to_plot = ['accuracy', 'mAP']
        for metric_name in metrics_to_plot:
            metric_values = [metrics['action']['horizon'][h]['overall'][metric_name] 
                            for h in horizons if 'overall' in metrics['action']['horizon'][h]]
            
            if metric_values:
                plt.figure(figsize=(10, 6))
                plt.plot(horizons[:len(metric_values)], metric_values, 'o-', linewidth=2, markersize=8)
                plt.xlabel('Prediction Horizon')
                plt.ylabel(metric_name.capitalize())
                plt.title(f'Action {metric_name.capitalize()} vs. Prediction Horizon')
                plt.grid(alpha=0.3)
                plt.ylim(0, 1)
                plt.xticks(horizons[:len(metric_values)])
                plt.savefig(os.path.join(save_dir, f'action_{metric_name}_vs_horizon.png'))
                plt.close()
    
    # 3. State MSE across horizons
    if 'state' in metrics and 'horizon' in metrics['state']:
        horizons = sorted(metrics['state']['horizon'].keys())
        
        mse_values = [metrics['state']['horizon'][h]['overall']['mse'] 
                     for h in horizons if 'overall' in metrics['state']['horizon'][h]]
        
        if mse_values:
            plt.figure(figsize=(10, 6))
            plt.plot(horizons[:len(mse_values)], mse_values, 'o-', linewidth=2, markersize=8)
            plt.xlabel('Prediction Horizon')
            plt.ylabel('MSE')
            plt.title('State Prediction Error vs. Horizon')
            plt.grid(alpha=0.3)
            plt.xticks(horizons[:len(mse_values)])
            plt.savefig(os.path.join(save_dir, 'state_mse_vs_horizon.png'))
            plt.close()
    
    logger.info(f"Saved visualizations to {save_dir}")

def generate_summary_report(metrics: Dict[str, Any], save_path: str, logger: Any) -> None:
    """
    Generate a summary report of evaluation results.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_path: Path to save the report
        logger: Logger instance
    """
    # Create markdown report
    report = "# World Model Inference Evaluation Summary\n\n"
    
    # Add overall action metrics
    if 'action' in metrics and 'overall' in metrics['action']:
        report += "## Action Prediction Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        report += f"| Accuracy | {metrics['action']['overall']['accuracy']:.4f} |\n"
        report += f"| Precision | {metrics['action']['overall']['precision']:.4f} |\n"
        report += f"| Recall | {metrics['action']['overall']['recall']:.4f} |\n"
        report += f"| F1 Score | {metrics['action']['overall']['f1']:.4f} |\n"
        report += f"| Mean Average Precision (mAP) | {metrics['action']['overall']['mAP']:.4f} |\n\n"
    
    # Add state prediction metrics
    if 'state' in metrics and 'overall' in metrics['state']:
        report += "## State Prediction Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        report += f"| Mean Squared Error (MSE) | {metrics['state']['overall']['mse']:.4f} |\n\n"
    
    # Add rollout metrics
    if 'rollout' in metrics and 'overall' in metrics['rollout']:
        report += "## Rollout Prediction Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        report += f"| Mean Error | {metrics['rollout']['overall']['mean_error']:.4f} |\n"
        report += f"| Growth Rate | {metrics['rollout']['overall']['growth_rate']:.4f} |\n\n"
    
    # Add horizon metrics
    if 'action' in metrics and 'horizon' in metrics['action']:
        horizons = sorted(metrics['action']['horizon'].keys())
        valid_horizons = [h for h in horizons if 'overall' in metrics['action']['horizon'][h]]
        
        if valid_horizons:
            report += "## Multi-Horizon Prediction Metrics\n\n"
            report += "| Horizon | Action Accuracy | Action mAP | State MSE |\n"
            report += "|---------|----------------|------------|----------|\n"
            
            for h in valid_horizons:
                action_acc = metrics['action']['horizon'][h]['overall']['accuracy']
                action_map = metrics['action']['horizon'][h]['overall']['mAP']
                state_mse = metrics['state']['horizon'][h]['overall']['mse']
                
                report += f"| {h} | {action_acc:.4f} | {action_map:.4f} | {state_mse:.4f} |\n"
            
            report += "\n"
    
    # Add analysis
    report += "## Analysis\n\n"
    
    # Action prediction analysis
    if 'action' in metrics and 'overall' in metrics['action']:
        mAP = metrics['action']['overall']['mAP']
        if mAP > 0.8:
            report += "The model demonstrates excellent action prediction performance with high mAP, "
            report += "suggesting strong accuracy in identifying the correct surgical actions.\n\n"
        elif mAP > 0.6:
            report += "The model shows good action prediction performance, but there's room for improvement "
            report += "in certain action classes.\n\n"
        else:
            report += "The model's action prediction performance could be improved. Consider further training "
            report += "or exploring different model architectures.\n\n"
    
    # Horizon analysis
    if 'action' in metrics and 'horizon' in metrics['action']:
        horizons = sorted(metrics['action']['horizon'].keys())
        valid_horizons = [h for h in horizons if 'overall' in metrics['action']['horizon'][h]]
        
        if valid_horizons and len(valid_horizons) > 1:
            first_h = valid_horizons[0]
            last_h = valid_horizons[-1]
            
            first_map = metrics['action']['horizon'][first_h]['overall']['mAP']
            last_map = metrics['action']['horizon'][last_h]['overall']['mAP']
            
            map_drop = first_map - last_map
            
            if map_drop < 0.1:
                report += "The model maintains consistent performance over longer prediction horizons, "
                report += "indicating strong temporal modeling capabilities.\n\n"
            elif map_drop < 0.3:
                report += "The model shows moderate degradation in performance over longer prediction horizons, "
                report += "which is expected for auto-regressive predictions.\n\n"
            else:
                report += "The model's performance degrades significantly over longer prediction horizons. "
                report += "This suggests accumulated errors in auto-regressive prediction. Consider techniques "
                report += "like scheduled sampling during training to improve long-horizon prediction.\n\n"
    
    # Save report
    with open(save_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved summary report to {save_path}")

def save_json(data: Dict[str, Any], path: str) -> None:
    """
    Save data as JSON, handling non-serializable types.
    
    Args:
        data: Data to save
        path: Path to save to
    """
    import json
    
    # Convert data to serializable types
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # Make data serializable
    serializable_data = make_serializable(data)
    
    # Save to file
    with open(path, 'w') as f:
        json.dump(serializable_data, f, indent=2)


def plot_map_anticipation_heatmap(metrics_per_frame, 
                                 video_ids, 
                                 horizons, 
                                 save_dir=None, 
                                 logger=None):
    """
    Create a heatmap showing mAP scores for each frame position and anticipation horizon.
    
    Args:
        metrics_per_frame: Dictionary with structure:
            {video_id: {frame_idx: {horizon: {'mAP': score}}}}
        video_ids: List of video IDs to generate plots for
        horizons: List of anticipation horizons used
        save_dir: Directory to save the plots
        logger: Optional logger instance
        
    Returns:
        Dictionary of matplotlib figures keyed by video_id
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    from matplotlib.colors import LinearSegmentedColormap
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Custom colormap - from blue (low) to green (medium) to red (high)
    colors = [(0, 0, 0.8), (0, 0.8, 0), (0.8, 0, 0)]  # blue, green, red
    cmap_name = 'map_performance'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    
    figures = {}
    
    for video_id in video_ids:
        if video_id not in metrics_per_frame:
            if logger:
                logger.warning(f"No data for video {video_id}")
            continue
            
        video_data = metrics_per_frame[video_id]
        frame_indices = sorted(list(video_data.keys()))
        num_frames = len(frame_indices)
        
        if num_frames == 0:
            if logger:
                logger.warning(f"No frames with metrics for video {video_id}")
            continue
        
        # Create a matrix for the heatmap [horizons × frames]
        map_matrix = np.zeros((len(horizons), num_frames))
        map_matrix[:] = np.nan  # Initialize with NaN for missing values
        
        # Fill the matrix with mAP scores
        for h_idx, horizon in enumerate(horizons):
            for f_idx, frame_idx in enumerate(frame_indices):
                if horizon in video_data[frame_idx]:
                    map_score = video_data[frame_idx][horizon].get('mAP', np.nan)
                    map_matrix[h_idx, f_idx] = map_score
        
        # Create figure
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot as scatter points with color representing mAP
        x, y = np.meshgrid(frame_indices, horizons)
        scatter = ax.scatter(x, y, c=map_matrix, cmap=cm, 
                             s=50, marker='s', alpha=0.8, vmin=0, vmax=1)
        
        # Set up axes and labels
        ax.set_xlabel('Frame Position in Video', fontsize=12)
        ax.set_ylabel('Anticipation Horizon (frames ahead)', fontsize=12)
        ax.set_title(f'Action Anticipation Performance (mAP) for Video {video_id}', fontsize=14)
        
        # Configure x-axis
        if num_frames > 20:
            # For many frames, show fewer tick marks
            tick_interval = max(1, num_frames // 10)
            ax.set_xticks(frame_indices[::tick_interval])
            ax.set_xticklabels(frame_indices[::tick_interval])
        else:
            ax.set_xticks(frame_indices)
            ax.set_xticklabels(frame_indices)
        
        # Configure y-axis
        ax.set_yticks(horizons)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('mAP Score', fontsize=12)
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if requested
        if save_dir:
            save_path = os.path.join(save_dir, f'map_anticipation_heatmap_{video_id}.png')
            fig.savefig(save_path)
            if logger:
                logger.info(f"Saved anticipation heatmap for video {video_id} to {save_path}")
        
        figures[video_id] = fig
    
    return figures

def plot_map_heatmap_per_video(metrics_per_position, video_lengths, horizons, save_dir=None, logger=None):
    """
    Create a heatmap showing mAP scores across both video duration and prediction horizons.
    
    Args:
        metrics_per_position: Dictionary mapping video_id to a list of position-specific metrics
                            Each position should have horizon-specific mAP scores
        video_lengths: Dictionary mapping video_id to video length in frames
        horizons: List of prediction horizons used
        save_dir: Directory to save plots
        logger: Optional logger
        
    Returns:
        Dictionary of heatmap figures
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
    figures = {}
    
    for video_id, positions_data in metrics_per_position.items():
        # Extract video length
        video_length = video_lengths.get(video_id, len(positions_data))
        
        # Create matrix for heatmap [horizons × positions]
        num_positions = len(positions_data)
        map_matrix = np.zeros((len(horizons), num_positions))
        
        # Fill matrix with mAP scores
        for pos_idx, pos_metrics in enumerate(positions_data):
            for h_idx, h in enumerate(horizons):
                map_matrix[h_idx, pos_idx] = pos_metrics.get(h, {}).get('mAP', 0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(map_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=1)
        
        # Set up axes
        ax.set_xlabel('Video Duration (frames)')
        ax.set_ylabel('Prediction Horizon')
        ax.set_title(f'mAP Performance Map for Video {video_id}')
        
        # Set ticks
        ax.set_yticks(range(len(horizons)))
        ax.set_yticklabels(horizons)
        
        # X-ticks - show frame numbers
        num_xticks = min(10, num_positions)
        xtick_indices = np.linspace(0, num_positions-1, num_xticks, dtype=int)
        ax.set_xticks(xtick_indices)
        ax.set_xticklabels([f'{int(pos/num_positions*video_length)}' for pos in xtick_indices])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('mAP Score')
        
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f'map_heatmap_{video_id}.png')
            fig.savefig(save_path)
            
        figures[video_id] = fig
    
    return figures
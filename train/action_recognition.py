import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from tqdm import tqdm
import ivtmetrics

# sklearn mean average precision
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

# Custom imports
from utils import calculate_map_recognition

def train_recognition_head(cfg, logger, model, train_loader, val_loader=None, device='cuda'):
    """
    Train the recognition head model.
    
    Args:
        cfg: Configuration dictionary
        logger: Logger object
        model: Recognition head model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        device: Device to train on
        
    Returns:
        Trained model
    """
    # Extract configuration parameters
    learning_rate = cfg['training']['learning_rate']
    num_epochs = cfg['training']['epochs']
    scheduler_cfg = cfg['training']['scheduler']
    
    # Initialize recognition metrics
    recognize = ivtmetrics.Recognition(num_class=100)

    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = scheduler_cfg.get('warmup_steps', 0)
    
    # Using OneCycleLR for cosine with warmup
    pct_start_value = warmup_steps/total_steps if total_steps > 0 else 0.1
    if pct_start_value > 1.0:
        logger.info(f"Clamping pct_start from {pct_start_value:.4f} to 1.0 (warmup_steps={warmup_steps}, total_steps={total_steps})")
        pct_start_value = 1.0
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        total_steps=total_steps,
        pct_start=pct_start_value,
        anneal_strategy='cos',
        div_factor=25.0,  # initial_lr = max_lr/div_factor
        final_div_factor=10000.0,  # final_lr = initial_lr/final_div_factor
    )
    logger.info(f"Initialized OneCycleLR scheduler with pct_start={pct_start_value:.4f}, warmup steps: {warmup_steps}")
    
    # Track current step globally to avoid exceeding total steps
    current_step = 0
    
    # Create checkpoint directory
    checkpoint_dir = os.path.join(logger.log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Tensorboard for logging
    tensorboard_dir = os.path.join(logger.log_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Track best model
    best_val_map = 0.0
    best_model_path = None
    
    # Load class labels for logging
    with open(cfg['data']['paths']['class_labels_file_path'], 'r') as f:
        class_labels = json.load(f)
    action_labels = [class_name for class_id, class_name in class_labels['action'].items()]
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0

        all_preds_actions = []
        all_targets_actions = []
        all_preds_instruments = []
        all_targets_instruments = []
        # Initialize recognition metrics
        recognize.reset()

        for batch_idx, (z_seq, _, _, _, c_a, c_i) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")):
            z_seq, c_a, c_i = z_seq.to(device), c_a.to(device), c_i.to(device)
            
            # Forward pass
            outputs = model(z_seq)
            action_logits = outputs['action_logits']
            instrument_logits = outputs['instrument_logits']
            
            # For simplicity, we'll use the last frame's actions as the target
            # In a real scenario, you might want to use all frames or a specific target
            target_actions = c_a[:, :]
            action_probs = torch.sigmoid(action_logits)
            target_instruments = c_i[:, :]
            instrument_probs = torch.sigmoid(instrument_logits)

            # Store predictions and targets for mAP calculation
            all_preds_actions.append(action_probs.detach().cpu().numpy())
            all_targets_actions.append(target_actions.detach().cpu().numpy())

            if outputs.get('instrument_logits') is not None:
                instrument_probs = torch.sigmoid(outputs['instrument_logits'])
                all_preds_instruments.append(instrument_probs.detach().cpu().numpy())
                all_targets_instruments.append(target_instruments.detach().cpu().numpy())

            # Calculate loss
            loss_c_a = criterion(action_logits, target_actions)
            loss_c_i = criterion(instrument_logits, target_instruments)
            loss = loss_c_a + loss_c_i
            # Normalize loss by batch size
            # loss = loss / z_seq.size(0)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Only step the scheduler if we haven't reached total_steps
            if current_step < total_steps - 1:
                scheduler.step()
                current_step += 1
            else:
                logger.info(f"Skipping scheduler step as we've reached the total steps: {current_step} >= {total_steps-1}")
            
            # Update statistics
            train_loss += loss.item() * z_seq.size(0)

            # Add to recognition metrics
            recognize.update(target_actions.cpu().detach().numpy(), action_probs.cpu().detach().numpy())
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            writer.add_scalar('Loss/Train_Action', loss_c_a.item(), global_step)
            writer.add_scalar('Loss/Train_Instrument', loss_c_i.item(), global_step)
            writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step)

            # Log every N steps
            if batch_idx % cfg['training']['log_every_n_steps'] == 0:
                logger.info(f"[TRAIN] Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        # Calculate mAP
        all_preds_actions = np.concatenate(all_preds_actions, axis=0)
        all_targets_actions = np.concatenate(all_targets_actions, axis=0)
        all_preds_instruments = np.concatenate(all_preds_instruments, axis=0)
        all_targets_instruments = np.concatenate(all_targets_instruments, axis=0)
        # Calculate mAP for actions
        action_map_scores = calculate_map_recognition(all_targets_actions, all_preds_actions)
        action_overall_map = action_map_scores['mAP']
        # action_per_class_ap = action_map_scores['AP_scores']
        # Calculate mAP for instruments
        instrument_map_scores = calculate_map_recognition(all_targets_instruments, all_preds_instruments)
        instrument_overall_map = instrument_map_scores['mAP']
        # instrument_per_class_ap = instrument_map_scores['AP_scores']
        # Log mAP to tensorboard
        writer.add_scalar('Metrics/Train_mAP_sklean_Action', action_overall_map, epoch)
        writer.add_scalar('Metrics/Train_mAP_sklearn_Instrument', instrument_overall_map, epoch)
        # Log mAP to console
        logger.info(f"[TRAIN] Epoch {epoch+1}/{num_epochs}, Train mAP: Instrument (sklearn): {instrument_overall_map:.4f}")
        logger.info(f"[TRAIN] Epoch {epoch+1}/{num_epochs}, Train mAP: Action (sklearn): {action_overall_map:.4f}")

        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        logger.info(f"[TRAIN] Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # Calculate mAP using ivtmetrics
        results_i = recognize.compute_AP('i')
        results_ivt = recognize.compute_AP('ivt')

        # Log mAP to tensorboard
        writer.add_scalar('Metrics/Train_mAP_ivt_Action', results_i["mAP"], epoch)
        writer.add_scalar('Metrics/Train_mAP_ivt_Instrument', results_ivt["mAP"], epoch)
        # Log mAP to console
        logger.info(f"[TRAIN] Epoch {epoch+1}/{num_epochs}, Train mAP: Instrument (ivt metrics): {results_i['mAP']:.4f}")
        logger.info(f"[TRAIN] Epoch {epoch+1}/{num_epochs}, Train mAP: Action (ivt metrics): {results_ivt['mAP']:.4f}")

        # Validation phase
        if val_loader is not None:
            model.eval()

            # Eval loop
            eval_results = run_recognition_inference(cfg, logger, model, val_loader, 
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

def run_recognition_inference(cfg, logger, model, test_video_loaders, device='cuda', epoch=None, tb_writer=None, run_type='inference'):
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
    recognize = ivtmetrics.Recognition(num_class=100)
    recognize.reset_global()

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

            for batch_idx, (z_seq, _, _, _, c_a, c_i) in enumerate(tqdm(video_loader, desc="Running inference")):
                z_seq, c_a, c_i = z_seq.to(device), c_a.to(device), c_i.to(device)
                
                # Forward pass
                outputs = model(z_seq)
                action_logits = outputs['action_logits']
                instrument_logits = outputs['instrument_logits']
                
                # Convert logits to probabilities
                action_probs = torch.sigmoid(action_logits)
                instrument_probs = torch.sigmoid(instrument_logits)
            
                target_actions = c_a[:, :]
                target_instruments = c_i[:, :]
                
                # Store predictions and targets
                video_preds_actions.append(action_probs.cpu().numpy())
                video_targets_actions.append(target_actions.cpu().numpy())
                video_preds_instruments.append(instrument_probs.cpu().numpy())
                video_targets_instruments.append(target_instruments.cpu().numpy())

                recognize.update(target_actions.cpu().numpy(), action_probs.cpu().numpy())

                # Log every N steps
                if batch_idx % cfg['training']['log_every_n_steps'] == 0:
                    logger.info(f"{run_string} Video {vid_id} | Batch {batch_idx}/{len(video_loader)}")

            # Add video and init new lists
            recognize.video_end()
        
            # Mean Average Precision (mAP) for actions
            video_targets_actions = np.concatenate(video_targets_actions, axis=0)
            video_preds_actions = np.concatenate(video_preds_actions, axis=0)
            # Calculate mAP using sklearn
            action_map_scores = calculate_map_recognition(video_targets_actions, video_preds_actions, action_labels)
            action_overall_map = action_map_scores['mAP']
            action_per_class_ap = action_map_scores['AP_scores']
            # Mean Average Precision (mAP) for instruments
            video_targets_instruments = np.concatenate(video_targets_instruments, axis=0)
            video_preds_instruments = np.concatenate(video_preds_instruments, axis=0)
            # Calculate mAP using sklearn
            instrument_map_scores = calculate_map_recognition(video_targets_instruments, video_preds_instruments, instrument_labels)
            instrument_overall_map = instrument_map_scores['mAP']
            instrument_per_class_ap = instrument_map_scores['AP_scores']
            # Log mAP to console
            logger.info(f"{run_string} Video {vid_id}: Test mAP (sklearn) Action: {action_overall_map:.4f}")
            logger.info(f"{run_string} Video {vid_id}: Test mAP (sklearn) Instrument: {instrument_overall_map:.4f}")
            # Add to overall results
            videos_scores['sklearn_vids_mAP_a'].append(action_overall_map)
            videos_scores['sklearn_vids_mAP_i'].append(instrument_overall_map)

            # Add the video mAP to the per class mAP
            per_video_sklearn_per_class_map_a[f'video_{vid_id}_sklearn_vids_mAP_a'] = action_per_class_ap
            per_video_sklearn_per_class_map_i[f'video_{vid_id}_sklearn_vids_mAP_i'] = instrument_per_class_ap

        # END VIDEO LOOP

        # sklearn metrics
        action_overall_map = np.mean(videos_scores['sklearn_vids_mAP_a'])
        instrument_overall_map = np.mean(videos_scores['sklearn_vids_mAP_i'])

        # Aggregate all the map scores per class on all videos in per_video_sklearn_per_class_map
        # for each video, add the per class map scores to the per class map scores
        # add the per class map scores to the per class map scores (there are None values)
        per_class_ap_scores['sklearn_vids_mAP_a'] = np.nanmean(np.array(list(per_video_sklearn_per_class_map_a.values())), axis=0).tolist()
        per_class_ap_scores['sklearn_vids_mAP_i'] = np.nanmean(np.array(list(per_video_sklearn_per_class_map_i.values())), axis=0).tolist()

        # Add the overall nan means to the per class map scores based on the intermediate results from the video AP
        per_class_ap_scores['sklearn_mAP_a_from_vid_ap'] = np.nanmean(np.array(per_class_ap_scores['sklearn_vids_mAP_a']), axis=0).tolist()
        per_class_ap_scores['sklearn_mAP_i_from_vid_ap'] = np.nanmean(np.array(per_class_ap_scores['sklearn_vids_mAP_i']), axis=0).tolist()

        # ivt metrics
        results_ivt = recognize.compute_video_AP('ivt', ignore_null=False) # try with ignore null classes
        results_i = recognize.compute_video_AP('i', ignore_null=False) # try with ignore null classes

        # Per class scores (all videos aggragated)
        per_class_ap_scores['ivt_map_a'] = results_ivt['AP'].tolist()
        per_class_ap_scores['ivt_map_i'] = results_i['AP'].tolist()
        per_class_ap_scores['ivt_mAP_a_from_vid_ap'] = np.nanmean(np.array(per_class_ap_scores['ivt_map_a']), axis=0).tolist()
        per_class_ap_scores['ivt_mAP_i_from_vid_ap'] = np.nanmean(np.array(per_class_ap_scores['ivt_map_i']), axis=0).tolist()

        # Save per video mAP results
        overall_results['sklearn_mAP_a_from_vid_ap'] = per_class_ap_scores['sklearn_mAP_a_from_vid_ap']
        overall_results['sklearn_mAP_a'] = np.round(action_overall_map, 4)
        overall_results['sklearn_mAP_i'] = np.round(instrument_overall_map, 4)
        overall_results['ivt_mAP_a_from_vid_ap'] = per_class_ap_scores['ivt_mAP_a_from_vid_ap']
        overall_results['ivt_mAP_a'] = np.round(results_ivt["mAP"], 4)
        overall_results['ivt_mAP_i'] = np.round(results_i["mAP"], 4)        
        logger.info(f"{run_string} Overall action mAP (sklearn): {overall_results['sklearn_mAP_a']:.4f}")
        logger.info(f"{run_string} Overall instrument mAP (sklearn): {overall_results['sklearn_mAP_i']:.4f}")
        logger.info(f"{run_string} Overall action mAP (ivt): {overall_results['ivt_mAP_a']:.4f}")
        logger.info(f"{run_string} Overall instrument mAP (ivt): {overall_results['ivt_mAP_i']:.4f}")

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
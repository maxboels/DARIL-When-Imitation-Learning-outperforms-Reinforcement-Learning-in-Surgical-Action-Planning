import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from tqdm import tqdm

# sklearn mean average precision
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score

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
    
    # Loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
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
        
        for batch_idx, (z_seq, _, _, _, c_a, c_i) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")):
            z_seq, c_a, c_i = z_seq.to(device), c_a.to(device), c_i.to(device)
            
            # Forward pass
            outputs = model(z_seq)
            action_logits = outputs['action_logits']
            instrument_logits = outputs['instrument_logits']
            
            # For simplicity, we'll use the last frame's actions as the target
            # In a real scenario, you might want to use all frames or a specific target
            target_actions = c_a[:, :]
            target_instruments = c_i[:, :]
            
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
            
            # Update statistics
            train_loss += loss.item() * z_seq.size(0)
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/Train', loss.item(), global_step)
            writer.add_scalar('Loss/Train_Action', loss_c_a.item(), global_step)
            writer.add_scalar('Loss/Train_Instrument', loss_c_i.item(), global_step)
            writer.add_scalar('Learning_Rate', learning_rate, global_step)
            
            # Log every N steps
            if batch_idx % cfg['training']['log_every_n_steps'] == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            all_preds_actions = []
            all_targets_actions = []
            all_preds_instruments = []
            all_targets_instruments = []
            
            with torch.no_grad():
                for batch_idx, (z_seq, _, _, _, c_a, c_i) in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation")):
                    z_seq, c_a, c_i = z_seq.to(device), c_a.to(device), c_i.to(device)
                    
                    # Forward pass
                    outputs = model(z_seq)
                    action_logits = outputs['action_logits']
                    
                    # For simplicity, use the last frame's actions as the target
                    target_actions = c_a[:, :]
                    target_instruments = c_i[:, :]
                    
                    # Calculate loss
                    loss_c_a = criterion(action_logits, target_actions)
                    loss_c_i = criterion(outputs['instrument_logits'], target_instruments)
                    loss = loss_c_a + loss_c_i
                    # Normalize loss by batch size
                    # loss = loss / z_seq.size(0)
                    val_loss += loss.item() * z_seq.size(0)
                    
                    # Store predictions and targets for mAP calculation
                    action_probs = torch.sigmoid(action_logits)
                    all_preds_actions.append(action_probs.cpu().numpy())
                    all_targets_actions.append(target_actions.cpu().numpy())

                    if outputs.get('instrument_logits') is not None:
                        instrument_probs = torch.sigmoid(outputs['instrument_logits'])
                        all_preds_instruments.append(instrument_probs.cpu().numpy())
                        all_targets_instruments.append(target_instruments.cpu().numpy())
                    
                    # Log to tensorboard
                    global_step = epoch * len(val_loader) + batch_idx
                    writer.add_scalar('Loss/Validation', loss.item(), global_step)
                    writer.add_scalar('Loss/Validation_Action', loss_c_a.item(), global_step)
                    writer.add_scalar('Loss/Validation_Instrument', loss_c_i.item(), global_step)
                    writer.add_scalar('Learning_Rate', learning_rate, global_step)

                    # Log every N steps
                    if batch_idx % cfg['training']['log_every_n_steps'] == 0:
                        logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(val_loader)}, Validation Loss: {loss.item():.4f}")
            
            # Calculate average validation loss
            val_loss /= len(val_loader.dataset)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}")
            
            # Calculate mAP
            all_preds_actions = np.concatenate(all_preds_actions, axis=0)
            all_targets_actions = np.concatenate(all_targets_actions, axis=0)
            all_preds_instruments = np.concatenate(all_preds_instruments, axis=0)
            all_targets_instruments = np.concatenate(all_targets_instruments, axis=0)
            # Calculate mAP for actions
            action_map_scores = average_precision_score(all_targets_actions, all_preds_actions, average=None)
            action_per_class_AP = {action_labels[i]: ap for i, ap in enumerate(action_map_scores)}
            action_overall_map = np.mean(action_map_scores)
            # Calculate mAP for instruments
            instrument_map_scores = average_precision_score(all_targets_instruments, all_preds_instruments, average=None)
            instrument_per_class_AP = {action_labels[i]: ap for i, ap in enumerate(instrument_map_scores)}
            instrument_overall_map = np.mean(instrument_map_scores)
            # Log mAP to tensorboard
            writer.add_scalar('Metrics/Validation_mAP_Action', action_overall_map, epoch)
            writer.add_scalar('Metrics/Validation_mAP_Instrument', instrument_overall_map, epoch)
            # Log mAP to console
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation mAP Action: {action_overall_map:.4f}")
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Validation mAP Instrument: {instrument_overall_map:.4f}")

            # Main metric for model selection
            val_map = action_overall_map

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
    
    return model

def run_recognition_inference(cfg, logger, model, test_loader, device='cuda'):
    """
    Run inference and evaluation on the test set.
    
    Args:
        model: Trained recognition head model
        test_loader: DataLoader for test data
        cfg: Configuration dictionary
        device: Device to run inference on
        log_dir: Directory to save outputs
        
    Returns:
        Output directory with results
    """
    # Log directory
    log_dir = os.path.join(logger.log_dir, 'inference')
    os.makedirs(log_dir, exist_ok=True)
    output_dir = os.path.join(log_dir, 'results')
    os.makedirs(output_dir, exist_ok=True)

    # Tensorboard for logging
    tensorboard_dir = os.path.join(log_dir, 'tensorboard')
    os.makedirs(tensorboard_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)
    
    # Load class labels
    with open(cfg['data']['paths']['class_labels_file_path'], 'r') as f:
        class_labels = json.load(f)
    action_labels = [class_name for class_id, class_name in class_labels['action'].items()]
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize containers for predictions and targets
    all_preds_actions = []
    all_targets_actions = []
    all_preds_instruments = []
    all_targets_instruments = []
    
    # Run inference
    with torch.no_grad():
        for batch_idx, (z_seq, _, _, _, c_a, c_i) in enumerate(tqdm(test_loader, desc="Running inference")):
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
            all_preds_actions.append(action_probs.cpu().numpy())
            all_targets_actions.append(target_actions.cpu().numpy())
            all_preds_instruments.append(instrument_probs.cpu().numpy())
            all_targets_instruments.append(target_instruments.cpu().numpy())

            # Log every N steps
            if batch_idx % cfg['training']['log_every_n_steps'] == 0:
                logger.info(f"Batch {batch_idx}/{len(test_loader)}, Predictions: {action_probs.shape}, Targets: {target_actions.shape}")
    
    # Concatenate all predictions and targets
    all_preds_actions = np.concatenate(all_preds_actions, axis=0)
    all_targets_actions = np.concatenate(all_targets_actions, axis=0)
    all_preds_instruments = np.concatenate(all_preds_instruments, axis=0)
    all_targets_instruments = np.concatenate(all_targets_instruments, axis=0)
    
    # Mean Average Precision (mAP) for actions
    action_map_scores = average_precision_score(all_targets_actions, all_preds_actions, average=None)
    action_per_class_AP = {action_labels[i]: ap for i, ap in enumerate(action_map_scores)}
    action_overall_map = np.mean(action_map_scores)
    # Mean Average Precision (mAP) for instruments
    instrument_map_scores = average_precision_score(all_targets_instruments, all_preds_instruments, average=None)
    instrument_per_class_AP = {action_labels[i]: ap for i, ap in enumerate(instrument_map_scores)}
    instrument_overall_map = np.mean(instrument_map_scores)
    # Log mAP to console
    logger.info(f"Test mAP Action: {action_overall_map:.4f}")
    logger.info(f"Test mAP Instrument: {instrument_overall_map:.4f}")
    # Save mAP results
    with open(os.path.join(output_dir, 'mAP_results.json'), 'w') as f:
        json.dump({
            'action_overall_map': action_overall_map,
            'instrument_overall_map': instrument_overall_map,
            'action_per_class_AP': action_per_class_AP,
            'instrument_per_class_AP': instrument_per_class_AP
        }, f, indent=4)
    logger.info(f"Saved mAP results to {os.path.join(output_dir, 'mAP_results.json')}")

    # return
    results = {
        "action_overall_map": action_overall_map,
        "instrument_overall_map": instrument_overall_map,
    }
    return results


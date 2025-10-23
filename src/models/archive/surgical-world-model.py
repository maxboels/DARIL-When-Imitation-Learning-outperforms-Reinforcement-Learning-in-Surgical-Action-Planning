import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import os
import logging
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model, get_cosine_schedule_with_warmup
from sklearn.metrics import precision_recall_curve, average_precision_score

class MultiLabelSurgicalWorldModel(nn.Module):
    """
    A robust world model for surgical action prediction that handles multi-label actions.
    Handles noisy action recognition during inference and incorporates tool information.
    """
    def __init__(
        self, 
        frame_dim=1024,          # Input frame embedding dimension
        action_dim=100,          # Number of possible action classes
        tool_dim=6,              # Number of possible tool classes
        hidden_dim=768,          # Hidden dimension for the model
        n_layer=6,               # Number of transformer layers
        max_sequence_length=20,  # Maximum sequence length to process
        dropout=0.1              # Dropout rate
    ):
        super().__init__()
        
        self.frame_dim = frame_dim
        self.action_dim = action_dim
        self.tool_dim = tool_dim
        self.hidden_dim = hidden_dim
        self.max_sequence_length = max_sequence_length
        
        # Frame embedding projection
        self.frame_projection = nn.Linear(frame_dim, hidden_dim // 2)
        
        # Multi-label action embedding (less reliable)
        # For binary action vectors, use linear projection
        self.action_projection = nn.Linear(action_dim, hidden_dim // 4)
        
        # Tool embedding (more reliable)
        self.tool_projection = nn.Linear(tool_dim, hidden_dim // 4)
        
        # Combined embedding projection
        self.combined_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_sequence_length, hidden_dim)
        )
        
        # Transformer model (GPT-2 based)
        self.config = GPT2Config(
            n_embd=hidden_dim,
            n_layer=n_layer,
            n_head=8,
            n_positions=max_sequence_length,
            resid_pdrop=dropout,
            attn_pdrop=dropout,
            summary_first_dropout=dropout
        )
        self.transformer = GPT2Model(self.config)
        
        # Output projection (predicts next frame embedding)
        self.output_projection = nn.Linear(hidden_dim, frame_dim)
        
        # Auxiliary action prediction head (for multi-task learning)
        self.action_prediction_head = nn.Linear(hidden_dim, action_dim)
        
        # Auxiliary tool prediction head
        self.tool_prediction_head = nn.Linear(hidden_dim, tool_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights similar to GPT-2"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
                
    def prepare_inputs(self, frames, actions=None, tools=None, 
                       action_confidence=0.4, tool_confidence=0.9):
        """
        Prepare inputs by combining frame, action, and tool information
        with confidence-weighted attention
        """
        batch_size, seq_len = frames.shape[:2]
        
        # Process frames
        frame_features = self.frame_projection(frames)
        
        # Process actions if provided
        if actions is not None:
            action_features = self.action_projection(actions)
            
            # Scale by confidence (lower for less reliable action recognition)
            action_features = action_features * action_confidence
        else:
            action_features = torch.zeros(
                batch_size, seq_len, self.hidden_dim // 4, 
                device=frames.device
            )
        
        # Process tools if provided
        if tools is not None:
            tool_features = self.tool_projection(tools)
            
            # Scale by confidence (higher for more reliable tool recognition)
            tool_features = tool_features * tool_confidence
        else:
            tool_features = torch.zeros(
                batch_size, seq_len, self.hidden_dim // 4, 
                device=frames.device
            )
        
        # Combine all features
        combined = torch.cat([frame_features, action_features, tool_features], dim=-1)
        
        # Project to model dimension
        hidden = self.combined_projection(combined)
        
        # Add positional embeddings (truncate if sequence is too long)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=frames.device
        ).unsqueeze(0)
        
        pos_emb = self.pos_embedding[:, :seq_len, :]
        hidden = hidden + pos_emb
        
        return hidden
        
    def forward(self, frames, actions=None, tools=None, 
                action_confidence=0.4, tool_confidence=0.9,
                return_dict=True):
        """
        Forward pass through the model
        
        Args:
            frames (torch.Tensor): Frame embeddings [batch_size, seq_len, frame_dim]
            actions (torch.Tensor, optional): Multi-label action vectors [batch_size, seq_len, action_dim]
            tools (torch.Tensor, optional): Tool vectors [batch_size, seq_len, tool_dim]
            action_confidence (float): Confidence weight for action inputs (0-1)
            tool_confidence (float): Confidence weight for tool inputs (0-1)
            return_dict (bool): Whether to return a dictionary of outputs
            
        Returns:
            Dictionary of outputs or just the predicted frame embeddings
        """
        # Prepare inputs
        hidden = self.prepare_inputs(
            frames, actions, tools, 
            action_confidence, tool_confidence
        )
        
        # Pass through transformer
        transformer_outputs = self.transformer(inputs_embeds=hidden)
        hidden_states = transformer_outputs.last_hidden_state
        
        # Predict next frames
        next_frame_embeddings = self.output_projection(hidden_states)
        
        if not return_dict:
            return next_frame_embeddings
            
        # Auxiliary predictions (actions and tools)
        predicted_actions = self.action_prediction_head(hidden_states)
        predicted_tools = self.tool_prediction_head(hidden_states)
        
        return {
            "predicted_frame_embeddings": next_frame_embeddings,
            "predicted_actions": predicted_actions,
            "predicted_tools": predicted_tools,
            "hidden_states": hidden_states,
            "all_hidden_states": transformer_outputs.hidden_states
        }
    
    def predict_next_frames(self, frames, actions=None, tools=None, 
                           horizon=5, temperature=0.8, use_autoregressive=True):
        """
        Predict future frames autoregressively
        
        Args:
            frames (torch.Tensor): Initial frame sequence [batch_size, seq_len, frame_dim]
            actions (torch.Tensor, optional): Action sequence [batch_size, seq_len, action_dim]
            tools (torch.Tensor, optional): Tool sequence [batch_size, seq_len, tool_dim]
            horizon (int): Number of future frames to predict
            temperature (float): Sampling temperature (0-1)
            use_autoregressive (bool): Whether to feed predictions back as inputs
            
        Returns:
            List of predicted frame embeddings
        """
        self.eval()
        device = frames.device
        batch_size, seq_len = frames.shape[:2]
        
        with torch.no_grad():
            # Initial prediction
            outputs = self.forward(frames, actions, tools)
            next_frame = outputs["predicted_frame_embeddings"][:, -1:, :]
            next_action = torch.sigmoid(outputs["predicted_actions"][:, -1:, :])
            next_tool = torch.sigmoid(outputs["predicted_tools"][:, -1:, :])
            
            # Initialize predictions
            predicted_frames = [next_frame]
            
            # Autoregressive generation
            if use_autoregressive:
                current_frames = frames
                current_actions = actions
                current_tools = tools
                
                for t in range(1, horizon):
                    # Update inputs with predicted frames
                    current_frames = torch.cat([current_frames[:, 1:], next_frame], dim=1)
                    
                    # Update actions and tools if available
                    if current_actions is not None:
                        current_actions = torch.cat([current_actions[:, 1:], next_action], dim=1)
                    
                    if current_tools is not None:
                        current_tools = torch.cat([current_tools[:, 1:], next_tool], dim=1)
                    
                    # Generate next prediction
                    outputs = self.forward(current_frames, current_actions, current_tools)
                    
                    # Add noise for diversity
                    if temperature > 0:
                        noise = torch.randn_like(outputs["predicted_frame_embeddings"][:, -1:, :]) * temperature
                        next_frame = outputs["predicted_frame_embeddings"][:, -1:, :] + noise
                    else:
                        next_frame = outputs["predicted_frame_embeddings"][:, -1:, :]
                    
                    next_action = torch.sigmoid(outputs["predicted_actions"][:, -1:, :])
                    next_tool = torch.sigmoid(outputs["predicted_tools"][:, -1:, :])
                    
                    predicted_frames.append(next_frame)
            else:
                # Non-autoregressive - predict multiple steps at once
                for t in range(1, horizon):
                    # Repeat last frame to pad sequence
                    padded_frames = torch.cat([frames, torch.stack(predicted_frames, dim=1)], dim=1)
                    
                    # Keep only the most recent frames
                    if padded_frames.shape[1] > self.max_sequence_length:
                        padded_frames = padded_frames[:, -self.max_sequence_length:]
                    
                    # Similar padding for actions and tools
                    padded_actions = None
                    if actions is not None:
                        padded_actions = torch.cat([actions, torch.ones_like(actions[:, :t]) * -1], dim=1)
                        if padded_actions.shape[1] > self.max_sequence_length:
                            padded_actions = padded_actions[:, -self.max_sequence_length:]
                            
                    padded_tools = None
                    if tools is not None:
                        padded_tools = torch.cat([tools, torch.ones_like(tools[:, :t]) * -1], dim=1)
                        if padded_tools.shape[1] > self.max_sequence_length:
                            padded_tools = padded_tools[:, -self.max_sequence_length:]
                    
                    # Predict next frame
                    outputs = self.forward(padded_frames, padded_actions, padded_tools)
                    
                    # Add temperature-scaled noise
                    if temperature > 0:
                        noise = torch.randn_like(outputs["predicted_frame_embeddings"][:, -1:, :]) * temperature
                        next_frame = outputs["predicted_frame_embeddings"][:, -1:, :] + noise
                    else:
                        next_frame = outputs["predicted_frame_embeddings"][:, -1:, :]
                        
                    predicted_frames.append(next_frame)
            
        # Stack all predictions
        return torch.cat(predicted_frames, dim=1)

    def rollout(self, initial_frames, recognized_actions=None, recognized_tools=None, 
               horizon=10, action_confidence=0.4, tool_confidence=0.9):
        """
        Perform a full rollout for evaluation, using recognized actions if available
        
        Args:
            initial_frames: Starting frames [batch_size, seq_len, frame_dim]
            recognized_actions: Recognized actions (can be None) [batch_size, seq_len, action_dim]
            recognized_tools: Recognized tools (can be None) [batch_size, seq_len, tool_dim]
            horizon: Number of frames to predict
            action_confidence: Confidence in action recognition
            tool_confidence: Confidence in tool recognition
            
        Returns:
            Predicted trajectory
        """
        return self.predict_next_frames(
            frames=initial_frames,
            actions=recognized_actions, 
            tools=recognized_tools,
            horizon=horizon, 
            temperature=0.7,
            use_autoregressive=True
        )


def noisy_action_generation(true_actions, noise_level=0.6):
    """
    Generate noisy actions to simulate recognition model errors
    
    Args:
        true_actions: Ground truth multi-label action vectors [batch_size, seq_len, action_dim]
        noise_level: Probability of corruption (0-1)
        
    Returns:
        Corrupted actions with realistic noise
    """
    batch_size, seq_len, action_dim = true_actions.shape
    
    # Initialize with true actions
    noisy_actions = true_actions.clone()
    
    # For each example and timestep
    for b in range(batch_size):
        for t in range(seq_len):
            # Decide whether to corrupt this timestep
            if random.random() < noise_level:
                # For multi-label actions, we have several corruption options:
                
                # 1. Miss true positives (false negatives)
                true_positives = torch.nonzero(true_actions[b, t]).squeeze(-1)
                if len(true_positives) > 0:
                    # Randomly drop some true actions
                    num_to_drop = random.randint(0, len(true_positives))
                    if num_to_drop > 0:
                        indices_to_drop = true_positives[torch.randperm(len(true_positives))[:num_to_drop]]
                        noisy_actions[b, t, indices_to_drop] = 0
                
                # 2. Add false positives
                false_positives = torch.nonzero(1 - true_actions[b, t]).squeeze(-1)
                if len(false_positives) > 0:
                    # Add some false actions (1-3)
                    num_to_add = random.randint(0, min(3, len(false_positives)))
                    if num_to_add > 0:
                        indices_to_add = false_positives[torch.randperm(len(false_positives))[:num_to_add]]
                        noisy_actions[b, t, indices_to_add] = 1
    
    return noisy_actions


def dual_path_train_step(model, batch, optimizer, action_noise_level=0.6, tool_noise_level=0.1,
                         alpha=0.5, teacher_forcing_ratio=0.5):
    """
    Perform a training step using dual-path approach with noisy recognition
    
    Args:
        model: World model
        batch: Dictionary with frames, actions, tools
        optimizer: Optimizer
        action_noise_level: Level of noise for actions (0-1)
        tool_noise_level: Level of noise for tools (0-1)
        alpha: Weight for noisy path loss (0-1)
        teacher_forcing_ratio: Probability of using teacher forcing
        
    Returns:
        Dictionary with losses and metrics
    """
    frames = batch['z']  # [batch_size, seq_len, frame_dim]
    actions = batch['c_a']  # [batch_size, action_dim]
    tools = batch['c_i']  # [batch_size, tool_dim]
    future_actions = batch['f_a']  # [batch_size, horizon, action_dim]
    
    # For sequence prediction, split into input and target
    input_frames = frames[:, :-1]
    target_frames = frames[:, 1:]
    input_actions = actions.unsqueeze(1).repeat(1, input_frames.shape[1], 1)
    input_tools = tools.unsqueeze(1).repeat(1, input_frames.shape[1], 1)
    
    # Path 1: Ground truth actions (teacher forcing)
    if random.random() < teacher_forcing_ratio:
        # Use ground truth with teacher forcing
        gt_outputs = model(input_frames, input_actions, input_tools, 
                         action_confidence=1.0, tool_confidence=1.0)
        
        # MSE loss for frame prediction
        frame_loss = F.mse_loss(gt_outputs["predicted_frame_embeddings"], target_frames)
        
        # BCE loss for action prediction
        action_loss = F.binary_cross_entropy_with_logits(
            gt_outputs["predicted_actions"], 
            input_actions
        )
        
        # BCE loss for tool prediction
        tool_loss = F.binary_cross_entropy_with_logits(
            gt_outputs["predicted_tools"],
            input_tools
        )
        
        # Path 1 total loss
        path1_loss = frame_loss + 0.1 * action_loss + 0.1 * tool_loss
    else:
        # Autoregressive prediction without teacher forcing
        predicted_frames = []
        current_frame = input_frames[:, 0:1]
        current_action = input_actions[:, 0:1]
        current_tool = input_tools[:, 0:1]
        
        for t in range(input_frames.shape[1]):
            # Predict next frame
            outputs = model(current_frame, current_action, current_tool,
                           action_confidence=1.0, tool_confidence=1.0)
            
            next_frame = outputs["predicted_frame_embeddings"]
            predicted_frames.append(next_frame)
            
            # Update for next timestep
            if t < input_frames.shape[1] - 1:
                current_frame = next_frame
                current_action = torch.sigmoid(outputs["predicted_actions"])
                current_tool = torch.sigmoid(outputs["predicted_tools"])
        
        # Stack predictions
        predicted_sequence = torch.cat(predicted_frames, dim=1)
        
        # Calculate losses
        frame_loss = F.mse_loss(predicted_sequence, target_frames)
        action_loss = 0
        tool_loss = 0
        
        # Path 1 total loss
        path1_loss = frame_loss
    
    # Path 2: Noisy actions (simulate recognition)
    noisy_actions = noisy_action_generation(input_actions, noise_level=action_noise_level)
    noisy_tools = noisy_action_generation(input_tools, noise_level=tool_noise_level)
    
    noisy_outputs = model(input_frames, noisy_actions, noisy_tools,
                          action_confidence=1.0-action_noise_level, 
                          tool_confidence=1.0-tool_noise_level)
    
    # MSE loss for frame prediction
    noisy_frame_loss = F.mse_loss(noisy_outputs["predicted_frame_embeddings"], target_frames)
    
    # BCE loss for action prediction (correct to true actions despite noise)
    noisy_action_loss = F.binary_cross_entropy_with_logits(
        noisy_outputs["predicted_actions"],
        input_actions
    )
    
    # BCE loss for tool prediction
    noisy_tool_loss = F.binary_cross_entropy_with_logits(
        noisy_outputs["predicted_tools"],
        input_tools
    )
    
    # Path 2 total loss
    path2_loss = noisy_frame_loss + 0.1 * noisy_action_loss + 0.1 * noisy_tool_loss
    
    # Combined loss
    total_loss = (1 - alpha) * path1_loss + alpha * path2_loss
    
    # Optimization step
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return {
        "total_loss": total_loss.item(),
        "path1_loss": path1_loss.item(),
        "path2_loss": path2_loss.item(),
        "frame_loss": frame_loss.item(),
        "action_loss": action_loss if isinstance(action_loss, float) else action_loss.item(),
        "tool_loss": tool_loss if isinstance(tool_loss, float) else tool_loss.item(),
        "noisy_frame_loss": noisy_frame_loss.item(),
    }


def train_world_model(model, train_loader, val_loader, cfg, logger, device='cuda'):
    """
    Train the world model
    
    Args:
        model: World model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        cfg: Configuration dictionary
        logger: Logger for logging training progress
        device: Device to train on
        
    Returns:
        Trained model and best model path
    """
    # Create directories
    save_dir = cfg['training']['checkpoint_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    log_dir = cfg['training']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg['training']['learning_rate'],
        weight_decay=cfg['training']['weight_decay']
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * cfg['training']['epochs']
    warmup_steps = cfg['training']['scheduler']['warmup_steps']
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training parameters
    num_epochs = cfg['training']['epochs']
    best_val_loss = float('inf')
    best_model_path = None
    
    # Initialize alpha for dual path (gradually increase importance of noisy path)
    initial_alpha = 0.1
    final_alpha = 0.8
    
    # Initialize teacher forcing ratio (gradually decrease)
    initial_tf_ratio = 1.0
    final_tf_ratio = 0.2
    
    # Initialize action noise level (gradually increase)
    initial_action_noise = 0.2  # Start with little noise
    final_action_noise = 0.6    # Increase to 60% noise (matching 40% recognition accuracy)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        
        # Calculate current training parameters based on epoch
        progress = epoch / max(1, num_epochs - 1)
        current_alpha = initial_alpha + progress * (final_alpha - initial_alpha)
        current_tf_ratio = initial_tf_ratio - progress * (initial_tf_ratio - final_tf_ratio)
        current_action_noise = initial_action_noise + progress * (final_action_noise - initial_action_noise)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} Training")
        logger.info(f"Alpha: {current_alpha:.2f}, TF Ratio: {current_tf_ratio:.2f}, Action Noise: {current_action_noise:.2f}")
        
        # Training loop
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Training step
            loss_dict = dual_path_train_step(
                model, batch, optimizer,
                action_noise_level=current_action_noise,
                tool_noise_level=0.1,  # Fixed low noise for tools (90% accuracy)
                alpha=current_alpha,
                teacher_forcing_ratio=current_tf_ratio
            )
            
            # Log losses
            for loss_name, loss_value in loss_dict.items():
                writer.add_scalar(f"Train/{loss_name}", loss_value, 
                                 epoch * len(train_loader) + batch_idx)
            
            # Update learning rate
            scheduler.step()
            
            # Track losses
            train_losses.append(loss_dict["total_loss"])
            
            # Logging
            if batch_idx % cfg['training']['log_every_n_steps'] == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss_dict['total_loss']:.4f}, LR: {lr:.6f}")
        
        # Calculate average training loss
        avg_train_loss = sum(train_losses) / len(train_losses)
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation
        if val_loader:
            val_loss, val_metrics = validate_world_model(
                model, val_loader, device=device, epoch=epoch, logger=logger
            )
            
            # Log validation metrics
            for metric_name, metric_value in val_metrics.items():
                writer.add_scalar(f"Val/{metric_name}", metric_value, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(
                    save_dir, f"best_model_epoch{epoch+1}_loss{val_loss:.4f}.pt"
                )
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': cfg
                }, best_model_path)
                
                logger.info(f"New best model saved at {best_model_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % cfg['training']['save_checkpoint_every_n_epochs'] == 0:
            checkpoint_path = os.path.join(
                save_dir, f"model_epoch{epoch+1}.pt"
            )
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg
            }, checkpoint_path)
            
            logger.info(f"Checkpoint saved at {checkpoint_path}")
    
    # Close TensorBoard writer
    writer.close()
    
    return model, best_model_path


def validate_world_model(model, val_loader, device='cuda', epoch=None, logger=None):
    """
    Validate the world model
    
    Args:
        model: World model
        val_loader: DataLoader for validation data
        device: Device to validate on
        epoch: Current epoch (for logging)
        logger: Logger for logging validation progress
        
    Returns:
        Validation loss and metrics
    """
    model.eval()
    val_losses = []
    
    # Tracking metrics
    frame_mse = []
    frame_cosine_sim = []
    action_map = []  # Mean Average Precision for actions
    
    if logger:
        logger.info("Starting validation")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            frames = batch['z']
            actions = batch['c_a']
            tools = batch['c_i']
            future_actions = batch['f_a']
            
            # For sequence prediction, split into input and target
            input_frames = frames[:, :-1]
            target_frames = frames[:, 1:]
            input_actions = actions.unsqueeze(1).repeat(1, input_frames.shape[1], 1)
            input_tools = tools.unsqueeze(1).repeat(1, input_frames.shape[1], 1)
            
            # Forward pass
            outputs = model(input_frames, input_actions, input_tools)
            predicted_frames = outputs["predicted_frame_embeddings"]
            
            # Calculate frame MSE loss
            loss = F.mse_loss(predicted_frames, target_frames)
            val_losses.append(loss.item())
            
            # Calculate frame cosine similarity
            frame_cos_sim = F.cosine_similarity(
                predicted_frames.reshape(-1, predicted_frames.shape[-1]),
                target_frames.reshape(-1, target_frames.shape[-1]),
                dim=1
            ).mean().item()
            frame_cosine_sim.append(frame_cos_sim)
            
            # Calculate action prediction MAP
            # For multi-label actions, use average precision
            predicted_actions = torch.sigmoid(outputs["predicted_actions"]).cpu().numpy()
            true_actions = input_actions.cpu().numpy()
            
            # Calculate average precision for each sample and class
            for b in range(true_actions.shape[0]):
                for c in range(true_actions.shape[2]):
                    y_true = true_actions[b, :, c]
                    y_pred = predicted_actions[b, :, c]
                    
                    # Only calculate AP if there are positive examples
                    if np.sum(y_true) > 0:
                        ap = average_precision_score(y_true, y_pred)
                        action_map.append(ap)
            
            # Track MSE
            frame_mse.append(loss.item())
            
            # Log progress
            if logger and batch_idx % 10 == 0:
                logger.info(f"Validation batch {batch_idx}/{len(val_loader)}, Loss: {loss.item():.4f}")
    
    # Calculate average validation metrics
    avg_val_loss = sum(val_losses) / len(val_losses)
    avg_frame_mse = sum(frame_mse) / len(frame_mse)
    avg_frame_cos_sim = sum(frame_cosine_sim) / len(frame_cosine_sim)
    avg_action_map = sum(action_map) / len(action_map) if action_map else 0
    
    # Log results
    if logger:
        logger.info(f"Validation Loss: {avg_val_loss:.4f}")
        logger.info(f"Frame MSE: {avg_frame_mse:.4f}")
        logger.info(f"Frame Cosine Similarity: {avg_frame_cos_sim:.4f}")
        logger.info(f"Action MAP: {avg_action_map:.4f}")
        
        if epoch is not None:
            logger.info(f"Epoch {epoch+1} Validation completed")
    
    return avg_val_loss, {
        "frame_mse": avg_frame_mse,
        "frame_cosine_similarity": avg_frame_cos_sim,
        "action_map": avg_action_map
    }


def run_world_model_evaluation(model, test_loaders, cfg, logger, device='cuda'):
    """
    Run comprehensive evaluation on the world model
    
    Args:
        model: Trained world model
        test_loaders: Dictionary of DataLoaders for test videos
        cfg: Configuration dictionary
        logger: Logger
        device: Device to run evaluation on
        
    Returns:
        Dictionary of evaluation results
    """
    model.eval()
    
    # Create output directory
    output_dir = os.path.join(logger.log_dir, 'evaluation_results')
    os.makedirs(output_dir, exist_ok=True)
    
    # Tracking metrics across videos
    video_metrics = {}
    overall_metrics = {
        "frame_mse": [],
        "frame_cosine_similarity": [],
        "action_prediction_map": [],
    }
    
    # Prediction horizons to evaluate
    horizons = cfg['eval']['world_model']['eval_horizons']
    max_horizon = cfg['eval']['world_model']['max_horizon']
    
    # Set up metrics for each horizon
    for h in horizons:
        overall_metrics[f"frame_mse_h{h}"] = []
        overall_metrics[f"action_map_h{h}"] = []
        overall_metrics[f"cosine_sim_h{h}"] = []
    
    with torch.no_grad():
        for video_id, loader in test_loaders.items():
            logger.info(f"Evaluating video {video_id}")
            
            video_metrics[video_id] = {
                "frame_mse": [],
                "frame_cosine_similarity": [],
                "action_prediction_map": [],
            }
            
            # Initialize per-horizon metrics for this video
            for h in horizons:
                video_metrics[video_id][f"frame_mse_h{h}"] = []
                video_metrics[video_id][f"action_map_h{h}"] = []
                video_metrics[video_id][f"cosine_sim_h{h}"] = []
            
            # Collect sequences for visualization
            vis_sequences = []
            
            for batch_idx, batch in enumerate(tqdm(loader, desc=f"Video {video_id}")):
                # Move batch to device
                batch = {k: v.to(device) for k, v in batch.items()}
                
                frames = batch['z']
                actions = batch['c_a']
                tools = batch['c_i']
                future_actions = batch['f_a']  # [batch_size, horizon, action_dim]
                
                # For sequence prediction, use all frames as input for initial prediction
                input_frames = frames
                input_actions = actions.unsqueeze(1).repeat(1, input_frames.shape[1], 1)
                input_tools = tools.unsqueeze(1).repeat(1, input_frames.shape[1], 1)
                
                # Generate future frame predictions
                predicted_future_frames = model.predict_next_frames(
                    frames=input_frames,
                    actions=input_actions, 
                    tools=input_tools,
                    horizon=max_horizon,
                    temperature=0.0  # No noise for evaluation
                )
                
                # Get predicted actions for future frames
                outputs = model(input_frames, input_actions, input_tools)
                predicted_future_actions = torch.sigmoid(outputs["predicted_actions"])
                
                # Calculate metrics for each prediction horizon
                for h_idx, h in enumerate(horizons):
                    if h <= max_horizon:
                        # Only evaluate horizons up to max_horizon
                        if h_idx < predicted_future_frames.shape[1] and h_idx < future_actions.shape[1]:
                            # Frame prediction error at horizon h
                            if batch_idx + h < len(loader.dataset):
                                # Get actual future frame if available in dataset
                                future_batch = loader.dataset[batch_idx + h]
                                actual_future_frame = future_batch['z'].to(device)
                                
                                # Calculate MSE for this horizon
                                horizon_mse = F.mse_loss(
                                    predicted_future_frames[:, h_idx], 
                                    actual_future_frame[:, 0]
                                ).item()
                                
                                # Calculate cosine similarity
                                horizon_cos_sim = F.cosine_similarity(
                                    predicted_future_frames[:, h_idx].reshape(-1, predicted_future_frames.shape[-1]),
                                    actual_future_frame[:, 0].reshape(-1, actual_future_frame.shape[-1]),
                                    dim=1
                                ).mean().item()
                                
                                video_metrics[video_id][f"frame_mse_h{h}"].append(horizon_mse)
                                video_metrics[video_id][f"cosine_sim_h{h}"].append(horizon_cos_sim)
                                overall_metrics[f"frame_mse_h{h}"].append(horizon_mse)
                                overall_metrics[f"cosine_sim_h{h}"].append(horizon_cos_sim)
                            
                            # Action prediction at horizon h
                            if h_idx < future_actions.shape[1]:
                                # Compare predicted actions with ground truth future actions
                                pred_actions_h = predicted_future_actions[:, h_idx].cpu().numpy()
                                true_actions_h = future_actions[:, h_idx].cpu().numpy()
                                
                                # Calculate MAP for each class
                                action_map_h = []
                                for c in range(true_actions_h.shape[1]):
                                    if np.sum(true_actions_h[:, c]) > 0:
                                        ap = average_precision_score(
                                            true_actions_h[:, c], 
                                            pred_actions_h[:, c]
                                        )
                                        action_map_h.append(ap)
                                
                                if action_map_h:
                                    avg_map_h = sum(action_map_h) / len(action_map_h)
                                    video_metrics[video_id][f"action_map_h{h}"].append(avg_map_h)
                                    overall_metrics[f"action_map_h{h}"].append(avg_map_h)
                
                # Store some sequences for visualization
                if len(vis_sequences) < 5 and batch_idx % 50 == 0:
                    vis_sequences.append({
                        "input_frames": frames.cpu().numpy(),
                        "predicted_future": predicted_future_frames.cpu().numpy(),
                        "true_actions": future_actions.cpu().numpy(),
                        "predicted_actions": predicted_future_actions.cpu().numpy(),
                    })
                
                # Log progress
                if batch_idx % 10 == 0:
                    logger.info(f"Processed batch {batch_idx}/{len(loader)}")
            
            # Calculate average metrics for this video
            for metric_name, values in video_metrics[video_id].items():
                if values:
                    video_metrics[video_id][f"avg_{metric_name}"] = sum(values) / len(values)
            
            # Log video results
            logger.info(f"Results for video {video_id}:")
            for h in horizons:
                if video_metrics[video_id][f"frame_mse_h{h}"]:
                    avg_mse_h = sum(video_metrics[video_id][f"frame_mse_h{h}"]) / len(video_metrics[video_id][f"frame_mse_h{h}"])
                    logger.info(f"  Horizon {h} - Frame MSE: {avg_mse_h:.4f}")
                
                if video_metrics[video_id][f"action_map_h{h}"]:
                    avg_map_h = sum(video_metrics[video_id][f"action_map_h{h}"]) / len(video_metrics[video_id][f"action_map_h{h}"])
                    logger.info(f"  Horizon {h} - Action MAP: {avg_map_h:.4f}")
            
            # Save visualization sequences
            if vis_sequences:
                np.save(os.path.join(output_dir, f"{video_id}_sequences.npy"), vis_sequences)
    
    # Calculate overall average metrics
    overall_avg_metrics = {}
    for metric_name, values in overall_metrics.items():
        if values:
            overall_avg_metrics[f"avg_{metric_name}"] = sum(values) / len(values)
    
    # Log overall results
    logger.info("Overall evaluation results:")
    for h in horizons:
        if f"frame_mse_h{h}" in overall_avg_metrics:
            logger.info(f"  Horizon {h} - Frame MSE: {overall_avg_metrics[f'avg_frame_mse_h{h}']:.4f}")
        
        if f"action_map_h{h}" in overall_avg_metrics:
            logger.info(f"  Horizon {h} - Action MAP: {overall_avg_metrics[f'avg_action_map_h{h}']:.4f}")
    
    # Save results
    results = {
        "video_metrics": video_metrics,
        "overall_metrics": overall_avg_metrics
    }
    
    # Save results to JSON
    import json
    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def visualize_predictions(model, test_sequence, output_dir, device='cuda'):
    """
    Visualize model predictions for a test sequence
    
    Args:
        model: Trained world model
        test_sequence: Dictionary with test inputs
        output_dir: Directory to save visualizations
        device: Device to run inference on
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test sequence
    frames = torch.tensor(test_sequence["input_frames"]).to(device)
    actions = torch.tensor(test_sequence.get("actions", None)).to(device) if "actions" in test_sequence else None
    tools = torch.tensor(test_sequence.get("tools", None)).to(device) if "tools" in test_sequence else None
    
    # Generate predictions with different settings
    results = {}
    
    # 1. Ground truth actions with 100% confidence
    if actions is not None:
        results["ground_truth"] = model.predict_next_frames(
            frames=frames,
            actions=actions,
            tools=tools,
            horizon=10,
            temperature=0.0
        ).cpu().numpy()
    
    # 2. Ground truth actions with 40% confidence (simulating recognition)
    if actions is not None:
        results["recognition_40"] = model.predict_next_frames(
            frames=frames,
            actions=actions,
            tools=tools,
            horizon=10,
            temperature=0.0,
            action_confidence=0.4,
            tool_confidence=0.9
        ).cpu().numpy()
    
    # 3. No action conditioning
    results["no_actions"] = model.predict_next_frames(
        frames=frames,
        actions=None,
        tools=None,
        horizon=10,
        temperature=0.0
    ).cpu().numpy()
    
    # 4. With temperature for diversity
    results["with_temperature"] = model.predict_next_frames(
        frames=frames,
        actions=actions,
        tools=tools,
        horizon=10,
        temperature=0.8
    ).cpu().numpy()
    
    # Save visualization data
    np.save(os.path.join(output_dir, "prediction_visualizations.npy"), results)
    
    # Generate plots if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        from sklearn.decomposition import PCA
        
        # Use PCA to project frame embeddings to 2D for visualization
        all_frames = []
        for name, predictions in results.items():
            all_frames.append(predictions.reshape(-1, predictions.shape[-1]))
        
        all_frames_concat = np.concatenate(all_frames, axis=0)
        pca = PCA(n_components=2)
        pca.fit(all_frames_concat)
        
        # Plot each prediction trajectory
        plt.figure(figsize=(12, 8))
        for name, predictions in results.items():
            flattened = predictions.reshape(-1, predictions.shape[-1])
            projected = pca.transform(flattened)
            plt.plot(projected[:, 0], projected[:, 1], 'o-', label=name)
        
        plt.title("Predicted Frame Embedding Trajectories (PCA)")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "trajectory_visualization.png"))
        plt.close()
    except ImportError:
        print("Matplotlib not available for visualization")

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Train and evaluate Surgical World Model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"], help="Mode")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for evaluation")
    args = parser.parse_args()
    
    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Set up logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(cfg['training']['log_dir'], "train.log"))
        ]
    )
    logger = logging.getLogger()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize model
    model = MultiLabelSurgicalWorldModel(
        frame_dim=cfg['models']['world_model']['embedding_dim'],
        action_dim=cfg['models']['world_model']['targets_dims']['_a'],
        tool_dim=cfg['models']['recognition']['transformer']['num_instrument_classes'],
        hidden_dim=cfg['models']['world_model']['hidden_dim'],
        n_layer=cfg['models']['world_model']['n_layer']
    ).to(device)
    
    # Load data
    from datasets import load_cholect50_data, NextFramePredictionDataset
    
    train_data = load_cholect50_data(cfg['data'], split='train', max_videos=cfg['experiment']['max_videos'])
    test_data = load_cholect50_data(cfg['data'], split='test', max_videos=cfg['experiment']['max_videos'])
    
    train_dataset = NextFramePredictionDataset(cfg['data'], train_data)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=True,
        num_workers=cfg['training']['num_workers'],
        pin_memory=cfg['training']['pin_memory']
    )
    
    # Create test loaders for each video
    from datasets import create_video_dataloaders
    test_loaders = create_video_dataloaders(
        cfg, 
        test_data, 
        batch_size=cfg['training']['batch_size'], 
        shuffle=False
    )
    
    if args.mode == "train":
        # Train model
        logger.info("Starting training")
        model, best_model_path = train_world_model(
            model, train_loader, test_loaders, cfg, logger, device
        )
        logger.info(f"Training completed. Best model saved at {best_model_path}")
    
    elif args.mode == "eval":
        # Load checkpoint for evaluation
        checkpoint_path = args.checkpoint or cfg['experiment']['pretrain_next_frame']['best_model_path']
        if checkpoint_path and os.path.exists(checkpoint_path):
            logger.info(f"Loading checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning("No checkpoint provided or found. Using initialized model.")
        
        # Run evaluation
        logger.info("Starting evaluation")
        results = run_world_model_evaluation(model, test_loaders, cfg, logger, device)
        logger.info("Evaluation completed")
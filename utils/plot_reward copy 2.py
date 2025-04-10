import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches

# Set style
sns.set(style="whitegrid")
plt.rcParams.update({'font.size': 12})

# Constants
NUM_FRAMES = 100
CONTEXT_LENGTH = 5  # c_a
ANTICIPATION_LENGTH = 3  # l_a
EMBEDDING_DIM = 1024

# Function to generate synthetic frame embeddings
def generate_embeddings(num_frames, embedding_dim):
    return np.random.randn(num_frames, embedding_dim) * 0.1

# Function to generate synthetic risk scores (1-5)
def generate_risk_scores(num_frames):
    return np.random.randint(1, 6, size=num_frames)

# Simulated GPT-2 Frame Predictor
class GPT2FramePredictor:
    def __init__(self, embedding_dim):
        # Simple weights for demonstration
        self.weights = np.random.randn(embedding_dim) * 0.01
        self.bias = np.random.randn(embedding_dim) * 0.001
        
    def predict(self, embedding):
        # Simple linear transformation with noise
        return embedding * self.weights + self.bias + np.random.randn(len(embedding)) * 0.01
    
    def generate_future(self, embedding, length):
        futures = []
        current = embedding.copy()
        
        for _ in range(length):
            current = self.predict(current)
            futures.append(current)
            
        return np.array(futures)

# Simulated Reward Prediction Model
class RewardModel:
    def __init__(self, embedding_dim, context_length):
        self.weights = np.random.randn(embedding_dim * context_length) * 0.001
        self.bias = 100  # Base survival time
        
    def predict(self, embeddings):
        # Flatten embeddings from context window
        flat_input = embeddings.flatten()
        
        # Simple weighted sum with noise
        prediction = self.bias
        prediction += np.sum(flat_input[:len(self.weights)] * self.weights)
        prediction += np.random.randn() * 5  # Add noise
        
        # Ensure prediction is reasonable (survival weeks)
        return max(10, min(200, prediction))

# Function to simulate training
def simulate_training(frame_predictor, reward_model, frame_embeddings, epochs=5):
    print("Simulating model training...")
    # In a real scenario, we would train models here
    # For this demo, we'll just use the pre-initialized weights

# Function to compute expected rewards over time
def compute_rewards_over_time(frame_embeddings, risk_scores, frame_predictor, reward_model):
    rewards = []
    future_rewards = []
    reward_diffs = []
    
    for t in range(NUM_FRAMES):
        # Get context (previous c_a frames, or as many as available)
        start_idx = max(0, t - CONTEXT_LENGTH + 1)
        context = frame_embeddings[start_idx:t+1]
        
        # Pad context if needed
        if len(context) < CONTEXT_LENGTH:
            padding = np.zeros((CONTEXT_LENGTH - len(context), EMBEDDING_DIM))
            context = np.vstack([padding, context])
        
        # Predict current expected reward
        current_reward = reward_model.predict(context)
        rewards.append(current_reward)
        
        # If we have enough frames left, predict future reward
        if t < NUM_FRAMES - ANTICIPATION_LENGTH:
            # Generate future embeddings
            future_embeddings = frame_predictor.generate_future(
                frame_embeddings[t], ANTICIPATION_LENGTH)
            
            # Combine context with future for anticipated reward
            # Remove oldest frames from context to accommodate future frames
            combined = np.vstack([
                context[ANTICIPATION_LENGTH:],
                future_embeddings
            ])
            
            future_reward = reward_model.predict(combined)
            future_rewards.append(future_reward)
            reward_diffs.append(future_reward - current_reward)
        else:
            # For last few frames, just append None
            future_rewards.append(None)
            reward_diffs.append(None)
    
    return rewards, future_rewards, reward_diffs

# Function to create a baseline trend (declining health over time, with recovery moments)
def create_baseline_trend(num_frames, start=150, end=50):
    # Create overall declining trend
    x = np.linspace(0, 1, num_frames)
    trend = start - (start - end) * x
    
    # Add some recovery moments with gaussian bumps
    recovery_points = [int(num_frames * 0.2), int(num_frames * 0.5), int(num_frames * 0.8)]
    for point in recovery_points:
        gaussian = 20 * np.exp(-0.05 * (np.arange(num_frames) - point) ** 2)
        trend += gaussian
    
    # Add small noise
    trend += np.random.randn(num_frames) * 3
    
    return trend

# Main function to run the simulation and create plots
def plot_reward_predictions():
    # Generate synthetic data for one video
    frame_embeddings = generate_embeddings(NUM_FRAMES, EMBEDDING_DIM)
    risk_scores = generate_risk_scores(NUM_FRAMES)
    
    # Initialize models
    frame_predictor = GPT2FramePredictor(EMBEDDING_DIM)
    reward_model = RewardModel(EMBEDDING_DIM, CONTEXT_LENGTH)
    
    # Simulate training
    simulate_training(frame_predictor, reward_model, frame_embeddings)
    
    # Create more realistic trends for our plot
    patient_trend = create_baseline_trend(NUM_FRAMES)
    
    # Compute rewards over time based on the trend
    rewards, future_rewards, reward_diffs = [], [], []
    for t in range(NUM_FRAMES):
        # Add noise to the trend to create our "predictions"
        current_reward = patient_trend[t] + np.random.randn() * 5
        rewards.append(current_reward)
        
        if t < NUM_FRAMES - ANTICIPATION_LENGTH:
            # Future reward is affected by risk score
            # Higher risk tends to decrease future reward
            risk_impact = (risk_scores[t] - 3) * -5  # Higher risk = more negative impact
            future_reward = current_reward + risk_impact + np.random.randn() * 8
            future_rewards.append(future_reward)
            reward_diffs.append(future_reward - current_reward)
        else:
            future_rewards.append(None)
            reward_diffs.append(None)
    
    # Create a timeline for x-axis (in seconds, assuming 30 fps)
    timeline = np.arange(NUM_FRAMES) / 30  # Convert to seconds
    
    # Create a DataFrame for easier plotting
    df = pd.DataFrame({
        'Frame': range(NUM_FRAMES),
        'Time (s)': timeline,
        'Expected Reward': rewards,
        'Future Reward': future_rewards,
        'Reward Difference': reward_diffs,
        'Risk Score': risk_scores
    })
    
    # Create a colormap for risk scores
    colors = ['#10b981', '#84cc16', '#eab308', '#f97316', '#ef4444']  # Green to red
    risk_cmap = LinearSegmentedColormap.from_list('risk_cmap', colors)
    
    # Create main plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Plot expected reward
    ax1.plot(df['Time (s)'], df['Expected Reward'], 
             color='#3b82f6', linewidth=2.5, label='Expected Reward')
    
    # Plot future reward (with some points)
    valid_future = df[df['Future Reward'].notna()]
    ax1.plot(valid_future['Time (s)'], valid_future['Future Reward'], 
             color='#8b5cf6', linestyle='--', linewidth=1.5, 
             alpha=0.7, label='Expected Reward (with future prediction)')
    
    # Add colored points for risk scores
    scatter = ax1.scatter(df['Time (s)'], df['Expected Reward'], 
               c=df['Risk Score'], cmap=risk_cmap, 
               s=50, zorder=5, alpha=0.8, vmin=1, vmax=5)
    
    # Set up plot
    ax1.set_title('Expected Survival Time Prediction Over Video Progression', fontsize=16)
    ax1.set_ylabel('Expected Survival (weeks)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add colorbar for risk scores
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Risk Score', fontsize=12)
    cbar.set_ticks([1, 2, 3, 4, 5])
    
    # Add legend
    ax1.legend(loc='upper right', fontsize=12)
    
    # Add annotations for significant moments
    significant_moments = [
        (10, "Initial assessment"),
        (30, "Critical maneuver"),
        (50, "Procedure midpoint"),
        (75, "Complication managed"),
        (95, "Closing steps")
    ]
    
    for frame, text in significant_moments:
        idx = frame
        ax1.annotate(text, 
                     xy=(timeline[idx], rewards[idx]),
                     xytext=(0, 20),
                     textcoords="offset points",
                     arrowprops=dict(arrowstyle="->", color='gray'),
                     fontsize=10)
    
    # Plot reward difference in lower subplot
    valid_diff = df[df['Reward Difference'].notna()]
    bars = ax2.bar(valid_diff['Time (s)'], valid_diff['Reward Difference'], 
            width=0.3, alpha=0.7, color=valid_diff['Reward Difference'].apply(
                lambda x: '#10b981' if x > 0 else '#ef4444'))
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Reward Difference', fontsize=12)
    ax2.set_xlabel('Time (seconds)', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Create legend for the reward difference
    pos_patch = mpatches.Patch(color='#10b981', label='Positive Impact')
    neg_patch = mpatches.Patch(color='#ef4444', label='Negative Impact')
    ax2.legend(handles=[pos_patch, neg_patch], loc='upper right')
    
    # Highlight sections with high risk and negative reward difference
    risky_sections = [
        (12, 15, "High Risk Zone"),
        (45, 47, "Critical Period"),
        (74, 76, "Intervention Needed")
    ]
    
    for start, end, label in risky_sections:
        ax1.axvspan(timeline[start], timeline[end], alpha=0.2, color='red')
        ax1.text(timeline[start] + (timeline[end] - timeline[start])/2, 
                 min(rewards) - 10, label, 
                 ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig("reward_prediction_over_time.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Average Expected Reward: {np.mean(rewards):.2f} weeks")
    print(f"Minimum Expected Reward: {np.min(rewards):.2f} weeks")
    print(f"Maximum Expected Reward: {np.max(rewards):.2f} weeks")
    print(f"Average Reward Difference: {np.nanmean(reward_diffs):.2f} weeks")
    
    # Risk score impact analysis
    risk_impact = []
    for risk in range(1, 6):
        mask = df['Risk Score'] == risk
        if mask.any() and df.loc[mask, 'Reward Difference'].notna().any():
            avg_diff = df.loc[mask, 'Reward Difference'].mean()
            risk_impact.append((risk, avg_diff))
    
    print("\nRisk Score Impact on Reward Difference:")
    for risk, impact in risk_impact:
        print(f"Risk Score {risk}: {impact:.2f} weeks")
    
    return df

if __name__ == "__main__":
    np.random.seed(42)  # For reproducibility
    results = plot_reward_predictions()
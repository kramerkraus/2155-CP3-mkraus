# Import all required libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from scipy.spatial.distance import jensenshannon
import os
import warnings
import json
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set plot style
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_dataset_splits(data_dir='dataset'):
    """
    Load pre-split dataset files from the specified directory.
    
    Dataset structure:
    - train/val/test/test2 splits for training and evaluation
    - original: Complete feature vectors (ground truth)
    - imputed: Vectors with missing values replaced by -1 
    - missing_mask: Boolean mask (True=observed, False=missing)
    
    For test2: We only have imputed and missing_mask (no original for evaluation)
    """
    splits = {}
    
    # Define the expected split types
    split_types = ['train', 'val', 'test', 'test2']
    
    print(f"Loading dataset splits from: {data_dir}")
    
    for split_type in split_types:
        print(f"\nLoading {split_type} split...")
        splits[split_type] = {}
        
        # Load original data (ground truth) - not available for test2
        if split_type != 'test2':
            original_path = os.path.join(data_dir, f'{split_type}_original.csv')
            if os.path.exists(original_path):
                splits[split_type]['original'] = pd.read_csv(original_path).values
                print(f"  ✓ {split_type}_original.csv: {splits[split_type]['original'].shape}")
            else:
                print(f"  ✗ {original_path} not found")
        
        # Load imputed data (with -1 for missing values)
        imputed_path = os.path.join(data_dir, f'{split_type}_imputed.csv')
        if os.path.exists(imputed_path):
            splits[split_type]['imputed'] = pd.read_csv(imputed_path).values
            print(f"  ✓ {split_type}_imputed.csv: {splits[split_type]['imputed'].shape}")
        else:
            print(f"  ✗ {imputed_path} not found")
            
        # Load missing mask
        mask_path = os.path.join(data_dir, f'{split_type}_missing_mask.csv')
        if os.path.exists(mask_path):
            splits[split_type]['missing_mask'] = pd.read_csv(mask_path).values.astype(bool)
            print(f"  ✓ {split_type}_missing_mask.csv: {splits[split_type]['missing_mask'].shape}")
        else:
            print(f"  ✗ {mask_path} not found")
    
    return splits

# Calculate metrics for missing values only
def calculate_jsd_and_mean_diff(imputed_values, ground_truth_values, feature_name):
    """
    Calculate Jensen-Shannon Divergence and mean difference between imputed and ground truth values.
    """
    if len(imputed_values) == 0 or len(ground_truth_values) == 0:
        return np.nan, np.nan
    
    # Remove any NaN or infinite values
    imputed_clean = imputed_values[np.isfinite(imputed_values)]
    gt_clean = ground_truth_values[np.isfinite(ground_truth_values)]
    
    if len(imputed_clean) == 0 or len(gt_clean) == 0:
        return np.nan, np.nan
    
    # Calculate mean difference
    mean_diff = abs(imputed_clean.mean() - gt_clean.mean())
    
    # Calculate Jensen-Shannon divergence using histograms
    try:
        # Create histograms with same bins
        data_range = (min(imputed_clean.min(), gt_clean.min()), 
                     max(imputed_clean.max(), gt_clean.max()))
        
        if data_range[1] == data_range[0]:
            return mean_diff, 0.0  # No divergence if all values are the same
        
        bins = np.linspace(data_range[0], data_range[1], 50)
        
        # Get histogram probabilities
        hist_imputed, _ = np.histogram(imputed_clean, bins=bins, density=True)
        hist_gt, _ = np.histogram(gt_clean, bins=bins, density=True)
        
        # Normalize to probabilities
        hist_imputed = hist_imputed + 1e-10  # Add small epsilon to avoid zeros
        hist_gt = hist_gt + 1e-10
        hist_imputed = hist_imputed / hist_imputed.sum()
        hist_gt = hist_gt / hist_gt.sum()
        
        # Calculate Jensen-Shannon divergence
        js_div = jensenshannon(hist_imputed, hist_gt)
        
        return mean_diff, js_div
        
    except Exception as e:
        print(f"Error calculating JSD for {feature_name}: {e}")
        return mean_diff, np.nan


def plot_distribution_comparison(last_4_features, feature_metrics, test_imputations_denorm, test_original_denorm, test_masks, feature_names):
    # Create distribution comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, feature_name in enumerate(last_4_features):        
        # Get feature index
        feature_idx = feature_names.index(feature_name)
        
        # Get imputed and ground truth values for missing positions only
        missing_positions = (test_masks[:, feature_idx] == 0)  # 0 = missing in model tensors
        
        if missing_positions.sum() > 0:
            imputed_values = test_imputations_denorm[missing_positions, feature_idx]
            ground_truth_values = test_original_denorm[missing_positions, feature_idx]
            
            # Remove any NaN or infinite values
            valid_mask = np.isfinite(imputed_values) & np.isfinite(ground_truth_values)
            imputed_clean = imputed_values[valid_mask]
            gt_clean = ground_truth_values[valid_mask]
            
            if len(imputed_clean) > 0 and len(gt_clean) > 0:
                # Create histograms
                ax = axes[idx]
                
                # Calculate bins for both distributions
                all_values = np.concatenate([imputed_clean, gt_clean])
                bins = np.linspace(all_values.min(), all_values.max(), 30)
                
                # Plot histograms
                ax.hist(gt_clean, bins=bins, alpha=0.7, label='Ground Truth', 
                    color='skyblue', density=True, edgecolor='black', linewidth=0.5)
                ax.hist(imputed_clean, bins=bins, alpha=0.7, label='Imputed', 
                    color='lightcoral', density=True, edgecolor='black', linewidth=0.5)
                
                # Add statistical information
                gt_mean, gt_std = gt_clean.mean(), gt_clean.std()
                imp_mean, imp_std = imputed_clean.mean(), imputed_clean.std()
                correlation = np.corrcoef(gt_clean, imputed_clean)[0, 1] if len(gt_clean) > 1 else 0
                
                # Set labels and title
                ax.set_xlabel(f'{feature_name} Values')
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add vertical lines for means
                ax.axvline(gt_mean, color='blue', linestyle='--', alpha=0.8, label='GT Mean')
                ax.axvline(imp_mean, color='red', linestyle='--', alpha=0.8, label='Imp Mean')
                
                # Add performance metrics text box
                metrics = feature_metrics[feature_name]

            else:
                axes[idx].text(0.5, 0.5, f'{feature_name}\nNo valid data', 
                            ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{feature_name} - No Valid Data')
        else:
            axes[idx].text(0.5, 0.5, f'{feature_name}\nNo missing values', 
                        ha='center', va='center', transform=axes[idx].transAxes)
            axes[idx].set_title(f'{feature_name} - No Missing Values')

    # Hide any unused subplots
    for idx in range(len(last_4_features), 4):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.show()

    # Additional statistical comparison table
    print(f"\nDetailed Statistical Comparison:")
    print(f"{'Feature':<15} {'GT Mean':<10} {'Imp Mean':<10} {'GT Std':<10} {'Imp Std':<10} {'Mean Diff':<10} {'Corr':<8}")
    print("-" * 85)

    for feature_name in last_4_features:
        feature_idx = feature_names.index(feature_name)
        missing_positions = (test_masks[:, feature_idx] == 0)  # 0 = missing in model tensors
        
        if missing_positions.sum() > 0:
            imputed_values = test_imputations_denorm[missing_positions, feature_idx]
            ground_truth_values = test_original_denorm[missing_positions, feature_idx]
            
            valid_mask = np.isfinite(imputed_values) & np.isfinite(ground_truth_values)
            imputed_clean = imputed_values[valid_mask]
            gt_clean = ground_truth_values[valid_mask]
            
            if len(imputed_clean) > 0 and len(gt_clean) > 0:
                gt_mean, gt_std = gt_clean.mean(), gt_clean.std()
                imp_mean, imp_std = imputed_clean.mean(), imputed_clean.std()
                mean_diff = abs(gt_mean - imp_mean)
                correlation = np.corrcoef(gt_clean, imputed_clean)[0, 1] if len(gt_clean) > 1 else 0
                
                print(f"{feature_name:<15} {gt_mean:<10.4f} {imp_mean:<10.4f} {gt_std:<10.4f} "
                    f"{imp_std:<10.4f} {mean_diff:<10.4f} {correlation:<8.3f}")

def plot_prediction_scatter(test_imputations, test_originals, test_masks, feature_names, n_features=8):
    """
    Create a nice scatter plot showing predicted vs ground truth values for random features.
    
    Args:
        test_imputations: Model predictions [n_samples, n_features]
        test_originals: Ground truth values [n_samples, n_features]
        test_masks: Binary masks (1=observed, 0=missing) [n_samples, n_features]
        feature_names: List of feature names
        n_features: Number of random features to plot
    """
    # Create masks for missing values (where we need to evaluate imputation)
    missing_mask = (test_masks == 0)  # True where values were missing
    
    # Find features that have missing values
    features_with_missing = [i for i in range(len(feature_names)) if missing_mask[:, i].sum() > 0]
    
    if len(features_with_missing) < n_features:
        n_features = len(features_with_missing)
        print(f"Only {n_features} features have missing values, showing all of them.")
    
    # Randomly select features to plot
    np.random.seed(42)  # For reproducible selection
    selected_features = np.random.choice(features_with_missing, size=n_features, replace=False)
    
    # Create subplots
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Predicted vs Ground Truth Values (Missing Positions Only)', fontsize=16, fontweight='bold')
    
    axes = axes.flatten()
    
    for idx, feature_idx in enumerate(selected_features):
        ax = axes[idx]
        
        # Get missing positions for this feature
        feature_missing_mask = missing_mask[:, feature_idx]
        
        if feature_missing_mask.sum() == 0:
            continue
            
        # Get predicted and ground truth values for missing positions only
        predicted_values = test_imputations[feature_missing_mask, feature_idx]
        true_values = test_originals[feature_missing_mask, feature_idx]
        
        # Create scatter plot
        ax.scatter(true_values, predicted_values, alpha=0.6, s=20, color='steelblue', edgecolors='navy', linewidth=0.5)
        
        # Add perfect prediction line (y=x)
        min_val = min(true_values.min(), predicted_values.min())
        max_val = max(true_values.max(), predicted_values.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
        
        # Calculate and display metrics
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        try:
            correlation = np.corrcoef(true_values, predicted_values)[0, 1]
        except:
            correlation = np.nan
        
        # Set labels and title
        ax.set_xlabel('Ground Truth', fontsize=10)
        ax.set_ylabel('Predicted', fontsize=10)
        ax.set_title(f'{feature_names[feature_idx]}\n'
                    f'R²={correlation:.3f}, MSE={mse:.4f}', fontsize=10, fontweight='bold')
        
        # Add grid and styling
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Add text box with number of missing values
        n_missing = feature_missing_mask.sum()
        ax.text(0.05, 0.95, f'n={n_missing}', transform=ax.transAxes, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                verticalalignment='top', fontsize=9)
    
    # Add legend to the last subplot
    axes[-1].legend(loc='lower right', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 Prediction Quality Summary (8 Random Features):")
    print("="*60)
    
    all_correlations = []
    all_mses = []
    all_maes = []
    
    for feature_idx in selected_features:
        feature_missing_mask = missing_mask[:, feature_idx]
        if feature_missing_mask.sum() == 0:
            continue
            
        predicted_values = test_imputations[feature_missing_mask, feature_idx]
        true_values = test_originals[feature_missing_mask, feature_idx]
        
        mse = mean_squared_error(true_values, predicted_values)
        mae = mean_absolute_error(true_values, predicted_values)
        try:
            correlation = np.corrcoef(true_values, predicted_values)[0, 1]
            if not np.isnan(correlation):
                all_correlations.append(correlation)
        except:
            pass
        
        all_mses.append(mse)
        all_maes.append(mae)
        
        print(f"{feature_names[feature_idx]:<15} | R²: {correlation:.3f} | MSE: {mse:.4f} | MAE: {mae:.4f} | Missing: {feature_missing_mask.sum()}")
    
    print("="*60)
    print(f"Average R²: {np.mean(all_correlations):.3f} ± {np.std(all_correlations):.3f}")
    print(f"Average MSE: {np.mean(all_mses):.4f} ± {np.std(all_mses):.4f}")
    print(f"Average MAE: {np.mean(all_maes):.4f} ± {np.std(all_maes):.4f}")


def generate_test2_samples_vae(model, X_test2, test2_loader, device, n_samples_per_test2=100):
    """Generate multiple samples for Test2 dataset using the trained VAE model.
    """
    # We'll generate multiple samples to capture uncertainty
    test2_samples = np.zeros((X_test2.shape[0], n_samples_per_test2, X_test2.shape[1]))

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        # Create a progress bar for all test2 samples
        from tqdm import tqdm
        
        for batch_idx, (batch_data, batch_mask) in enumerate(tqdm(test2_loader, desc="Generating Test2 Samples")):
            batch_data = batch_data.to(device)
            batch_mask = batch_mask.to(device)
            
            # Calculate the indices for this batch
            start_idx = batch_idx * test2_loader.batch_size
            end_idx = min(start_idx + test2_loader.batch_size, X_test2.shape[0])
            actual_batch_size = end_idx - start_idx
            
            # Generate multiple samples for each item in the batch
            for j in range(n_samples_per_test2):
                # Get reconstruction and uncertainty
                reconstruction, mu, logvar, uncertainty = model(batch_data, batch_mask)
                
                # Apply mask: keep original values where available, use reconstructed values where missing
                mask_float = batch_mask.float()
                imputed = batch_data * mask_float + reconstruction * (1 - mask_float)
                
                # Store the samples (already in original scale since we didn't normalize)
                test2_samples[start_idx:end_idx, j, :] = imputed.cpu().numpy()
    print(f"✓ Generated samples shape: {test2_samples.shape}")
    print(f"  - {test2_samples.shape[0]} test2 samples")
    print(f"  - {test2_samples.shape[1]} generated variations per sample")
    print(f"  - {test2_samples.shape[2]} features per sample")

    # Data is already in original scale (no denormalization needed)
    test2_samples_final = test2_samples.copy()

    # Calculate summary statistics
    mean_across_samples = test2_samples_final.mean(axis=1)  # Mean across the 100 samples

    print(f"  - Range of means: [{mean_across_samples.min():.4f}, {mean_across_samples.max():.4f}]")

    return test2_samples

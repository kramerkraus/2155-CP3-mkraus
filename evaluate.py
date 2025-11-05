#!/usr/bin/env python3
"""
Simple Evaluation Script for Generated Car Design Data
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import jensenshannon


def calculate_js_divergence(predicted, ground_truth):
    """Calculate Jensen-Shannon divergence between two distributions."""
    try:
        # Create histograms with same bins
        combined_data = np.concatenate([predicted, ground_truth])
        bins = np.linspace(combined_data.min(), combined_data.max(), 50)
        
        hist_pred, _ = np.histogram(predicted, bins=bins, density=True)
        hist_true, _ = np.histogram(ground_truth, bins=bins, density=True)
        
        # Normalize to make them probability distributions
        hist_pred = hist_pred / (hist_pred.sum() + 1e-10)
        hist_true = hist_true / (hist_true.sum() + 1e-10)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        hist_pred = hist_pred + epsilon
        hist_true = hist_true + epsilon
        
        # Calculate JS divergence
        js_div = jensenshannon(hist_pred, hist_true) ** 2
        return js_div
        
    except Exception as e:
        return np.nan


def load_data(set_name='test2'):
    """Load ground truth test data and masks."""
    print("Loading data...")
    
    # Load test2 data from current directory
    try:
        test2_original = pd.read_csv(f'dataset/{set_name}_original.csv').values
        test2_mask = pd.read_csv(f'dataset/{set_name}_missing_mask.csv').values.astype(bool)
        print(f"✓ Found data in current directory")
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find {set_name}_original.csv and {set_name}_missing_mask.csv")

    print(f"✓ Test data shape: {test2_original.shape}")
    print(f"✓ Missing ratio: {test2_mask.mean()*100:.1f}%")
    
    return test2_original, test2_mask


def evaluate_samples(generated_samples, set_name='test2'):
    """Evaluate generated samples against ground truth."""    
    # Load data
    test2_original, test2_mask = load_data(set_name=set_name)
    
    print(f"✓ Generated samples shape: {generated_samples.shape}")
    
    # Use mean of generated samples
    generated_mean = generated_samples.mean(axis=1)
    
    # Calculate metrics for missing values only
    missing_positions = test2_mask  # True = missing
    
    if missing_positions.sum() == 0:
        print("No missing values found!")
        return
    
    # Get all missing values
    true_missing = test2_original[missing_positions]
    pred_missing = generated_mean[missing_positions]
    
    # Calculate metrics
    mse = mean_squared_error(true_missing, pred_missing)
    mae = mean_absolute_error(true_missing, pred_missing)
    correlation = np.corrcoef(true_missing, pred_missing)[0, 1]
    js_divergence = calculate_js_divergence(pred_missing, true_missing)
    
    print(f"\n📊 EVALUATION RESULTS")
    print(f"=" * 50)
    print(f"Missing values evaluated: {len(true_missing):,}")
    print(f"MSE          : {mse:.4f}")
    print(f"MAE          : {mae:.4f}")
    print(f"Correlation  : {correlation:.3f}")
    print(f"JS Divergence: {js_divergence:.4f}")
    
    # Feature-wise results
    print(f"\n📈 FEATURE-WISE RESULTS")
    print(f"=" * 80)
    print(f"{'Feature':<8} {'MSE':<10} {'MAE':<10} {'Corr':<8} {'JSDiv':<8} {'Count':<8}")
    print(f"=" * 80)
    
    # Collect metrics for averaging
    feature_metrics = []
    
    n_features = test2_original.shape[1]
    for i in range(min(10, n_features)):  # Show first 10 features
        feature_missing = test2_mask[:, i]
        if feature_missing.sum() == 0:
            continue
            
        true_feat = test2_original[feature_missing, i]
        pred_feat = generated_mean[feature_missing, i]
        
        feat_mse = mean_squared_error(true_feat, pred_feat)
        feat_mae = mean_absolute_error(true_feat, pred_feat)
        feat_corr = np.corrcoef(true_feat, pred_feat)[0, 1]
        feat_js_div = calculate_js_divergence(pred_feat, true_feat)
        
        # Store metrics for averaging
        feature_metrics.append({
            'mse': feat_mse,
            'mae': feat_mae,
            'corr': feat_corr,
            'js_div': feat_js_div
        })
        
        print(f"{i:<8} {feat_mse:<10.4f} {feat_mae:<10.4f} "
              f"{feat_corr:<8.3f} {feat_js_div:<8.4f} {len(true_feat):<8}")
    
    # Calculate and display mean metrics across features
    if feature_metrics:
        mean_mse = np.mean([m['mse'] for m in feature_metrics])
        mean_mae = np.mean([m['mae'] for m in feature_metrics])
        mean_corr = np.mean([m['corr'] for m in feature_metrics])
        mean_js_div = np.mean([m['js_div'] for m in feature_metrics if not np.isnan(m['js_div'])])
        
        print(f"=" * 80)
        print(f"{'MEAN':<8} {mean_mse:<10.4f} {mean_mae:<10.4f} "
              f"{mean_corr:<8.3f} {mean_js_div:<8.4f} {len(feature_metrics):<8}")

        return {
            'mean_mse': mean_mse,
            'mean_mae': mean_mae,
            'mean_correlation': mean_corr,
            'mean_js_divergence': mean_js_div
        }


def compute_score(generated_samples, set_name='test2'):
    """Main function."""
    mean_metrics = evaluate_samples(generated_samples, set_name=set_name)
    print("\n✅ Evaluation completed!")
    
    if mean_metrics:
        print(f"\n📋 SUMMARY - Mean metrics across features:")
        print(f"Mean MSE: {mean_metrics['mean_mse']:.4f}")
        print(f"Mean Correlation: {mean_metrics['mean_correlation']:.3f}")
        print(f"Mean JS Divergence: {mean_metrics['mean_js_divergence']:.4f}")

    return  mean_metrics['mean_correlation'] - mean_metrics['mean_js_divergence'] - mean_metrics['mean_mse']


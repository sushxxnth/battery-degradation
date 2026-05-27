#!/usr/bin/env python3
"""
Battery Degradation Robustness Study
Runs a 10-fold Monte Carlo seed robustness study across all 6 models
to evaluate statistical stability (Mean ± Std Dev) and generate publication-ready plots.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time, os, warnings
warnings.filterwarnings('ignore')

from battery_data_loader import load_all_data
from battery_symbolic_sindy import (
    prepare_data, scale, run_sr1, run_sr2, run_sindy1, run_sindy2, run_rf, run_lstm,
    USED_FEATURES, SR_FEATURES, REPORTS_DIR
)

# Set styling for publication-grade charts
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.titlesize': 18,
    'legend.fontsize': 12
})

# Tailored harmonic color palette
COLORS = {
    'SR-1 (SOH)': '#E63946',
    'SR-2 (SOH)': '#F4A261',
    'SINDy-1 (SOH)': '#2A9D8F',
    'SINDy-2 (SOH)': '#457B9D',
    'RF (SOH)': '#9B5DE5',
    'LSTM (SOH)': '#00F5D4'
}

def main():
    print("=" * 60)
    print("  Battery Degradation: 10-Fold Seed Robustness Study")
    print("=" * 60)

    # 1. Load data
    df = load_all_data()
    train_df, test_df = prepare_data(df)
    
    # SOH Features
    feat_cols = [c for c in USED_FEATURES if c != 'cycle_number']
    sr_feat_cols = [c for c in SR_FEATURES if c != 'cycle_number']

    X_train, X_test, _ = scale(train_df, test_df, feat_cols)
    X_train_sr, X_test_sr, _ = scale(train_df, test_df, sr_feat_cols)
    
    # OPTION A Fix: Isolate 'capacity_fade' for SR-1 so it doesn't get stuck on a constant
    sr1_feat_idx = [sr_feat_cols.index('capacity_fade')]
    X_train_sr1 = X_train_sr[:, sr1_feat_idx]
    X_test_sr1 = X_test_sr[:, sr1_feat_idx]

    y_train, y_test = train_df['SOH'].values, test_df['SOH'].values

    # Seeds to evaluate (10 distinct seeds)
    SEEDS = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    results_list = []

    print(f"\nStarting robustness evaluation across {len(SEEDS)} runs...")
    
    for run_idx, seed in enumerate(SEEDS, 1):
        print(f"\n[Run {run_idx}/{len(SEEDS)}] Evaluating with Seed = {seed}")
        print("-" * 50)
        
        # Train & Evaluate SR models
        _, _, m_sr1 = run_sr1(X_train_sr1, y_train, X_test_sr1, y_test, target='SOH', seed=seed)
        _, _, m_sr2 = run_sr2(X_train_sr, y_train, X_test_sr, y_test, target='SOH', seed=seed)
        
        # Train & Evaluate SINDy models (deterministic solves)
        _, _, m_sindy1 = run_sindy1(train_df, test_df)
        _, _, m_sindy2 = run_sindy2(train_df, test_df)
        
        # Train & Evaluate baselines
        _, _, m_rf = run_rf(X_train, y_train, X_test, y_test, target='SOH', seed=seed)
        _, _, m_lstm = run_lstm(train_df, test_df, feat_cols, target='SOH', epochs=100, seed=seed)
        
        # Collect metrics
        for m in [m_sr1, m_sr2, m_sindy1, m_sindy2, m_rf, m_lstm]:
            m['seed'] = seed
            m['run'] = run_idx
            results_list.append(m)

    # 2. Aggregate Results
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(os.path.join(REPORTS_DIR, "robustness_raw_results.csv"), index=False)
    print(f"\nRaw results saved to: {os.path.join(REPORTS_DIR, 'robustness_raw_results.csv')}")

    # Compute statistics
    summary_stats = results_df.groupby('model').agg({
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'R2': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).reset_index()

    summary_stats.columns = ['model', 'RMSE_mean', 'RMSE_std', 'MAE_mean', 'MAE_std', 'R2_mean', 'R2_std', 'MAPE_mean', 'MAPE_std']
    summary_stats.to_csv(os.path.join(REPORTS_DIR, "robustness_summary.csv"), index=False)
    
    # Sort by R2 mean
    summary_stats_sorted = summary_stats.sort_values(by='R2_mean', ascending=False)

    print("\n" + "=" * 80)
    print("  Robustness Results: SOH Prediction Metrics (Mean ± Std Dev)")
    print("=" * 80)
    print(f"{'Model':<15} {'RMSE (mean±std)':<20} {'MAE (mean±std)':<20} {'R² (mean±std)':<20} {'MAPE (mean±std)':<15}")
    print("-" * 95)
    for _, row in summary_stats_sorted.iterrows():
        print(f"{row['model']:<15} "
              f"{row['RMSE_mean']:.4f} ± {row['RMSE_std']:.4f}  "
              f"{row['MAE_mean']:.4f} ± {row['MAE_std']:.4f}  "
              f"{row['R2_mean']:.4f} ± {row['R2_std']:.4f}  "
              f"{row['MAPE_mean']:.2f}% ± {row['MAPE_std']:.2f}%")
    print("=" * 80)

    # 3. Generate Visualizations
    ordered_models = summary_stats_sorted['model'].tolist()
    
    # Chart 1: Bar Chart (R² + RMSE)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    colors_list = [COLORS[m] for m in ordered_models]

    bars1 = ax1.bar(
        summary_stats_sorted['model'].str.replace(' \(SOH\)', '', regex=True),
        summary_stats_sorted['R2_mean'],
        yerr=summary_stats_sorted['R2_std'],
        capsize=6, color=colors_list, edgecolor='black', alpha=0.88, width=0.55
    )
    ax1.set_title("$R^2$ Score (Mean ± Std Dev)", pad=15, weight='bold', fontsize=15)
    ax1.set_ylabel("$R^2$ Score", weight='bold')
    ax1.set_ylim(0.5, 1.03)
    ax1.axhline(y=0.95, color='gray', linestyle='--', alpha=0.4)
    ax1.tick_params(axis='x', rotation=10)
    for bar, r2, std in zip(bars1, summary_stats_sorted['R2_mean'], summary_stats_sorted['R2_std']):
        label = f"{r2:.4f}\n±{std:.4f}" if std > 1e-6 else f"{r2:.4f}\n(deterministic)"
        y_pos = bar.get_height() - 0.04 if r2 > 0.97 else bar.get_height() + 0.01
        color = 'white' if r2 > 0.97 else 'black'
        ax1.text(bar.get_x() + bar.get_width()/2, y_pos, label,
                 ha='center', va='bottom', fontsize=9, weight='bold', color=color)

    bars2 = ax2.bar(
        summary_stats_sorted['model'].str.replace(' \(SOH\)', '', regex=True),
        summary_stats_sorted['RMSE_mean'],
        yerr=summary_stats_sorted['RMSE_std'],
        capsize=6, color=colors_list, edgecolor='black', alpha=0.88, width=0.55
    )
    ax2.set_title("RMSE (Mean ± Std Dev)", pad=15, weight='bold', fontsize=15)
    ax2.set_ylabel("RMSE (lower = better)", weight='bold')
    ax2.tick_params(axis='x', rotation=10)
    for bar, rmse, std in zip(bars2, summary_stats_sorted['RMSE_mean'], summary_stats_sorted['RMSE_std']):
        label = f"{rmse:.4f}\n±{std:.4f}" if std > 1e-6 else f"{rmse:.4f}"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 label, ha='center', va='bottom', fontsize=9, weight='bold')

    fig.suptitle("SOH Prediction Robustness: 10-Fold Seed Sensitivity Analysis",
                 fontsize=17, weight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, "robustness_error_bars.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Chart 2: Box plot
    fig2, ax3 = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=results_df, x='model', y='R2', order=ordered_models,
                palette=COLORS, width=0.5, linewidth=2, fliersize=6, ax=ax3)
    sns.stripplot(data=results_df, x='model', y='R2', order=ordered_models,
                  color='black', alpha=0.5, size=6, jitter=0.15, ax=ax3)
    ax3.set_title("Distribution of $R^2$ Scores Across 10 Independent Seeds", weight='bold', fontsize=16, pad=15)
    ax3.set_ylabel("$R^2$ Score", weight='bold')
    ax3.set_xlabel("")
    ax3.set_ylim(0.5, 1.02)
    ax3.set_xticklabels([m.replace(' (SOH)', '') for m in ordered_models], rotation=15)
    fig2.tight_layout()
    fig2.savefig(os.path.join(REPORTS_DIR, "robustness_box_plot.png"), dpi=300, bbox_inches='tight')
    plt.close(fig2)

if __name__ == "__main__":
    main()

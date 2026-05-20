

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EOL_THRESHOLD = 0.80

PALETTE = {
    'SR-1':    '#E63946',
    'SR-2':    '#F4A261',
    'SINDy-1': '#2A9D8F',
    'SINDy-2': '#457B9D',
    'RF':      '#9B5DE5',
    'LSTM':    '#00F5D4',
    'Actual':  '#264653',
}

MODEL_NAMES = ['SR-1', 'SR-2', 'SINDy-1', 'SINDy-2', 'RF', 'LSTM']


def main():
    npz_path = os.path.join(REPORTS_DIR, 'saved_predictions.npz')
    if not os.path.exists(npz_path):
        print(f"ERROR: {npz_path} not found.")
        print("Run 'python3 battery_symbolic_sindy.py' first to generate predictions.")
        return

    print("=" * 60)
    print("  Reproducing Results from Saved Predictions")
    print("=" * 60)

    data = np.load(npz_path, allow_pickle=True)

    test_soh = data['test_soh']
    test_rul = data['test_rul']
    test_cycle = data['test_cycle']
    test_cell_id = data['test_cell_id']
    test_source = data['test_source']

    # Reconstruct predictions dict
    predictions = {}
    for name in MODEL_NAMES:
        key = f'pred_{name}'
        if key in data:
            predictions[name] = data[key]

    # Reconstruct RUL results
    rul_results = {}
    for name in MODEL_NAMES:
        true_key = f'rul_true_{name}'
        pred_key = f'rul_pred_{name}'
        if true_key in data and pred_key in data:
            rul_results[name] = (data[true_key], data[pred_key])

    # ─── Print metrics ───────────────────────────────────────────────────
    print("\n=== SOH Prediction Results ===")
    print(f"{'Model':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'MAPE':<10}")
    soh_rows = []
    for name, preds in predictions.items():
        valid = np.isfinite(preds)
        if sum(valid) > 0:
            rmse = np.sqrt(mean_squared_error(test_soh[valid], preds[valid]))
            mae = mean_absolute_error(test_soh[valid], preds[valid])
            r2 = r2_score(test_soh[valid], preds[valid])
            mape = np.mean(np.abs((test_soh[valid] - preds[valid]) / (test_soh[valid] + 1e-10))) * 100
            print(f"{name:<15} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f} {mape:<10.2f}%")
            soh_rows.append({'model': f'{name} (SOH)', 'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape})

    print("\n=== Per-Dataset SOH Metrics ===")
    print(f"{'Model':<15} {'Dataset':<8} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
    for name, preds in predictions.items():
        for source in ['NASA', 'CALCE']:
            mask = test_source == source
            valid = mask & np.isfinite(preds)
            if sum(valid) > 0:
                rmse = np.sqrt(mean_squared_error(test_soh[valid], preds[valid]))
                mae = mean_absolute_error(test_soh[valid], preds[valid])
                r2 = r2_score(test_soh[valid], preds[valid])
                print(f"{name:<15} {source:<8} {rmse:<10.4f} {mae:<10.4f} {r2:<10.4f}")

    print("\n=== RUL Prediction Results ===")
    for name, (true, pred) in rul_results.items():
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        print(f"{name:<15} RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}")

    # ─── Plot 1: SOH prediction curves ───────────────────────────────────
    cell_ids = np.unique(test_cell_id)
    n_cells = len(cell_ids)
    cols = min(3, n_cells)
    rows = (n_cells + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, cid in enumerate(cell_ids):
        ax = axes_flat[i]
        mask = test_cell_id == cid
        cycles = test_cycle[mask]
        order = np.argsort(cycles)
        cycles = cycles[order]

        ax.plot(cycles, test_soh[mask][order], color=PALETTE['Actual'],
                linewidth=2.5, label='Actual SOH', zorder=5)
        for name, preds in predictions.items():
            p = preds[mask][order]
            ax.plot(cycles, p, linewidth=1.5, label=name,
                    color=PALETTE.get(name), alpha=0.85)
        ax.axhline(EOL_THRESHOLD, color='gray', linestyle='--', linewidth=1, label='EOL (80%)')
        ax.set_xlabel('Cycle Number')
        ax.set_ylabel('SOH')
        ax.set_title(f'Cell {cid}', fontweight='bold')
        ax.set_ylim(0.5, 1.1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, 'soh_prediction_curves.png'), dpi=150)
    plt.close(fig)
    print("\n  Saved: soh_prediction_curves.png")

    # ─── Plot 2: Metrics bar chart ───────────────────────────────────────
    soh_df = pd.DataFrame(soh_rows)
    models = soh_df['model'].tolist()
    x = np.arange(len(models))
    colors = [PALETTE.get(m.split(' ')[0], '#888') for m in models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    bars1 = ax1.bar(x, soh_df['RMSE'], 0.35, color=colors, edgecolor='white')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax1.set_ylabel('RMSE (lower = better)')
    ax1.set_title('RMSE Comparison', fontweight='bold')
    ax1.bar_label(bars1, fmt='%.4f', fontsize=9, padding=3)
    ax1.grid(True, axis='y', alpha=0.3)

    bars2 = ax2.bar(x, soh_df['R2'], 0.35, color=colors, edgecolor='white')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax2.set_ylabel('R² Score (higher = better)')
    ax2.set_title('R² Comparison', fontweight='bold')
    ax2.bar_label(bars2, fmt='%.4f', fontsize=9, padding=3)
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.4)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, 'metrics_comparison.png'), dpi=150)
    plt.close(fig)
    print("  Saved: metrics_comparison.png")

    # ─── Plot 3: Error histograms ────────────────────────────────────────
    n = len(predictions)
    fig, axes_err = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes_err = [axes_err]
    for ax, (name, preds) in zip(axes_err, predictions.items()):
        errors = test_soh - preds
        errors = errors[np.isfinite(errors)]
        ax.hist(errors, bins=30, color=PALETTE.get(name, '#888'), edgecolor='white', alpha=0.85)
        ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Error (Actual − Predicted)')
        ax.grid(True, alpha=0.3)
    axes_err[0].set_ylabel('Count')
    fig.suptitle('Prediction Error Distribution', fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, 'error_histograms.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("  Saved: error_histograms.png")

    # ─── Plot 4: RUL scatter ─────────────────────────────────────────────
    if rul_results:
        fig, axes_rul = plt.subplots(1, len(rul_results), figsize=(6 * len(rul_results), 5))
        if len(rul_results) == 1:
            axes_rul = [axes_rul]
        for ax, (name, (true, pred)) in zip(axes_rul, rul_results.items()):
            ax.scatter(true, pred, color=PALETTE.get(name, '#888'), alpha=0.7, s=60, edgecolor='white')
            lim = max(max(true), max(pred)) * 1.1
            ax.plot([0, lim], [0, lim], 'k--', linewidth=1.2, label='Perfect')
            rmse = np.sqrt(mean_squared_error(true, pred))
            mae = mean_absolute_error(true, pred)
            ax.set_title(f'{name} — RUL\nRMSE={rmse:.1f}  MAE={mae:.1f}', fontweight='bold')
            ax.set_xlabel('Actual RUL (cycles)')
            ax.set_ylabel('Predicted RUL (cycles)')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(REPORTS_DIR, 'rul_comparison.png'), dpi=150)
        plt.close(fig)
        print("  Saved: rul_comparison.png")

    # ─── Plot 5: Parity plot ─────────────────────────────────────────────
    fig, axes_par = plt.subplots(1, len(predictions), figsize=(5 * len(predictions), 5))
    if len(predictions) == 1:
        axes_par = [axes_par]
    for ax, (name, preds) in zip(axes_par, predictions.items()):
        ax.scatter(test_soh, preds, alpha=0.5, color=PALETTE.get(name, '#888'), edgecolor='white')
        ax.plot([0.5, 1.1], [0.5, 1.1], 'k--', alpha=0.75, label='Perfect')
        ax.set_title(f"{name} Parity", fontweight='bold')
        ax.set_xlabel('Actual SOH')
        ax.set_ylabel('Predicted SOH')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, 'parity_plot.png'), dpi=150)
    plt.close(fig)
    print("  Saved: parity_plot.png")

    # ─── Plot 6: Correlation heatmap ─────────────────────────────────────
    corr = data['heatmap_corr']
    labels = data['heatmap_labels']
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(corr, index=labels, columns=labels),
                annot=True, cmap='coolwarm', fmt=".2f", ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(REPORTS_DIR, 'feature_correlation_heatmap.png'), dpi=150)
    plt.close(fig)
    print("  Saved: feature_correlation_heatmap.png")

    print("\n" + "=" * 60)
    print("  All 6 plots regenerated from saved predictions!")
    print("  These are IDENTICAL to the original author's results.")
    print("=" * 60)


if __name__ == "__main__":
    main()

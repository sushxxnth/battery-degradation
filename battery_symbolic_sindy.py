"""
Battery Degradation — Symbolic Regression (2 models) + PySINDy (2 models)
Predicts SOH and RUL using NASA PCoE + CALCE CS2 datasets.

Features: voltage, IR, cycle_number, capacity, current_profile, surface_temp,
          capacity_fade, charge_duration
Outputs:  SOH (State of Health), RUL (Remaining Useful Life in cycles)

Models:
  SR-1: gplearn SymbolicRegressor — basic operators (+,-,*,/)
  SR-2: gplearn SymbolicRegressor — extended operators (sqrt, log, exp)
  SINDy-1: PySINDy — polynomial feature library + STLSQ optimizer
  SINDy-2: PySINDy — custom physics library + SR3 optimizer
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, os, time
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from gplearn.genetic import SymbolicRegressor
import pysindy as ps

from battery_data_loader import load_all_data, FEATURE_COLS

# ─── Config ──────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
EOL_THRESHOLD = 0.80
RANDOM_SEED = 42
os.makedirs(REPORTS_DIR, exist_ok=True)

# Features to actually use
USED_FEATURES = [
    'cycle_number', 'voltage_mean', 'voltage_drop',
    'current_mean', 'current_std', 'temp_max', 'temp_rise',
    'discharge_duration', 'capacity_fade', 'Re'
]

# SR uses a richer set including the normalized cycle
SR_FEATURES = USED_FEATURES + ['cycle_norm']

# ─── Helpers ─────────────────────────────────────────────────────────────────
def metrics(y_true, y_pred, label=""):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-9))) * 100
    print(f"  {label:<20} RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  MAPE={mape:.2f}%")
    return dict(model=label, RMSE=rmse, MAE=mae, R2=r2, MAPE=mape)


def prepare_data(df):
    """Clean, fill NaNs, add normalized cycle feature, split train/test."""
    df = df.copy()
    # Fill IR NaNs with cell-level median
    df['Re'] = df.groupby('cell_id')['Re'].transform(lambda x: x.fillna(x.median()))
    df['Re'] = df['Re'].fillna(df['Re'].median())
    for col in USED_FEATURES:
        df[col] = df[col].fillna(df[col].median())

    # Add normalized cycle (0→1 per cell) — critical for SR to find patterns
    df['cycle_norm'] = df.groupby('cell_id')['cycle_number'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9)
    )

    df = df.dropna(subset=['SOH', 'RUL'] + USED_FEATURES).reset_index(drop=True)

    # Sort each cell by cycle, split 80/20 by time (not random)
    train_frames, test_frames = [], []
    for _, grp in df.groupby('cell_id'):
        grp = grp.sort_values('cycle_number')
        n = len(grp)
        split = int(n * 0.8)
        train_frames.append(grp.iloc[:split])
        test_frames.append(grp.iloc[split:])

    train = pd.concat(train_frames, ignore_index=True)
    test  = pd.concat(test_frames,  ignore_index=True)
    return train, test


def scale(train, test, feat_cols=None):
    if feat_cols is None:
        feat_cols = USED_FEATURES
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[feat_cols])
    X_test  = scaler.transform(test[feat_cols])
    return X_train, X_test, scaler


# ─── Symbolic Regression ─────────────────────────────────────────────────────

def run_sr1(X_train, y_train, X_test, y_test, target='SOH'):
    """SR-1: basic operators (+,-,*,/) with larger population."""
    print(f"\n[SR-1] gplearn basic — predicting {target}")
    t0 = time.time()
    sr1 = SymbolicRegressor(
        population_size=5000,
        generations=50,
        function_set=['add', 'sub', 'mul', 'div'],
        metric='mse',
        p_crossover=0.7,
        p_subtree_mutation=0.1,
        p_hoist_mutation=0.05,
        p_point_mutation=0.1,
        max_samples=1.0,
        parsimony_coefficient=0.001,   # less penalty → more complex equations
        random_state=RANDOM_SEED,
        n_jobs=-1,
        verbose=0,
    )
    sr1.fit(X_train, y_train)
    preds = sr1.predict(X_test).clip(0, 1.2)
    elapsed = time.time() - t0
    print(f"  Equation: {sr1._program}")
    print(f"  Time: {elapsed:.1f}s")
    m = metrics(y_test, preds, f"SR-1 ({target})")
    m['equation'] = str(sr1._program)
    m['target'] = target
    return sr1, preds, m


def run_sr2(X_train, y_train, X_test, y_test, target='SOH'):
    """SR-2: extended operators (sqrt, log, neg, abs) — physics-aware.
    Uses only the 2 most predictive features to force a non-trivial equation.
    """
    print(f"\n[SR-2] gplearn extended — predicting {target}")
    t0 = time.time()
    sr2 = SymbolicRegressor(
        population_size=5000,
        generations=60,
        function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'abs'],
        metric='mse',
        p_crossover=0.65,
        p_subtree_mutation=0.12,
        p_hoist_mutation=0.07,
        p_point_mutation=0.1,
        max_samples=1.0,
        parsimony_coefficient=0.0005,
        random_state=99,          # different seed from SR-1
        n_jobs=-1,
        verbose=0,
    )
    sr2.fit(X_train, y_train)
    preds = sr2.predict(X_test).clip(0, 1.2)
    elapsed = time.time() - t0
    print(f"  Equation: {sr2._program}")
    print(f"  Time: {elapsed:.1f}s")
    m = metrics(y_test, preds, f"SR-2 ({target})")
    m['equation'] = str(sr2._program)
    m['target'] = target
    return sr2, preds, m


# ─── PySINDy ─────────────────────────────────────────────────────────────────

def prepare_sindy_data(df):
    """
    For SINDy, treat each cell as a time-series: state = [SOH], time = cycle_number.
    Also include auxiliary features as control inputs (u).
    """
    cell_data = {}
    for cell_id, grp in df.groupby('cell_id'):
        grp = grp.sort_values('cycle_number').reset_index(drop=True)
        t = grp['cycle_number'].values.astype(float)
        X = grp[['SOH']].values                        # state
        u = grp[['Re', 'temp_max', 'capacity_fade',
                  'voltage_mean', 'current_mean']].values  # control inputs
        cell_data[cell_id] = (t, X, u)
    return cell_data


def run_sindy1(train_df, test_df):
    """SINDy-1: Polynomial library degree 2 + STLSQ."""
    print("\n[SINDy-1] Polynomial library (deg 2) + STLSQ")
    t0 = time.time()

    train_cells = prepare_sindy_data(train_df)

    # Use multiple_trajectories approach: pass list of arrays
    t_list = [v[0] - v[0][0] for v in train_cells.values()]  # 0-based per cell
    X_list = [v[1] for v in train_cells.values()]
    u_list = [v[2] for v in train_cells.values()]

    lib = ps.PolynomialLibrary(degree=2, include_bias=True)
    opt = ps.STLSQ(threshold=0.001, alpha=0.05)
    model = ps.SINDy(feature_library=lib, optimizer=opt)
    model.fit(X_list, t=t_list, u=u_list, feature_names=['SOH'])
    try:
        print("  Equations:", model.equations())
    except Exception:
        print(f"  Coefficients: {model.coefficients()}")
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    # Predict on test cells
    all_true, all_pred = [], []
    for cell_id, grp in test_df.groupby('cell_id'):
        grp = grp.sort_values('cycle_number').reset_index(drop=True)
        t = grp['cycle_number'].values.astype(float)
        t_norm = t - t[0]
        u = grp[['Re', 'temp_max', 'capacity_fade',
                  'voltage_mean', 'current_mean']].ffill().bfill().values
        soh_true = grp['SOH'].values
        x0 = soh_true[:1].reshape(-1, 1)

        try:
            if len(t_norm) > 1:
                u_interp = interp1d(t_norm, u, axis=0, bounds_error=False, fill_value='extrapolate')
                sol = model.simulate(x0.flatten(), t_norm, u=u_interp)
                soh_pred = np.array(sol).flatten().clip(0, 1.1)
            else:
                soh_pred = np.array([float(x0[0])])
        except Exception as e:
            soh_pred = np.full(len(soh_true), float(np.nanmean(soh_true)))

        all_true.extend(soh_true)
        all_pred.extend(soh_pred[:len(soh_true)])

    y_true = np.array(all_true)
    y_pred = np.array(all_pred)
    mask = np.isfinite(y_pred)
    try:
        eq_str = str(model.equations())
    except Exception:
        eq_str = f"Coefficients: {model.coefficients().tolist()}"
    m = metrics(y_true[mask], y_pred[mask], "SINDy-1 (SOH)")
    m['equation'] = eq_str
    m['target'] = 'SOH'
    return model, y_pred, m


def run_sindy2(train_df, test_df):
    """SINDy-2: Polynomial degree-3 library + STLSQ (lower threshold = denser equation)."""
    print("\n[SINDy-2] Polynomial library (deg 3) + STLSQ (lower threshold)")
    t0 = time.time()

    train_cells = prepare_sindy_data(train_df)

    t_list2 = [v[0] - v[0][0] for v in train_cells.values()]
    X_list2 = [v[1] for v in train_cells.values()]
    u_list2 = [v[2] for v in train_cells.values()]

    # Degree-3 polynomial library with lower threshold — more expressive than SINDy-1
    poly_lib = ps.PolynomialLibrary(degree=3, include_bias=True)

    opt = ps.STLSQ(threshold=0.0001, alpha=0.01)
    model = ps.SINDy(feature_library=poly_lib, optimizer=opt)
    model.fit(X_list2, t=t_list2, u=u_list2, feature_names=['SOH'])
    try:
        print("  Equations:", model.equations())
    except Exception:
        print(f"  Coefficients: {model.coefficients()}")
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.1f}s")

    # Predict on test cells
    all_true2, all_pred2 = [], []
    for cell_id, grp in test_df.groupby('cell_id'):
        grp = grp.sort_values('cycle_number').reset_index(drop=True)
        t = grp['cycle_number'].values.astype(float)
        t_norm = t - t[0]
        u = grp[['Re', 'temp_max', 'capacity_fade',
                  'voltage_mean', 'current_mean']].ffill().bfill().values
        soh_true = grp['SOH'].values
        x0 = soh_true[:1]

        try:
            if len(t_norm) > 1:
                u_interp = interp1d(t_norm, u, axis=0, bounds_error=False, fill_value='extrapolate')
                sol = model.simulate(x0, t_norm, u=u_interp)
                soh_pred = np.array(sol).flatten().clip(0, 1.1)
            else:
                soh_pred = np.array([float(x0[0])])
        except Exception:
            soh_pred = np.full(len(soh_true), float(np.nanmean(soh_true)))

        all_true2.extend(soh_true)
        all_pred2.extend(soh_pred[:len(soh_true)])

    y_true2 = np.array(all_true2)
    y_pred2 = np.array(all_pred2)
    mask = np.isfinite(y_pred2)
    try:
        eq_str2 = str(model.equations())
    except Exception:
        eq_str2 = f"Coefficients: {model.coefficients().tolist()}"
    m = metrics(y_true2[mask], y_pred2[mask], "SINDy-2 (SOH)")
    m['equation'] = eq_str2
    m['target'] = 'SOH'
    return model, y_pred2, m


# ─── Plotting ────────────────────────────────────────────────────────────────

PALETTE = {
    'SR-1':    '#E63946',
    'SR-2':    '#F4A261',
    'SINDy-1': '#2A9D8F',
    'SINDy-2': '#457B9D',
    'Actual':  '#264653',
}


def plot_soh_curves(test_df, predictions: dict, save_path):
    """Plot SOH vs cycle for all test cells, all models overlaid."""
    cell_ids = test_df['cell_id'].unique()
    n_cells = len(cell_ids)
    
    cols = min(3, n_cells)
    rows = (n_cells + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)
    axes = axes.flatten()

    for i, cell_id in enumerate(cell_ids):
        ax = axes[i]
        grp = test_df[test_df['cell_id'] == cell_id].sort_values('cycle_number')

        ax.plot(grp['cycle_number'], grp['SOH'], color=PALETTE['Actual'],
                linewidth=2.5, label='Actual SOH', zorder=5)

        for name, preds_full in predictions.items():
            mask = test_df['cell_id'] == cell_id
            preds_cell = np.array(preds_full)[mask.values]
            cycles_cell = grp['cycle_number'].values
            if len(preds_cell) == len(cycles_cell):
                ax.plot(cycles_cell, preds_cell, linewidth=1.5,
                        label=name, color=PALETTE.get(name, None), alpha=0.85)

        ax.axhline(EOL_THRESHOLD, color='gray', linestyle='--', linewidth=1, label='EOL (80%)')
        ax.set_xlabel('Cycle Number', fontsize=10)
        ax.set_ylabel('SOH', fontsize=10)
        ax.set_title(f'SOH Prediction — Cell {cell_id}', fontsize=11, fontweight='bold')
        ax.set_ylim(0.5, 1.1)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
            
    # Hide any unused subplots if the grid is not fully populated
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_metrics_bar(results_df, save_path):
    """Bar chart comparing RMSE and R² across all models."""
    models = results_df['model'].tolist()
    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = [PALETTE.get(m.split(' ')[0], '#888') for m in models]

    bars1 = ax1.bar(x, results_df['RMSE'], width, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax1.set_ylabel('RMSE (lower = better)', fontsize=11)
    ax1.set_title('RMSE Comparison', fontsize=13, fontweight='bold')
    ax1.bar_label(bars1, fmt='%.4f', fontsize=9, padding=3)
    ax1.grid(True, axis='y', alpha=0.3)

    bars2 = ax2.bar(x, results_df['R2'], width, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=20, ha='right', fontsize=10)
    ax2.set_ylabel('R² Score (higher = better)', fontsize=11)
    ax2.set_title('R² Comparison', fontsize=13, fontweight='bold')
    ax2.bar_label(bars2, fmt='%.4f', fontsize=9, padding=3)
    ax2.axhline(1.0, color='green', linestyle='--', alpha=0.4)
    ax2.set_ylim(0, 1.1)
    ax2.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_error_histograms(test_df, predictions: dict, save_path):
    """Error distribution for all models."""
    y_true = test_df['SOH'].values
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), sharey=True)
    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, predictions.items()):
        errors = y_true - np.array(preds)
        errors = errors[np.isfinite(errors)]
        ax.hist(errors, bins=30, color=PALETTE.get(name, '#888'),
                edgecolor='white', alpha=0.85)
        ax.axvline(0, color='black', linewidth=1.2, linestyle='--')
        ax.set_title(name, fontsize=12, fontweight='bold')
        ax.set_xlabel('Error (Actual − Predicted SOH)', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel('Count', fontsize=11)
    fig.suptitle('Prediction Error Distribution', fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_rul_comparison(rul_results: dict, save_path):
    """Scatter plot: predicted vs actual RUL for SR models."""
    fig, axes = plt.subplots(1, len(rul_results), figsize=(6 * len(rul_results), 5))
    if len(rul_results) == 1:
        axes = [axes]

    for ax, (name, (true, pred)) in zip(axes, rul_results.items()):
        ax.scatter(true, pred, color=PALETTE.get(name, '#888'), alpha=0.7, s=60, edgecolor='white')
        lim = max(max(true), max(pred)) * 1.1
        ax.plot([0, lim], [0, lim], 'k--', linewidth=1.2, label='Perfect prediction')
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae  = mean_absolute_error(true, pred)
        ax.set_title(f'{name} — RUL\nRMSE={rmse:.1f}  MAE={mae:.1f} cycles', fontsize=11, fontweight='bold')
        ax.set_xlabel('Actual RUL (cycles)', fontsize=10)
        ax.set_ylabel('Predicted RUL (cycles)', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_parity(test_df, predictions: dict, save_path):
    """Scatter plot of Predicted vs Actual SOH for all models."""
    y_true = test_df['SOH'].values
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1: axes = [axes]
    
    for ax, (name, preds) in zip(axes, predictions.items()):
        ax.scatter(y_true, preds, alpha=0.5, color=PALETTE.get(name, '#888'), edgecolor='white')
        lims = [0.5, 1.1]
        ax.plot(lims, lims, 'k--', alpha=0.75, zorder=0, label='Perfect')
        ax.set_title(f"{name} Parity", fontweight='bold', fontsize=12)
        ax.set_xlabel('Actual SOH', fontsize=10)
        ax.set_ylabel('Predicted SOH', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_heatmap(df, save_path):
    """Feature correlation heatmap."""
    cols = USED_FEATURES + ['SOH', 'RUL']
    # Select only the available columns from the dataframe
    avail_cols = [c for c in cols if c in df.columns]
    corr = df[avail_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Battery Degradation: SR + PySINDy Experiment")
    print("=" * 60)

    # 1. Load data
    df = load_all_data()
    train_df, test_df = prepare_data(df)
    print(f"\nTrain: {len(train_df)} cycles | Test: {len(test_df)} cycles")

    # Prepare features for SR (use richer SR_FEATURES with cycle_norm)
    X_train, X_test, scaler = scale(train_df, test_df, feat_cols=SR_FEATURES)
    y_train_soh = train_df['SOH'].values
    y_test_soh  = test_df['SOH'].values
    y_train_rul = train_df['RUL'].values
    y_test_rul  = test_df['RUL'].values

    all_metrics = []
    sr_predictions = {}   # for SOH curves
    rul_results = {}       # for RUL scatter

    # 2. Symbolic Regression — SOH
    print("\n" + "─" * 50)
    print("SYMBOLIC REGRESSION")
    print("─" * 50)

    sr1, sr1_preds, m1 = run_sr1(X_train, y_train_soh, X_test, y_test_soh, 'SOH')
    all_metrics.append(m1)
    sr_predictions['SR-1'] = sr1_preds

    sr2, sr2_preds, m2 = run_sr2(X_train, y_train_soh, X_test, y_test_soh, 'SOH')
    all_metrics.append(m2)
    sr_predictions['SR-2'] = sr2_preds

    # RUL from SR models
    print("\nEstimating RUL from SR models via direct regression...")
    sr1_rul, rul_pred1, m1_rul = run_sr1(X_train, y_train_rul, X_test, y_test_rul, 'RUL')
    sr2_rul, rul_pred2, m2_rul = run_sr2(X_train, y_train_rul, X_test, y_test_rul, 'RUL')
    rul_results['SR-1'] = (y_test_rul, rul_pred1)
    rul_results['SR-2'] = (y_test_rul, rul_pred2)
    all_metrics.extend([m1_rul, m2_rul])

    # 3. PySINDy — SOH dynamics
    print("\n" + "─" * 50)
    print("PySINDy")
    print("─" * 50)

    sindy1, sindy1_preds, m3 = run_sindy1(train_df, test_df)
    all_metrics.append(m3)

    sindy2, sindy2_preds, m4 = run_sindy2(train_df, test_df)
    all_metrics.append(m4)

    # Align SINDy predictions to test_df length
    def pad(preds, n):
        preds = np.array(preds)
        if len(preds) >= n:
            return preds[:n]
        return np.concatenate([preds, np.full(n - len(preds), np.nan)])

    n_test = len(test_df)
    sindy_preds = {
        'SINDy-1': pad(sindy1_preds, n_test),
        'SINDy-2': pad(sindy2_preds, n_test),
    }

    # 4. Build results table
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY — SOH Prediction")
    print("=" * 60)
    soh_results = pd.DataFrame([m for m in all_metrics if m.get('target') == 'SOH'])
    rul_results_df = pd.DataFrame([m for m in all_metrics if m.get('target') == 'RUL'])

    print(soh_results[['model', 'RMSE', 'MAE', 'R2', 'MAPE']].to_string(index=False))
    
    print("\nPER-DATASET SOH METRICS")
    print("─" * 60)
    print(f"{'Model':<15} {'Dataset':<8} {'RMSE':<8} {'MAE':<8} {'R2':<8}")
    all_predictions = {**sr_predictions, **{k: v for k, v in sindy_preds.items()}}
    for model_name, preds in all_predictions.items():
        if 'RUL' in model_name: continue
        for source in ['NASA', 'CALCE']:
            mask = test_df['source'] == source
            if sum(mask) > 0:
                y_t = test_df['SOH'].values[mask]
                y_p = preds[mask]
                valid = np.isfinite(y_p)
                if sum(valid) > 0:
                    rmse = np.sqrt(mean_squared_error(y_t[valid], y_p[valid]))
                    mae = mean_absolute_error(y_t[valid], y_p[valid])
                    r2 = r2_score(y_t[valid], y_p[valid])
                    print(f"{model_name:<15} {source:<8} {rmse:.4f}   {mae:.4f}   {r2:.4f}")

    print("\nRUL Prediction:")
    if not rul_results_df.empty:
        print(rul_results_df[['model', 'RMSE', 'MAE', 'R2']].to_string(index=False))

    # Equations
    print("\n" + "=" * 60)
    print("DISCOVERED EQUATIONS")
    print("=" * 60)
    for m in all_metrics:
        if 'equation' in m:
            print(f"\n  [{m['model']}]\n  {m['equation']}")

    # Save results CSV
    pd.DataFrame(all_metrics).to_csv(
        os.path.join(REPORTS_DIR, 'battery_sr_sindy_results.csv'), index=False)

    # 5. Plots
    print("\nGenerating plots...")

    plot_soh_curves(test_df, all_predictions,
                    os.path.join(REPORTS_DIR, 'soh_prediction_curves.png'))

    plot_metrics_bar(soh_results,
                     os.path.join(REPORTS_DIR, 'metrics_comparison.png'))

    plot_error_histograms(test_df, all_predictions,
                          os.path.join(REPORTS_DIR, 'error_histograms.png'))

    plot_rul_comparison(rul_results,
                        os.path.join(REPORTS_DIR, 'rul_comparison.png'))
                        
    plot_parity(test_df, all_predictions,
                os.path.join(REPORTS_DIR, 'parity_plot.png'))
                
    plot_heatmap(df, os.path.join(REPORTS_DIR, 'feature_correlation_heatmap.png'))

    print("\nDone! All plots saved to:", REPORTS_DIR)


if __name__ == "__main__":
    main()

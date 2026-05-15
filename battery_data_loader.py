"""
Battery Data Loader
Extracts per-cycle features from NASA PCoE and CALCE CS2 datasets.
Features: voltage, IR, cycle_number, capacity, current_profile, surface_temp, capacity_fade, charge_duration
Targets: SOH, RUL
"""

import numpy as np
import pandas as pd
import scipy.io
import glob
import os
import warnings
warnings.filterwarnings('ignore')

# ─── Data Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

NASA_PATH = os.path.join(DATA_DIR, "nasa_set5", "raw")
CALCE_PATH = os.path.join(DATA_DIR, "calce", "CS2_data", "CS2_data")
EOL_THRESHOLD = 0.80   # 80% SOH = End of Life


# ─── NASA ───────────────────────────────────────────────────────────────────

def load_nasa_cell(mat_path):
    """Extract per-discharge-cycle features from one NASA .mat file."""
    mat = scipy.io.loadmat(mat_path)
    cell_id = os.path.basename(mat_path).replace(".mat", "")
    battery = mat[cell_id]
    cycles = battery['cycle'][0, 0]
    n_cycles = cycles.shape[1]

    discharge_rows = []
    charge_duration_map = {}   # index -> charge duration (s)
    impedance_map = {}         # index -> (Re, Rct)

    # First pass: collect charge durations and impedance values
    for i in range(n_cycles):
        c = cycles[0, i]
        ctype = str(c['type'][0])
        data = c['data'][0, 0]
        if ctype == 'charge':
            t = data['Time'].flatten()
            charge_duration_map[i] = float(t.max()) if len(t) > 0 else np.nan
        elif ctype == 'impedance':
            re = float(data['Re'].flatten()[0])
            rct = float(data['Rct'].flatten()[0])
            impedance_map[i] = (re, rct)

    # Helper: nearest impedance before OR after cycle i (closest wins)
    imp_indices = sorted(impedance_map.keys())

    def nearest_impedance(i):
        before = [k for k in imp_indices if k <= i]
        after  = [k for k in imp_indices if k > i]
        if before:
            return impedance_map[before[-1]]
        elif after:
            return impedance_map[after[0]]
        return np.nan, np.nan

    # Helper: nearest charge duration before cycle i
    charge_indices = sorted(charge_duration_map.keys())

    def nearest_charge_duration(i):
        candidates = [k for k in charge_indices if k <= i]
        if not candidates:
            return np.nan
        return charge_duration_map[candidates[-1]]

    # Second pass: extract discharge features
    discharge_count = 0
    first_capacity = None
    for i in range(n_cycles):
        c = cycles[0, i]
        if str(c['type'][0]) != 'discharge':
            continue
        data = c['data'][0, 0]

        v = data['Voltage_measured'].flatten()
        curr = data['Current_measured'].flatten()
        temp = data['Temperature_measured'].flatten()
        t = data['Time'].flatten()
        cap = float(data['Capacity'].flatten()[0])

        if first_capacity is None:
            first_capacity = cap

        re, rct = nearest_impedance(i)
        charge_dur = nearest_charge_duration(i)
        capacity_fade = (first_capacity - cap) / first_capacity if first_capacity else np.nan

        discharge_rows.append({
            'cell_id': cell_id,
            'source': 'NASA',
            'cycle_number': discharge_count,
            'voltage_mean': float(np.mean(v)),
            'voltage_drop': float(v[0] - v[-1]) if len(v) > 1 else np.nan,
            'current_mean': float(np.mean(np.abs(curr))),
            'current_std': float(np.std(curr)),
            'temp_max': float(np.max(temp)),
            'temp_rise': float(np.max(temp) - temp[0]) if len(temp) > 0 else np.nan,
            'discharge_duration': float(np.max(t)) if len(t) > 0 else np.nan,
            'charge_duration': charge_dur,
            'capacity': cap,
            'capacity_fade': capacity_fade,
            'Re': re,
            'Rct': rct,
        })
        discharge_count += 1

    df = pd.DataFrame(discharge_rows)
    if df.empty:
        return df

    # Compute SOH and RUL
    nominal = df['capacity'].iloc[0]
    df['SOH'] = df['capacity'] / nominal
    eol_idx = df[df['SOH'] < EOL_THRESHOLD].index
    total_cycles = len(df)
    if len(eol_idx) > 0:
        eol_cycle = df.loc[eol_idx[0], 'cycle_number']
    else:
        eol_cycle = total_cycles
    df['RUL'] = eol_cycle - df['cycle_number']
    df['RUL'] = df['RUL'].clip(lower=0)
    return df


def load_nasa_all(cells=('B0005', 'B0006', 'B0007', 'B0018')):
    """Load multiple NASA cells."""
    frames = []
    for cell in cells:
        path = os.path.join(NASA_PATH, f"{cell}.mat")
        if not os.path.exists(path):
            print(f"  [WARN] {path} not found, skipping")
            continue
        print(f"  Loading NASA {cell}...")
        df = load_nasa_cell(path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─── CALCE ───────────────────────────────────────────────────────────────────

def load_calce_file(filepath):
    """Load one CALCE cycle file (txt or xlsx)."""
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == '.txt':
            df = pd.read_csv(filepath, sep='\t', encoding='latin-1')
        else:
            df = pd.read_excel(filepath)
        # Rename columns
        df = df.rename(columns={
            'mV': 'voltage_mv', 'mA': 'current_ma',
            'Temperature': 'temperature', 'Duration': 'duration',
            'Capacity': 'capacity_raw', 'Pgm cycle': 'pgm_cycle',
            'Charge count': 'charge_count', 'Discharge count': 'discharge_count'
        })
        return df
    except Exception as e:
        print(f"  [WARN] Could not load {filepath}: {e}")
        return pd.DataFrame()


def extract_calce_cycles(df_raw, cell_id):
    """Extract per-cycle features from raw CALCE dataframe."""
    if df_raw.empty:
        return pd.DataFrame()

    df_raw = df_raw.copy()
    df_raw['voltage_V'] = pd.to_numeric(df_raw.get('voltage_mv', np.nan), errors='coerce') / 1000.0
    df_raw['current_A'] = pd.to_numeric(df_raw.get('current_ma', np.nan), errors='coerce') / 1000.0
    df_raw['temperature'] = pd.to_numeric(df_raw.get('temperature', np.nan), errors='coerce')
    df_raw['capacity_raw'] = pd.to_numeric(df_raw.get('capacity_raw', np.nan), errors='coerce')

    # Group by discharge_count if available, else treat whole file as one cycle
    if 'discharge_count' in df_raw.columns:
        df_raw['discharge_count'] = pd.to_numeric(df_raw['discharge_count'], errors='coerce').fillna(0)
        groups = df_raw[df_raw['discharge_count'] > 0].groupby('discharge_count')
    else:
        groups = [(0, df_raw)]

    rows = []
    for cycle_id, grp in groups:
        v = grp['voltage_V'].dropna().values
        curr = grp['current_A'].dropna().values
        temp = grp['temperature'].dropna().values
        cap_vals = grp['capacity_raw'].dropna().values

        if len(v) < 3 or len(curr) < 3:
            continue

        cap = float(cap_vals.max()) if len(cap_vals) > 0 else np.nan

        # IR proxy: ΔV/ΔI at start
        dv = np.abs(np.diff(v[:5])) if len(v) >= 5 else np.array([np.nan])
        di = np.abs(np.diff(curr[:5])) if len(curr) >= 5 else np.array([np.nan])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ir_proxy = float(np.nanmean(dv / (di + 1e-9)))

        rows.append({
            'cell_id': cell_id,
            'source': 'CALCE',
            'cycle_number': int(cycle_id),
            'voltage_mean': float(np.mean(v)),
            'voltage_drop': float(v.max() - v.min()),
            'current_mean': float(np.mean(np.abs(curr))),
            'current_std': float(np.std(curr)),
            'temp_max': float(np.max(temp)) if len(temp) > 0 else np.nan,
            'temp_rise': float(np.max(temp) - temp[0]) if len(temp) > 1 else np.nan,
            'discharge_duration': float(grp.get('duration', pd.Series([np.nan])).max()),
            'charge_duration': np.nan,  # will fill from charge groups
            'capacity': cap,
            'capacity_fade': np.nan,    # fill after
            'Re': ir_proxy,
            'Rct': np.nan,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values('cycle_number').reset_index(drop=True)
    first_cap = df['capacity'].dropna().iloc[0] if not df['capacity'].dropna().empty else np.nan
    if first_cap and first_cap > 0:
        df['capacity_fade'] = (first_cap - df['capacity']) / first_cap
        df['SOH'] = df['capacity'] / first_cap
    else:
        df['SOH'] = np.nan
        df['capacity_fade'] = np.nan

    eol_idx = df[df['SOH'] < EOL_THRESHOLD].index
    total = len(df)
    if len(eol_idx) > 0:
        eol_cycle = df.loc[eol_idx[0], 'cycle_number']
    else:
        eol_cycle = total
    df['RUL'] = (eol_cycle - df['cycle_number']).clip(lower=0)
    return df


def load_calce_cell(cell_name):
    """Load all files for one CALCE cell."""
    pattern_txt = os.path.join(CALCE_PATH, f"type_*/{cell_name}/{cell_name}/*.txt")
    pattern_xlsx = os.path.join(CALCE_PATH, f"type_*/{cell_name}/{cell_name}/*.xlsx")
    files = sorted(glob.glob(pattern_txt) + glob.glob(pattern_xlsx))

    if not files:
        print(f"  [WARN] No files for CALCE cell {cell_name}")
        return pd.DataFrame()

    all_raw = []
    for f in files:
        raw = load_calce_file(f)
        if not raw.empty:
            all_raw.append(raw)

    if not all_raw:
        return pd.DataFrame()

    combined = pd.concat(all_raw, ignore_index=True)
    return extract_calce_cycles(combined, cell_name)


def load_calce_all(cells=('CS2_8', 'CS2_21')):
    """Load multiple CALCE cells."""
    frames = []
    for cell in cells:
        print(f"  Loading CALCE {cell}...")
        df = load_calce_cell(cell)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ─── Combined Loader ─────────────────────────────────────────────────────────

FEATURE_COLS = [
    'cycle_number', 'voltage_mean', 'voltage_drop',
    'current_mean', 'current_std', 'temp_max', 'temp_rise',
    'discharge_duration', 'charge_duration',
    'capacity_fade', 'Re', 'Rct'
]


def load_all_data():
    """Load and merge NASA + CALCE data."""
    print("Loading NASA data...")
    nasa_df = load_nasa_all()
    print(f"  → {len(nasa_df)} discharge cycles from NASA\n")

    print("Loading CALCE data...")
    calce_df = load_calce_all()
    print(f"  → {len(calce_df)} discharge cycles from CALCE\n")

    df = pd.concat([nasa_df, calce_df], ignore_index=True)
    df = df.dropna(subset=['SOH', 'RUL'])
    df = df[df['SOH'] > 0].reset_index(drop=True)
    print(f"Combined dataset: {len(df)} cycles | {df['cell_id'].nunique()} cells")
    return df


if __name__ == "__main__":
    df = load_all_data()
    print("\nSample:")
    print(df[['cell_id', 'source', 'cycle_number', 'voltage_mean', 'Re', 'SOH', 'RUL']].head(10))
    print("\nNaN counts:\n", df[FEATURE_COLS + ['SOH', 'RUL']].isna().sum())

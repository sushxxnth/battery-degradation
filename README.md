# Physics-Informed Battery Degradation Models

This repository contains an end-to-end experimental pipeline that applies **Symbolic Regression** and **PySINDy (Sparse Identification of Nonlinear Dynamics)** to discover explicit, physics-aware mathematical equations for battery degradation. 

It predicts both **State of Health (SOH)** and **Remaining Useful Life (RUL)** across different battery chemistries and testing profiles, utilizing the NASA PCoE and CALCE CS2 datasets.

##  Key Features
- **Self-Contained Data Pipeline**: Automatically parses, cleans, and engineers features from raw `.mat` files without external dependencies.
- **Explainable AI (XAI)**: Instead of black-box neural networks, this pipeline discovers human-readable algebraic formulas governing capacity fade and internal resistance.
- **Cross-Dataset Generalization**: Trains on early-life cycles and predicts late-stage degradation accurately across both NASA and CALCE cells simultaneously.

## Results Summary

The **SINDy-1 (Degree-2 Polynomial)** model achieved state-of-the-art accuracy in tracking SOH across unseen test cells:
*   **NASA Dataset**: `RMSE = 0.0079` | `R² = 0.9843`
*   **CALCE Dataset**: `RMSE = 0.0039` | `R² = 0.8865`

### Symbolic Regression Equation Discovered
The pipeline successfully isolated the primary driver of degradation using Genetic Programming, proving that a mathematically parsimonious linear relationship with capacity fade yields robust predictions:
```text
SOH = -0.102 × capacity_fade + 0.888
```

##  How to Run Locally

### 1. Prerequisites
Ensure you have Python 3.8+ installed. 
Clone the repository to your local machine:
```bash
git clone https://github.com/sushxxnth/battery-degradation.git
cd battery-degradation
```

### 2. Install Dependencies
Install the required scientific computing and machine learning libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy gplearn pysindy
```

### 3. Execute the Experiment
Run the main script. The script automatically handles data loading, model training, RUL regression, and plot generation.
```bash
python3 battery_symbolic_sindy.py
```
*All generated plots and the final metrics CSV will be saved in the `reports/` directory.*

## How to Run on Google Colab

You can run this entire experiment in the cloud via Google Colab with zero local setup!

1. Upload the `Battery_Experiment_Colab.ipynb` file to your Google Drive or open it directly in Colab.
2. The notebook is pre-configured to automatically clone this repository, install the dependencies, execute the experiment, and display the beautiful output plots directly in your browser.
3. Simply click **"Run All"** in the Colab interface.

---


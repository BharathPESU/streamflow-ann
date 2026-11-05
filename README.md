# Streamflow Forecasting with Artificial Neural Networks

Daily one-step-ahead streamflow (runoff) prediction for the Leaf River basin (Collins, MS/LA) comparing a **3-Layer Feed-Forward Neural Network (ANN)** and a **Long Short-Term Memory (LSTM) Recurrent Neural Network**.

*Course Mini-Project for **UE24CS352A - Machine Learning**, based on research by Mian Xiao & Shanni You (Stanford University, 2018), Hsu et al. (1995), and Kratzert et al. (2018).*

---

## 1. Project Introduction

### What Does This Project Do?
In hydrology and civil engineering, **streamflow forecasting** predicts the volume of water moving through a river channel over time. Accurate daily forecasts are essential for:
* **Flood warning systems**: Giving downstream communities advance notice before rivers breach their banks.
* **Reservoir and dam management**: Scheduling releases and preserving storage capacity.
* **Water resource allocation**: Ensuring municipal, agricultural, and industrial water security.

This project implements machine learning models that learn the nonlinear relationship between rainfall events and river discharge, forecasting tomorrow's river flow based on recent historical rainfall and runoff.

### Problem Formulation
* **Type of ML Problem**: **Supervised Multivariate Time-Series Regression**.
  * *Supervised*: Trained against historical ground-truth sensor measurements.
  * *Multivariate*: Combines two distinct historical time-series signals (rainfall and river flow).
  * *Time-Series Regression*: The target is a continuous physical quantity measured sequentially in time.

### Inputs and Outputs

```mermaid
flowchart LR
    subgraph Inputs ["Input Features (Past Lags)"]
        R["Rainfall Lags: R(t-1) ... R(t-m)<br/>(inches)"]
        Q["Runoff Lags: Q(t-1) ... Q(t-n)<br/>(ft³/s)"]
    end
    subgraph Models ["Neural Network Models"]
        FF["Feed-Forward ANN<br/>(3-Layer MLP)"]
        LSTM["LSTM Network<br/>(Recurrent Sequence)"]
    end
    subgraph Output ["Target Prediction"]
        Y["Tomorrow's Runoff: Q(t)<br/>(ft³/s)"]
    end
    Inputs --> Models --> Output
```

* **Inputs (Features)**:
  * **Rainfall Lags $R(t-1), \dots, R(t-m)$**: Historical daily precipitation in inches (from NOAA weather station at Collins).
  * **Runoff Lags $Q(t-1), \dots, Q(t-n)$**: Historical daily mean discharge in $\text{ft}^3/\text{s}$ (from USGS river gauge 02472000).
  * *Default lookback*: 3 days rain / 4 days runoff for Feed-Forward; 5 days paired sequence for LSTM.
* **Output (Target)**:
  * **Next-Day Streamflow $Q(t)$**: Predicted river discharge for day $t$ in cubic feet per second ($\text{ft}^3/\text{s}$).

---

## 2. Models: Base Paper vs. Our Implementation

This project builds upon the foundational research of **Xiao & You (Stanford University, 2018)** while expanding the model suite to resolve the severe flood crest underestimation that affected their study.

### High-Level Comparison Summary

| Dimension | Base Paper (Xiao & You, 2018) | Our Implementation |
| :--- | :--- | :--- |
| **Model Suite** | **2 Models** (Feed-Forward ANN, Standard LSTM) | **4 Models** (Persistence Baseline, Feed-Forward ANN, Standard LSTM, Peak-Weighted LSTM) |
| **Baseline Model** | ❌ None (No reference baseline tested) | ✅ **Lag-1 Persistence Model ($\hat{Q}_t = Q_{t-1}$)** |
| **Feed-Forward ANN** | 3-Layer MLP ($d_{\text{in}}=7, h=12, d_{\text{out}}=1$), Sigmoid hidden | 3-Layer MLP ($d_{\text{in}}=7, h=12, d_{\text{out}}=1$), Sigmoid hidden *(Exact replication)* |
| **Standard LSTM** | 5-day sequence ($5 \times 2$), $h=12$, Linear head | 5-day sequence ($5 \times 2$), $h=12$, Linear head *(Exact replication)* |
| **Peak-Weighted LSTM** | ❌ None (Suggested post-hoc manual safety multiplier) | ✅ **Novel Model**: LSTM trained with **Threshold-Weighted Loss ($\mathcal{L}_{PW}$, $3\times$ weight for $Q > 1500$)** |
| **Loss Function** | Standard Unweighted MSE | **Standard MSE** + **Custom Asymmetric Peak-Weighted MSE** |
| **Framework** | TensorFlow 1.x (2018) | PyTorch (Modern, dynamic computation graphs) |
| **Evaluation Metrics** | Only RMSE and RMSE % of Max Flow | **Segregated RMSE**, **Nash-Sutcliffe (NSE)**, and **Kling-Gupta (KGE)** |

---

### A. Models Used in the Base Paper (Xiao & You, 2018)

Xiao & You evaluated two neural network architectures, optimizing lag windows ($m, n \in [2, 7]$) and hidden layer sizes ($h \in [8, 20]$):

1. **3-Layer Standard Feed-Forward Neural Network (MLP)**:
   * **Theoretical Basis**: Follows Hsu et al. (1995) and Minns & Hall (1996).
   * **Input Vector**: 7 lagged features: $\mathbf{x}_t = [R_{t-1}, R_{t-2}, R_{t-3}, Q_{t-1}, Q_{t-2}, Q_{t-3}, Q_{t-4}]$ ($m=3$ rainfall lags, $n=4$ runoff lags).
   * **Hidden Layer**: 12 units with Sigmoid ($\sigma$) non-linear activation.
   * **Output**: 1 linear neuron predicting next-day flow $\hat{Q}(t)$.
   * **Optimization**: Adam optimizer, $lr = 0.009$, 200 epochs, standard unweighted MSE loss.
   * **Reported Finding**: Reported an overall error of 0.23% (an anomalous figure reflecting temporal data leakage).

2. **Long Short-Term Memory (LSTM) Recurrent Network**:
   * **Theoretical Basis**: Follows Kratzert et al. (2018) for rainfall-runoff sequence learning.
   * **Input Tensor**: Chronological 3D sequence requiring equal lags ($m=n$): 5 antecedent daily steps, each containing paired $[R_\tau, Q_\tau]$ features ($\tau \in [t-5, \dots, t-1]$).
   * **Hidden Units**: 12 LSTM cells (with input, forget, and output gating mechanisms).
   * **Output**: Linear dense head mapping final hidden state $h_5 \to \hat{Q}(t)$.
   * **Optimization**: Adam optimizer, $lr = 0.001$, 200 epochs, standard unweighted MSE loss.
   * **Paper's Failure Mode & Conclusion**: The authors found that their LSTM **failed at peak runoff crests** (underestimating peaks with 6.78% error vs 0.63% for FF). They concluded that the LSTM was *"un-robotic and unstable"* and suggested that hydrologists manually multiply predictions by an ad-hoc *"safety factor > 1"*.

---

### B. Models Implemented in This Project

We implemented four models to establish rigorous baselines, replicate the paper's models faithfully, and engineer a machine learning solution to their flood crest underestimation:

1. **Naive Persistence Baseline ($\hat{Q}_t = Q_{t-1}$)**:
   * **What it does**: Predicts tomorrow's river flow will be identical to today's observed flow.
   * **Why we added it**: In daily river dynamics, streamflow has strong autocorrelation ($r > 0.85$). If an ML model cannot beat persistence, it has learned nothing beyond copying yesterday's gauge reading. Adding persistence provides the essential scientific lower bound.
   * **Performance**: $\text{NSE} = 0.769$, Overall Error = 2.91% (865 cfs).

2. **3-Layer Feed-Forward ANN (Exact Architecture Replication)**:
   * **What it does**: Direct PyTorch implementation of the base paper's optimal MLP ($d_{\text{in}}=7, h=12, d_{\text{out}}=1$, Sigmoid activation, $lr=0.009$).
   * **Why we used it**: Evaluates the non-recurrent MLP architecture under **strict, leak-free chronological validation** (2008–2012).
   * **Performance**: $\text{NSE} = 0.843$, $\text{KGE} = \mathbf{0.914}$, Overall Error = 2.40% (713 cfs). It soundly outperforms persistence and achieves exceptional water volume conservation (KGE).

3. **Standard LSTM Network (Exact Architecture Replication)**:
   * **What it does**: Direct PyTorch implementation of the base paper's LSTM sequence model ($(B, 5, 2)$ sequence, 12 hidden units, linear head, $lr=0.001$, standard MSE).
   * **Why we used it**: Re-evaluates sequential recurrent modeling under standard unweighted loss.
   * **Performance**: Overall Error = **2.38%** (708 cfs) and Peak Error = **6.84%** (2,030 cfs), closely replicating the paper's reported numbers (**2.43%** and **6.78%**).

4. **Peak-Weighted LSTM Network (Our Core Innovation)**:
   * **What it does**: Uses the exact same recurrent LSTM architecture as Model 3, but is optimized with a **custom asymmetric, threshold-weighted MSE loss function**:
     $$\mathcal{L}_{PW} = \frac{1}{B} \sum_{i=1}^B w_i (y_i - \hat{y}_i)^2, \quad w_i = \begin{cases} 3.0 & \text{if } y_i > \frac{1500}{Q_{\max}} \\ 1.0 & \text{otherwise} \end{cases}$$
   * **Why we built it**: The base paper gave up on fixing the LSTM's peak underestimation during training, resorting to a post-hoc manual multiplier. We recognized that standard MSE treats all samples equally, causing the optimizer to favor the >85% baseflow samples and flatten extreme flood crests. By penalizing errors during flood stages ($Q > 1500\text{ cfs}$) by $3.0\times$ during backpropagation, the network actively tracks rising limbs and preserves peak flood crests.
   * **Performance**:
     * Reduces peak flood error from 6.84% down to **6.70%** (1,989 cfs), saving nearly $500\text{ cfs}$ in crest error compared to persistence ($8.30\%$).
     * Achieves the highest overall hydrological efficiency: **$\text{NSE} = \mathbf{0.848}$** and lowest overall error (**$2.37\%$**).

---

## 3. Repository & Folder Structure

```
streamflow-ann/
├── data/
│   ├── leaf_river.csv            # Cleaned & merged 10-year dataset (2003–2012)
│   ├── raw_noaa_rain.csv         # Raw NOAA precipitation measurements
│   └── raw_usgs_discharge.csv    # Raw USGS daily mean river flow
├── docs/
│   ├── Guidelines and Instructions_Mini Project Assignment.pdf
│   ├── Xiao_You.pdf              # Stanford research paper (reference study)
│   └── Xiao_You (1).pdf          # Stanford research presentation poster
├── new_data/                     # Unprocessed original agency downloads
│   ├── 4412641.csv               # Raw NOAA download
│   └── daily-data-mean.csv       # Raw USGS download
├── results/                      # Generated model weights, metrics, and plots
│   ├── metrics.json              # Full evaluation results (RMSE, NSE, KGE)
│   ├── model_feedforward.pt      # Saved PyTorch Feed-Forward weights
│   ├── model_lstm.pt             # Saved PyTorch LSTM weights
│   ├── pred_comparison.png       # Side-by-side comparison hydrograph
│   ├── peak_event_zoom.png       # 60-day zoom-in hydrograph of major flood peak
│   ├── pred_feedforward.png      # Feed-Forward validation & scatter plots
│   └── pred_lstm.png             # LSTM validation & scatter plots
├── src/
│   ├── __init__.py
│   ├── data.py                   # Data cleaning, lag windowing, normalization
│   ├── metrics.py                # RMSE, peak error, NSE, and KGE calculations
│   ├── models.py                 # PyTorch architectures (FeedForward, LSTMNet)
│   ├── predict.py                # CLI model testing tool for custom scenarios or dates
│   ├── prepare_data.py           # Data pipeline: merges NOAA rain & USGS flow into CSV
│   ├── sweep.py                  # Hyperparameter grid search & heatmap generator
│   └── train.py                  # Training loop with peak-weighted MSE loss
├── tests/
│   ├── test_data.py              # Tests for window alignment & data leakage
│   └── test_metrics.py           # Unit tests for RMSE, NSE, and KGE
├── app.py                        # Interactive Streamlit Web Dashboard (live demo)
├── main.py                       # Main experiment runner (trains, evaluates, plots, saves)
├── requirements.txt              # Project Python dependencies
└── README.md                     # Project documentation
```

### File Descriptions

| File / Directory | Description |
| :--- | :--- |
| **`app.py`** | **Interactive Streamlit Web Dashboard**. Features What-If flood simulation, 2008–2012 historical date testing, live flood warning badges, and hydrograph views. Ideal for viva/live demos. |
| **`main.py`** | Primary execution script. Trains Feed-Forward and LSTM models, benchmarks against persistence, outputs metrics, generates plots, and saves model checkpoints (`.pt`). |
| **`src/predict.py`** | **CLI Model Testing Tool**. Allows immediate inference with custom rain/runoff values or historical date lookups from the terminal. |
| **`src/sweep.py`** | Hyperparameter tuning script. Sweeps across lag windows ($2\text{ to }7$ days) and hidden layer sizes ($8\text{ to }20$), generating parameter-space heatmaps. |
| **`src/prepare_data.py`** | Preprocessing pipeline. Filters the primary NOAA station at Collins, distributes multi-day precipitation totals, aligns USGS discharge, and writes `data/leaf_river.csv`. |
| **`src/data.py`** | Handles missing-value interpolation (up to 3 days on discharge), lag window sliding, chronologic 2003–2007 train and 2008–2012 validation split, and $[0, 1]$ feature scaling. |
| **`src/models.py`** | PyTorch neural network modules: 3-layer `FeedForward` (Linear $\to$ Sigmoid $\to$ Linear) and recurrent `LSTMNet` with sequence tensor reshaping. |
| **`src/train.py`** | Training engine with Adam optimizer, batching, best-validation-epoch tracking, and customizable **Peak-Weighted Loss**. |
| **`src/metrics.py`** | Evaluates model performance using RMSE, Peak RMSE ($Q > 1500\text{ cfs}$), Non-Peak RMSE, Nash-Sutcliffe Efficiency (NSE), and Kling-Gupta Efficiency (KGE). |
| **`tests/`** | Automated unit test suite verifying window alignments, leakage prevention, and mathematical correctness of metrics. |


---

## 4. Setup and Execution Guide

### Step 1: Prerequisites
* Python 3.10 or higher.
* Recommended: Git and pip.

### Step 2: Clone & Create Virtual Environment
Open your terminal in the project directory:

```bash
# 1. Create a virtual environment named .venv
python3 -m venv .venv

# 2. Activate the virtual environment
# On Linux / macOS:
source .venv/bin/activate
# On Windows (Command Prompt):
# .venv\Scripts\activate.bat
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```
*(Packages installed: `torch`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `pytest`)*

### Step 4: Run the Complete Training Pipeline
To train both models, evaluate them, generate all comparison plots, and save model weights:
```bash
python main.py --data data/leaf_river.csv
```

### Step 5: Train with Peak-Weighted Loss (Flood Enhancement)
To penalize peak flood errors ($Q > 1500\text{ ft}^3/\text{s}$) by 3x:
```bash
python main.py --data data/leaf_river.csv --peak-weight 3.0
```

### Step 6: Launch the Interactive Streamlit Web Dashboard (Live Demo)
To launch the real-time Flood Warning & Scenario Testing UI in your browser:
```bash
streamlit run app.py
```
*(Features: Quick Scenario Presets, What-If Rainfall/Runoff sliders, 2008–2012 Historical Date testing, and live Flood Alert Badges).*

### Step 7: Test Models via Command Line (CLI Inference)
To test predictions without the browser UI:
```bash
# Test on a specific historical flood date:
python src/predict.py --date 2008-03-20

# Test custom What-If scenario (past 3 days rain in inches, past 4 days runoff in cfs):
python src/predict.py --rain 0.5 1.8 0.2 --runoff 300 450 1100 1650
```

### Step 8: Run Hyperparameter Grid Sweeps
To search across lag days and hidden layer sizes:
```bash
# Feed-Forward sweep (lag combinations & hidden sizes)
python src/sweep.py --model ff --data data/leaf_river.csv

# LSTM sweep
python src/sweep.py --model lstm --data data/leaf_river.csv
```


### Step 9: Run Automated Unit Tests
To verify pipeline integrity, windowing, and metric equations:
```bash
python -m pytest tests
```

### Step 10: Quick Pipeline Dry-Run (No Data Needed)
```bash
python main.py --synthetic --epochs 30
```


---

## 5. Evaluation Metrics & Experimental Results

Hydrological models cannot be judged on mean error alone because rivers have extreme non-stationary dynamics. This project evaluates models across five complementary metrics:

### Metric Definitions & Mathematical Formulation

1. **Overall Error (% of Max Flow)**:
   $$\text{RMSE}_{\text{overall}} = \sqrt{\frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2}, \quad \text{Overall Error (\%)} = \frac{\text{RMSE}_{\text{overall}}}{Q_{\max}} \times 100\%$$
   * Evaluates average root-mean-squared deviation across all 1,827 validation days normalized by peak catchment capacity ($Q_{\max} = 29,700\text{ ft}^3/\text{s}$).

2. **Peak Flood Error (% of Max Flow)**:
   $$\text{Peak Error (\%)} = \frac{\text{RMSE}(y_{t \in \text{peaks}}, \hat{y}_{t \in \text{peaks}})}{Q_{\max}} \times 100\%$$
   * Evaluated strictly on hazardous flood days exceeding the flood warning threshold ($Q > 1500\text{ ft}^3/\text{s}$, $n=204$ days). Isolates whether the model suppresses extreme crests during life-critical flood emergencies.

3. **Non-Peak Error (% of Max Flow)**:
   $$\text{Non-Peak Error (\%)} = \frac{\text{RMSE}(y_{t \in \text{non-peaks}}, \hat{y}_{t \in \text{non-peaks}})}{Q_{\max}} \times 100\%$$
   * Evaluated on standard dry-weather and baseflow days ($Q \le 1500\text{ ft}^3/\text{s}$, $n=1623$ days). Ensures the model remains stable and does not hallucinate false flood surges during dry periods.

4. **Nash-Sutcliffe Efficiency (NSE)** *(Hydrology International Gold Standard)*:
   $$\text{NSE} = 1 - \frac{\sum_{i=1}^N (y_i - \hat{y}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
   * Normalizes the mean squared residual error by the natural variance of the observed river discharge.

5. **Kling-Gupta Efficiency (KGE)** *(Multi-Objective Balance Metric)*:
   $$\text{KGE} = 1 - \sqrt{(r - 1)^2 + (\alpha - 1)^2 + (\beta - 1)^2}$$
   * Decomposes prediction quality into three orthogonal physical components:
     * **Linear correlation ($r$)**: Measures hydrograph timing and peak wave arrival phase.
     * **Variability ratio ($\alpha = \sigma_{\text{pred}} / \sigma_{\text{obs}}$)**: Evaluates whether the model captures correct peak-to-trough amplitude without flattening flood crests ($\alpha = 1.0$ is ideal).
     * **Volume bias ratio ($\beta = \mu_{\text{pred}} / \mu_{\text{obs}}$)**: Evaluates cumulative water mass balance over time ($\beta = 1.0$ is zero volume bias).

---

### How to Evaluate and Interpret These Parameters

Use the following benchmark grading scale (established by **Moriasi et al., 2007** and **Knoben et al., 2019**) to evaluate model quality:

| Metric Parameter | Better Direction | Ideal Score | International Hydrologic Grading Bands | What It Measures Physically |
| :--- | :---: | :---: | :--- | :--- |
| **Overall Error (%)** | **Lower $\downarrow$** | `0.0%` | • $<3.0\%$: **Excellent**<br/>• $3.0\%\text{--}5.0\%$: **Good**<br/>• $>5.0\%$: **Suboptimal** | Total average deviation relative to the river's historical maximum flood stage. |
| **Peak Flood Error (%)** | **Lower $\downarrow$** | `0.0%` | • $<7.0\%$: **High Peak Fidelity**<br/>• $7.0\%\text{--}9.0\%$: **Moderate Suppression**<br/>• $>10.0\%$: **Dangerous Crest Collapse** | Crest prediction error during critical flood warning conditions ($Q > 1500\text{ cfs}$). |
| **Non-Peak Error (%)** | **Lower $\downarrow$** | `0.0%` | • $<1.0\%$: **Exceptional Baseflow Stability**<br/>• $1.0\%\text{--}2.0\%$: **Acceptable** | Day-to-day baseflow adherence during dry weather. |
| **NSE (Overall)** | **Higher $\uparrow$** | `1.0` | • **$\text{NSE} \ge 0.80$**: **Very Good (Top Tier)**<br/>• $0.70 \le \text{NSE} < 0.80$: **Good**<br/>• $0.50 \le \text{NSE} < 0.70$: **Satisfactory**<br/>• $\text{NSE} \le 0.0$: **Failing** (worse than historical mean $\bar{y}$) | Hydrograph shape and timing variance explained by the model compared to simple mean climatology. |
| **KGE (Overall)** | **Higher $\uparrow$** | `1.0` | • **$\text{KGE} \ge 0.75$**: **Good to Excellent**<br/>• $0.50 \le \text{KGE} < 0.75$: **Intermediate / Acceptable**<br/>• $\text{KGE} \le -0.41$: **Failing** (worse than mean benchmark) | Balanced multi-criteria fit across flood wave timing ($r$), peak spread ($\alpha$), and total water volume ($\beta$). |

---

### Benchmark Comparison (Validation Period: 2008–2012)

| Model | Overall Error (%) | Peak Flood Error (%) | Non-Peak Error (%) | NSE (Overall) | KGE (Overall) | Primary Strength |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LSTM (Peak-Weighted 3x)** | **2.37%** (705 cfs) | **6.70%** (1,989 cfs) | 0.72% (214 cfs) | **0.848** | 0.862 | **Best peak flood capture & highest overall NSE** |
| **LSTM (Standard)** | 2.38% (708 cfs) | 6.84% (2,030 cfs) | **0.58%** (172 cfs) | 0.846 | 0.868 | Balanced sequential tracking |
| **Feed-Forward ANN** | 2.40% (713 cfs) | 6.82% (2,026 cfs) | 0.62% (183 cfs) | 0.843 | **0.914** | **Best cumulative water volume balance (KGE)** |
| *Persistence Baseline* | *2.91%* (865 cfs) | *8.30%* (2,465 cfs) | *0.70%* (208 cfs) | *0.769* | *0.883* | Naive Lag-1 lower bound ($\hat{Q}_t = Q_{t-1}$) |

---

### How to Compare Models: Operational Decision Guide

When evaluating and selecting between these models for operational deployment, use this trade-off decision framework:

1. **Step 1: Check if the Model Adds Real Value Beyond Persistence ($\text{NSE}_{\text{model}} > \text{NSE}_{\text{persistence}}$)**:
   * River discharge is highly autocorrelated from day to day ($r > 0.85$). Naive persistence achieves $\text{NSE} = 0.769$.
   * Both neural networks achieve **$\text{NSE} > 0.84$** ($\approx 10\%$ relative gain over persistence), proving that they have learned actual rainfall-runoff non-linear transformation rather than merely copying yesterday's river stage.

2. **Step 2: Choose Based on Real-World Operational Objectives**:
   * **For Disaster Management & Flood Warning Systems $\to$ Prioritize Peak Flood Error (%) & NSE**:
     * **Recommended Model: LSTM (Peak-Weighted 3x)**
     * *Why*: In emergency flood warning, underestimating a crest by $2,500\text{ cfs}$ causes fatal inundation. The peak-weighted LSTM delivers the lowest peak error (**6.70%** / 1,989 cfs vs. persistence's 8.30% / 2,465 cfs) and the highest overall hydrograph agreement (**$\text{NSE} = 0.848$**).
   * **For Dam Operations, Reservoir Storage & Water Allocation $\to$ Prioritize KGE (Overall) & Overall Error (%)**:
     * **Recommended Model: Feed-Forward ANN**
     * *Why*: Reservoir managers care about cumulative monthly water inflow volume (mass conservation). The Feed-Forward network achieves the highest Kling-Gupta Efficiency (**$\text{KGE} = 0.914$**), demonstrating near-perfect volume bias ($\beta \approx 1$) and variability matching ($\alpha \approx 1$) with $3\times$ lower compute latency.
   * **For Embedded / Edge IoT Sensor Deployment $\to$ Prioritize Parameter Footprint & Latency**:
     * **Recommended Model: Feed-Forward ANN**
     * *Why*: The 3-layer MLP requires only 7 tabular input features and 12 hidden units, enabling microsecond inference on lightweight edge microcontrollers installed directly at solar-powered river gauging stations.

---

## 6. Visual Analysis & Conceptual Insights

### A. Model Comparison Across 5-Year Validation Series
The figure below compares observed discharge with the Feed-Forward and LSTM predictions over the entire validation timeline (2008–2012):

![Model Comparison Hydrograph](results/pred_comparison.png)

* **Key Takeaway**: Both models track the baseflow and intermediate rainfall-runoff events with high fidelity. When flows are below the 1,500 cfs threshold, the models adhere closely to the observed hydrograph.

---

### B. Storm Event Zoom-In: Peak Flood Dynamics
Hydrological models typically fail at extreme peak crests. The plot below zooms into a 60-day window around the largest recorded flood event in the validation period:

![Storm Event Zoom-In](results/peak_event_zoom.png)

* **Rising Limb**: Both models capture the rapid rise in river discharge following storm rainfall.
* **Crest Underestimation**: Standard unweighted training slightly underestimates the highest peak crest due to the scarcity of extreme events in the training data.
* **Recession Limb**: The models accurately reproduce the logarithmic drainage and return to baseflow conditions.

---

### C. Observed vs. Predicted Scatter Plots
The scatter plots below display observed discharge against model predictions along with the $y = x$ 1:1 perfect prediction line:

| Feed-Forward Network | LSTM Network |
| :---: | :---: |
| ![Feed-Forward Prediction](results/pred_feedforward.png) | ![LSTM Prediction](results/pred_lstm.png) |

* Points clustered tightly around the diagonal indicate high predictive accuracy for low and intermediate flows ($Q < 1500\text{ ft}^3/\text{s}$). The wider spread at high discharge illustrates the greater variance inherent in flood peak prediction.

---

## 7. How to Use the Saved Models for Inference

The trained model weights are saved automatically to `results/`:
* `results/model_feedforward.pt`
* `results/model_lstm.pt`

You can reload the trained weights in Python for instant prediction without retraining:

```python
import torch
from src.models import FeedForward, LSTMNet

# Load Feed-Forward model (7 input features: 3 rain lags + 4 runoff lags, 12 hidden units)
model_ff = FeedForward(n_in=7, hidden=12)
model_ff.load_state_dict(torch.load("results/model_feedforward.pt"))
model_ff.eval()

# Load LSTM model (2 features: rain & runoff, 12 hidden units)
model_lstm = LSTMNet(hidden=12)
model_lstm.load_state_dict(torch.load("results/model_lstm.pt"))
model_lstm.eval()
print("Models successfully loaded and ready for inference!")
```

---

## 8. Limitations & Future Work

1. **Short Lookback Window**: The current models use 3–5 day lag windows. While sufficient for one-step-ahead prediction, real-world river basins retain moisture memory over weeks or months. Expanding the LSTM lookback to 30–180 days with temperature/evapotranspiration features could improve peak accuracy.
2. **One-Step-Ahead Autoregression**: Because yesterday's runoff $Q(t-1)$ is an input, the model relies heavily on recent river state. Multi-step forecasting (e.g., $t+1$ to $t+7$) would test the models' true rainfall routing ability.
3. **Loss Function**: As shown by `--peak-weight`, applying asymmetric penalties to flood peaks improves peak capture and should be explored further with direct differentiable NSE loss functions.

---

## 9. Academic References

1. **Xiao, M., & You, S. (2018).** *Application of Artificial Neural Network in Streamflow Forecasting.* Department of Civil & Environmental Engineering, Stanford University. [docs/Xiao_You.pdf](docs/Xiao_You.pdf).
2. **Hsu, K. L., Gupta, H. V., & Sorooshian, S. (1995).** *Artificial neural network modeling of the rainfall-runoff process.* Water Resources Research, 31(10), 2517-2530.
3. **Kratzert, F., Klotz, D., Brenner, C., Schulz, K., & Herrnegger, M. (2018).** *Rainfall–runoff modelling using Long Short-Term Memory (LSTM) networks.* Hydrology and Earth System Sciences, 22(11), 6005-6022.
4. **Moriasi, D. N., et al. (2007).** *Model evaluation guidelines for systematic quantification of accuracy in watershed simulations.* Transactions of the ASABE, 50(3), 885-900.

---

## 10. Team
* **Chathurth (PES1UG24CS563)**
* **Bharath CD, PES1UG24CS562**
* **Course**: UE24CS352A - Machine Learning Mini-Project
* **Repository**: [https://github.com/BharathPESU/streamflow-ann.git](https://github.com/BharathPESU/streamflow-ann.git)


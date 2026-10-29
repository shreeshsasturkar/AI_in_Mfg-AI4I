# Predictive Maintenance System for Industrial Milling Machines (AI4I Dataset)

An end-to-end, production-grade predictive maintenance and condition-monitoring platform for industrial milling machines built on the AI4I 2020 Predictive Maintenance Dataset.

This project integrates physics-informed machine learning, hierarchical failure diagnosis, real-time telemetry streaming via FastAPI, automated distributional drift monitoring (KS Test & PSI), and local SHAP explainability to generate actionable operator risk alerts.

---

## Key Achievements & Highlights

* **5-Class Failure Prediction System:** Engineered a hierarchical two-stage classification architecture achieving **92% recall** and an **81% F1-score** across failure modes using SMOTE and cost-sensitive class-weighted training.


* **Real-Time Sensor Ingestion Architecture:** Developed an asynchronous microservice using **FastAPI** to ingest high-frequency sensor readings into an in-memory rolling buffer alongside a real-time **Streamlit** dashboard.


* **Distributional Drift Monitoring:** Automated continuous detection of sensor degradation using two-sample **Kolmogorov-Smirnov (KS) tests** and 10-quantile **Population Stability Index (PSI)** against baseline training distributions.


* **Real-Time Explainability & Operator Flags:** Deployed `shap.TreeExplainer` to compute instant local feature attributions per telemetry cycle, converting mathematical Shapley values into human-interpretable operational flags.



---

## System Architecture

```
                                    +-----------------------------------------+
                                    |        Industrial Telemetry Stream      |
                                    |   (Air/Process Temp, RPM, Torque, Wear) |
                                    +-----------------------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |         Async Stream Producer           |
                                    |        (simulator/stream_producer.py)   |
                                    +-----------------------------------------+
                                                         |  HTTP POST /ingest
                                                         v
+-----------------------------------------------------------------------------------------------------------------+
|                                              FastAPI Backend Engine (api/main.py)                               |
|                                                                                                                 |
|  1. Physics-Informed Feature Engineering:                                                                       |
|     - Thermal Delta: ΔT = T_proc - T_air                                                                        |
|     - Mechanical Power: P = Torque × ω = Torque × (2π × RPM / 60)                                               |
|     - Tool Overstrain: Overstrain = Tool wear × Torque                                                          |
|                                                                                                                 |
|  2. Preprocessing & Scaling: Fitted ColumnTransformer (StandardScaler + OneHotEncoder)                          |
|                                                                                                                 |
|  3. Hierarchical Two-Stage Inference:                                                                           |
|     - Stage 1 Gatekeeper: XGBoost binary classifier (scale_pos_weight + tuned threshold)                        |
|                           -> Output: Probability of machine failure                                             |
|     - Stage 2 Diagnoser: XGBoost multi-class classifier (class-weighted, objective='multi:softprob')            |
|                          -> Triggered only on Stage 1 alert                                                     |
|                          -> Output: Diagnosed mode (TWF, HDF, PWF, OSF, RNF)                                    |
|                                                                                                                 |
|  4. Statistical Drift Monitoring:                                                                               |
|     - 2-Sample Kolmogorov-Smirnov Test (p-value thresholding)                                                   |
|     - Population Stability Index (PSI across 10 quantile bins)                                                  |
|                                                                                                                 |
|  5. SHAP Real-Time Attribution:                                                                                 |
|     - TreeExplainer local contributions -> Synthesized Operator Risk Flag                                       |
+-----------------------------------------------------------------------------------------------------------------+
                                        |                                        |
                   GET /state (Polling) |                                        | Streaming Metrics
                                        v                                        v
+-----------------------------------------------------------------------------------------------------------------+
|                                     Streamlit Operations Dashboard (dashboard/app.py)                           |
|                                                                                                                 |
|  * Real-Time Machine Telemetry & Power Gauge Cards                                                              |
|  * Dynamic Failure Probability Trend Line                                                                       |
|  * Live SHAP Local Feature Attribution Horizontal Bar Chart                                                     |
|  * Color-Coded Distributional Drift Table (KS p-values, PSI values, Drift Status)                               |
|  * Real-Time Synthetic Sensor Drift Injection Toggle (Demo Mode)                                                |
+-----------------------------------------------------------------------------------------------------------------+

```

---

## Dataset & Physics-Informed Feature Engineering

The system uses the **AI4I 2020 Predictive Maintenance Dataset** (10,000 operational records reflecting synthetic manufacturing milling machines).

### 1. Underlying Machine Failure Modes

* **Tool Wear Failure (TWF):** Cutting tool breaks or wears down (reaches 200–240 minutes of cumulative wear).


* **Heat Dissipation Failure (HDF):** Machine overheats due to poor heat clearance ($\Delta T < 8.6\text{ K}$ and rotational speed $< 1380\text{ rpm}$).


* **Power Failure (PWF):** Mechanical power generated falls outside acceptable bounds ($< 3500\text{ W}$ or $> 9000\text{ W}$).


* **Overstrain Failure (OSF):** Excessive load product between tool wear and cutting torque.


* **Random Failure (RNF):** Uncorrelated operational anomalies (constant 0.1% background rate).



### 2. Domain Physics Formulations

Raw sensor values are enriched with deterministic physical equations before being passed to model inference:

* **Thermal Differential ($\Delta T$):**

$$\Delta T = T_{\text{process}} - T_{\text{air}}$$





* **Mechanical Spindle Power ($P$):**

$$P = \tau \cdot \omega = \text{Torque} \times \left(\frac{2\pi \cdot \text{Rotational speed}}{60}\right)$$





* **Overstrain Metric:**

$$\text{Overstrain} = \text{Tool wear} \times \text{Torque}$$






---

## Methodological Innovations

### Two-Stage Hierarchical Classification

Because physical failures represent only $\approx 3.4\%$ of normal manufacturing operations, training a single multi-class model directly on the raw data leads to minority-class starvation:

1. **Stage 1 (Binary Gatekeeper):** An XGBoost model trained on all 8,500 training records utilizing class weighting (`scale_pos_weight = 28.51`) and an empirically tuned decision threshold derived from out-of-fold Precision-Recall curves. It maximizes recall to ensure failing machines are rarely missed.
2. **Stage 2 (Root-Cause Diagnoser):** A multi-class XGBoost model (`multi:softprob`) trained strictly on historical failure records using balanced sample weights. It isolates the exact breakdown mode (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and returns a diagnosis confidence score.



### Distributional Drift Monitoring (KS Test & PSI)

The system maintains a rolling window of the last 250 operational cycles and tests them against the pre-computed training distribution:

* **Two-Sample Kolmogorov-Smirnov (KS) Test:** Non-parametric test checking if the empirical cumulative distribution functions of the live stream and reference data diverge significantly:



$$D = \sup_x \vert{}F_{\text{ref}}(x) - F_{\text{stream}}(x)\vert{}$$





* **Population Stability Index (PSI):** Evaluated over 10 reference quantile bins:



$$\text{PSI} = \sum_{i=1}^{10} (A_i - E_i) \times \ln\left(\frac{A_i}{E_i}\right)$$





* $\text{PSI} < 0.10$: **Stable** (Normal Operations)


* $0.10 \le \text{PSI} < 0.25$: **Warning** (Moderate Drift)


* $\text{PSI} \ge 0.25$ or $p < 0.01$: **Critical Drift** (Sensor recalibration needed)





### Real-Time SHAP Attribution

Every sensor reading passed through the inference pipeline is processed by `shap.TreeExplainer`. The local attribution values rank the physical parameters that pushed the machine toward failure, converting mathematical values into a synthesized human-readable alert:

> *"CRITICAL ALERT [HDF]: Immediate inspection required. Primary risk drivers: Temp_Difference (+1.82), Torque (+0.74)."*
> 
> 

---

## Repository Structure

```text
predictive-maintenance/
├── ai4i2020.csv                                 # Raw AI4I 2020 dataset
├── two_stage_predictive_maintenance_bundle.pkl  # Serialized models, encoders, and threshold
├── reference_baseline.pkl                       # Training feature distribution for drift detection
├── test_stream.csv                              # Holdout test stream partition
├── export_reference.py                          # Setup script to generate baselines
├── requirements.txt                             # Python dependencies
├── src/
│   ├── drift.py                                 # KS-test & 10-bin PSI calculation engine
│   └── explain.py                               # Real-time SHAP explainer & operator risk generator
├── api/
│   └── main.py                                  # FastAPI ingestion & inference service
├── simulator/
│   └── stream_producer.py                       # Real-time PLC sensor stream generator
└── dashboard/
    └── app.py                                   # Streamlit real-time monitoring dashboard

```

---

## Installation & Setup

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/your-username/predictive-maintenance-ai4i.git
cd predictive-maintenance-ai4i

# Create and activate virtual environment
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Generate Reference Baselines

Run the setup script to construct the reference baseline and stream data partitions:

```bash
python export_reference.py

```

---

## Running the Application

To run the complete streaming and monitoring architecture, open **three independent terminal tabs** in your root directory:

### Terminal 1: FastAPI Ingestion Service

```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

```

* Interactive API documentation is available at `[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)`.

### Terminal 2: Streamlit Condition Monitoring Dashboard

```bash
streamlit run dashboard/app.py

```

* Access the operator interface in your browser at `http://localhost:8501`.

### Terminal 3: Real-Time PLC Sensor Simulator

```bash
python simulator/stream_producer.py

```

* Streams records sequentially over HTTP POST to `/ingest`.

---

## API Endpoints Reference

| Endpoint | Method | Description |
| --- | --- | --- |
| `/ingest` | `POST` | Accepts a single sensor cycle payload, computes physics features, evaluates Stage 1 and Stage 2 models, generates SHAP values, and appends to the rolling buffer. |
| `/state` | `GET` | Returns the latest ingested record, telemetry history, buffer size, and calculated KS/PSI drift reports for the dashboard. |
| `/toggle_drift` | `POST` | Injects synthetic sensor bias into incoming readings to verify drift monitoring alarms in real time. |

---

## Live Demonstration Guide

1. Launch all three services as described above.
2. Watch the **Streamlit Dashboard** update live as the simulator sends machine cycles.
3. Observe normal cycles displaying `NOMINAL` status with green risk banners.
4. When an anomalous cycle occurs (e.g., UDI 4226), note:
* The red **Operator Risk Flag** detailing root cause and driver features.


* The **SHAP attribution bar chart** displaying the exact feature contributions.


* The **Diagnosed Mode** metric showing the specific breakdown category.




5. In the sidebar, click **"Toggle Synthetic Sensor Drift"**:
* The system injects a systematic offset onto torque and thermal sensors.
* Watch the bottom **Distributional Drift Monitoring** table transition from green `STABLE` to red `CRITICAL DRIFT` as PSI exceeds $0.25$ and the KS test $p$-value drops below $0.01$.

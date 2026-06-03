# Activity Classifier

Human activity recognition using wearable sensor data from the [WISDM dataset](https://www.cis.fordham.edu/wisdm/dataset.php). Classifies 18 activities (jogging, walking, eating, typing, etc.) from watch and phone accelerometer and gyroscope readings using a PySpark-based stacked ensemble model.

---

## Dataset

The WISDM dataset contains accelerometer and gyroscope readings from 51 subjects performing 18 activities simultaneously wearing a smartwatch and carrying a smartphone. Raw data lives in:

```
data/wisdm-dataset/raw/
    watch/accel/    # watch accelerometer (.txt)
    watch/gyro/     # watch gyroscope (.txt)
    phone/accel/    # phone accelerometer (.txt)
    phone/gyro/     # phone gyroscope (.txt)
```

**Activities:** Walking, Jogging, Stairs, Sitting, Standing, Typing, Teeth brushing, Soup, Chips, Pasta, Drinking, Sandwich, Kicking, Catch, Dribbling, Writing, Clapping, Folding

---

## Project Structure

```
activity_classifier/
├── data/                            # raw and processed data
├── rf_pipeline/                     # Random Forest pipeline (automated)
│   ├── feature_engineering.py       # data loading, cleaning, joining, features
│   ├── model.py                     # windowing, stacked RF model, evaluation
│   └── eda.py                       # EDA plots saved to plots/
├── lstm_pipeline/                   # CNN-LSTM pipeline (run on Google Colab)
│   ├── eda.ipynb
│   ├── datawrangling.ipynb
│   ├── load_data.ipynb
│   └── cnn_lstm_classifier.ipynb
├── plots/                           # generated EDA plots
├── main.py                          # runs rf_pipeline sequentially
├── requirements.txt
├── launch-arm64.sh                  # Docker launch script
└── README.md
```

---

## Pipelines

This project has two independent pipelines starting from the same raw data:

| | Random Forest Pipeline | CNN-LSTM Pipeline |
|--|--|--|
| Location | `rf_pipeline/` | `lstm_pipeline/` |
| How to run | `python main.py` | Notebooks on Google Colab |
| Model | Stacked RF + Decision Tree | CNN-LSTM |
| Hardware | Local CPU | Colab GPU/TPU |

---

## Random Forest Pipeline

### 1. Feature Engineering (`rf_pipeline/feature_engineering.py`)

- Loads 4 raw CSV streams (watch accel, watch gyro, phone accel, phone gyro)
- Cleans the `z` column (trailing semicolons and whitespace)
- Rounds timestamps to nearest 50 nanoseconds to align sensors for joining
- Joins watch accel + watch gyro on `(subject_id, timestamp)` → one `watch` DataFrame
- Same for phone → one `phone` DataFrame
- Engineers the following features per reading:
  - `seq_id` — unique identifier per person-activity-sequence
  - `dx/dy/dz` — displacement from previous reading
  - `ax/ay/az` — acceleration (displacement ÷ delta time between readings)
  - `magnitude` — orientation-independent total movement `sqrt(x²+y²+z²)`
  - `xy/xz/yz` — cross-axis correlations
  - `jerk` — magnitude of the acceleration vector
- Writes output to parquet:
  - `data/watch_features.parquet`
  - `data/phone_features.parquet`

### 2. Modelling (`rf_pipeline/model.py`)

**Windowing**
- Groups readings by subject and activity label, sorted by timestamp
- Applies sliding windows of 200 readings (~10 seconds at 20Hz) with 50% overlap
- Summarises each window using mean and standard deviation of all 28 features → 56 values per window

**Train/Val/Test Split**
- Split by subject ID (60/20/20) — no subject appears in more than one split, preventing subject leakage

**Stacked Ensemble**
- Watch Random Forest (100 trees) trained on windowed watch features
- Phone Random Forest (100 trees) trained on windowed phone features
- Validation set probability vectors from both models joined on `(subject_id, activity_label, window_index)`
- Decision Tree meta-classifier trained on concatenated probability vectors

### 3. EDA (`rf_pipeline/eda.py`)

- Dataset overview, row counts, subject distribution
- Activity distribution per source
- Descriptive statistics for raw sensor columns
- Boxplots of sensor values by activity
- Magnitude distribution by activity
- Feature correlation heatmaps
- Null/NaN check

---

## CNN-LSTM Pipeline

Run the notebooks in order on **Google Colab** (GPU/TPU recommended):

1. `lstm_pipeline/eda.ipynb` — initial data exploration and loading
2. `lstm_pipeline/datawrangling.ipynb` — creates `.parquet` files used by the classifier
3. `lstm_pipeline/cnn_lstm_classifier.ipynb` — trains the CNN-LSTM model

> Update the `base` file path in `cnn_lstm_classifier.ipynb` to point to your parquet files, or place them in the same directory.

---

## Results

| Model | Accuracy |
|-------|----------|
| Watch Random Forest (val) | 61.5% |
| Phone Random Forest (val) | 25.9% |
| Stacked model (test) | 43.4% |

**Best classified activities:** Jogging (98.5%), Stairs (89.7%), Drinking (70.1%)  
**Most challenging:** Soup, Chips, Sandwich eating (0%) — low-amplitude signals with high inter-subject variability

---

## Getting the Data

The raw dataset is hosted on a public S3 bucket. Run the download script to fetch it:

```bash
pip install boto3
python download_data.py
```

This downloads the raw sensor files into `data/wisdm-dataset/raw/` automatically.

---

## Requirements

```bash
pip install -r requirements.txt
```

| Package | Version |
|---------|---------|
| pyspark | 4.1.1 |
| pandas | 3.0.3 |
| pyarrow | 24.0.0 |
| numpy | 2.4.6 |
| matplotlib | 3.10.9 |
| seaborn | 0.13.2 |

---

## Running

Use `main.py` to run the full pipeline sequentially:

```bash
# Full run (first time — ~40 min total)
python main.py

# Skip feature engineering if parquet files already exist (~30 min)
python main.py --skip-features

# Run everything including EDA
python main.py --skip-features --eda
```

Or run individual steps manually:

```bash
python rf_pipeline/feature_engineering.py   # ~10 min — reads raw CSVs, writes parquet
python rf_pipeline/model.py                  # ~30 min — trains stacked model, prints results
python rf_pipeline/eda.py                    # generates plots to plots/
```

---

## Getting Started with Docker

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### Launch the Container

```bash
bash "launch-arm64.sh" -d /path/to/your/project
```

**Optional flags:**
- `-t` / `--tag` — specify an image version (default: `latest`), e.g. `--tag 2026.1`
- `-d` / `--directory` — base directory to mount

**Example:**
```bash
bash "launch-arm64.sh" --tag 2026.1 --directory ~/activity_classifier
```

### Edit Code in VS Code
Your local directory is mounted into the container at `/home/work`, so any edits you make in VS Code are reflected inside the container in real time.

### Stop the Container
Press `Ctrl + C` in the terminal where the script is running.

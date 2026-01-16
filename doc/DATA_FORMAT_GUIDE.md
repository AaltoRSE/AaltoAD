# Data Format Guide

## Expected Output Format from preprocess.py

All datasets are processed into a standardized format with three numpy arrays:

### Standard Files Generated
For each dataset `{dataset_name}`, preprocess.py creates:

1. **train.npy** - Training time series data
   - Shape: `(num_train_samples, num_features)`
   - Type: `float64`
   - Values: Normalized to range [0, 1] (or approximately)

2. **test.npy** - Test time series data  
   - Shape: `(num_test_samples, num_features)`
   - Type: `float64`
   - Values: Normalized using same scaling as training data

3. **labels.npy** - Ground truth anomaly labels
   - Shape: `(num_test_samples, num_features)` - **Same shape as test.npy**
   - Type: `float64`
   - Values: Binary (0 or 1) indicating anomalies
   - **Critical**: Must have same number of samples and features as test.npy

### Directory Structure
```
output_folder/
  {dataset_name}/
    train.npy
    test.npy
    labels.npy
```

## Key Constraints for load_dataset Compatibility

The `load_dataset()` function in main.py expects:
1. Files named exactly: `train.npy`, `test.npy`, `labels.npy`
2. All arrays as `float64` type
3. test.npy and labels.npy to have **identical shapes**
4. Data normalized to approximately [0, 1] range
5. Files located in: `output_folder/{dataset_name}/`

## Normalization Methods Available

- **normalize()** - Symmetric scaling around 0.5, range ~[0.25, 0.75]
- **normalize2()** - Global min-max, range [0, 1], returns (normalized, min, max)
- **normalize3()** - Per-feature min-max, range [0, 1], preserves per-feature scaling
- **convertNumpy()** - Downsample by 10x and normalize (for large datasets)

---

## Example: NAB Dataset

### Raw Data Format
NAB (Numenta Anomaly Benchmark) consists of univariate time series CSV files with two columns:
```
timestamp,value
2014-03-07 03:41:00,45.868
2014-03-07 03:46:00,47.606
2014-03-07 03:51:00,42.58
```

Example files in `data/NAB/`:
- `ec2_request_latency_system_failure.csv` - EC2 request latency measurements
- `cpu_utilization_asg_misconfiguration.csv` - CPU utilization percentage
- `machine_temperature_system_failure.csv` - Temperature sensor readings
- `ambient_temperature_system_failure.csv` - Ambient temperature
- `nyc_taxi.csv` - NYC taxi passenger count
- `rogue_agent_key_hold.csv` - Key hold duration
- `rogue_agent_key_updown.csv` - Key up/down timing

### Processing Pipeline (from preprocess.py)
```python
# 1. Load CSV
df = pd.read_csv('data/NAB/ec2_request_latency_system_failure.csv')

# 2. Extract the value column (column 1)
vals = df.values[:, 1]  # Shape: (4034,)

# 3. Normalize to [0, 1]
min_temp, max_temp = np.min(vals), np.max(vals)
vals = (vals - min_temp) / (max_temp - min_temp)

# 4. Use same data for train and test (not split)
train = vals.astype(float)  # Shape: (4034,)
test = vals.astype(float)   # Shape: (4034,)

# 5. Create labels from labels.json
labels = np.zeros_like(vals)  # Mark anomaly indices
labels[anomaly_indices] = 1

# 6. Reshape to 2D for consistency
train, test, labels = train.reshape(-1, 1), test.reshape(-1, 1), labels.reshape(-1, 1)
# Final shapes: (4034, 1) each
```

### Processed Data Format
After preprocessing, each NAB file produces:
- **Number of Features**: **1** (univariate time series)
- **Number of Samples**: Varies by file (~4000 samples)
- **Feature Representation**: 
  - Single numeric value representing the metric
  - Values normalized to [0, 1] range
  - What it measures depends on the file:
    - **ec2_request_latency**: System request latency in milliseconds
    - **cpu_utilization**: CPU usage as percentage (0-100%, normalized to 0-1)
    - **machine_temperature**: Temperature sensor reading
    - **nyc_taxi**: Passenger count in taxi pickups
    - **key_hold/updown**: Keyboard event timing in milliseconds

### Example Output Files
```
output_folder/NAB/
  ec2_request_latency_system_failure_train.npy  # Shape: (4034, 1), dtype: float64
  ec2_request_latency_system_failure_test.npy   # Shape: (4034, 1), dtype: float64
  ec2_request_latency_system_failure_labels.npy # Shape: (4034, 1), dtype: float64, binary (0/1)
  cpu_utilization_asg_misconfiguration_train.npy
  ... (one set for each CSV file)
```

### Key Characteristics
- **Univariate**: Each file contains only 1 feature (the value column)
- **Time Series**: Samples ordered chronologically (5-minute intervals)
- **Ground Truth Labels**: Marked in `labels.json` with timestamp ranges of known anomalies
- **No Feature Engineering**: Raw numeric values extracted directly from CSV
- **Simple Normalization**: Global min-max scaling (not per-feature, since only 1 feature)

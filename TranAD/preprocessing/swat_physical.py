import os
import numpy as np
import pandas as pd
from TranAD.preprocessing.utils import normalize
from TranAD.constants import DEFAULT_DATA_FOLDER



def load_SWaT_payload_netflow(folder, data_folder=DEFAULT_DATA_FOLDER, test_size=0):
    dataset_folder = os.path.join(data_folder, 'SWaT_payload_netflow')
    baseline_file = os.path.join(dataset_folder, 'NetflowPayload-Baseline-10s.csv')
    attack_file = os.path.join(dataset_folder, 'NetflowPayload-Attack-10s.csv')

    df_baseline = pd.read_csv(baseline_file)
    df_attack = pd.read_csv(attack_file)

    n_test_baseline = int(test_size*len(df_baseline))
    df_train_len = int(len(df_baseline)) - n_test_baseline
    df_train = df_baseline[:df_train_len]
    df_test = pd.concat([df_baseline[df_train_len:], df_attack], ignore_index=True)
    test_timestamps = df_test['Timestamp']

    labels = df_test['Tag']
    labels = pd.to_numeric(labels, errors="coerce").ne(0).astype(int).to_numpy()
    df_train = df_train.drop(columns=['Normal/Attack', 'Timestamp', 'tag', 'Tag'], errors='ignore')
    df_test = df_test.drop(columns=['Normal/Attack', 'Timestamp', 'tag', 'Tag'], errors='ignore')

    for col in df_train.columns:
        train_col, min_a, max_a = normalize(df_train[col])
        test_col = normalize(df_test[col], min_a, max_a)[0]
        df_train[col] = train_col
        df_test[col] = test_col

    train = df_train.values
    test = df_test.values
    labels = np.repeat(labels.reshape(-1, 1), test.shape[1], axis=1)
    for file in ['train', 'test', 'labels']:
        np.save(os.path.join(folder, f'{file}.npy'), eval(file))

    pd.DataFrame(test_timestamps).to_csv(os.path.join(folder, 'timestamps.csv'), index=False, header=False)


def load_SWaT_netflow(folder, data_folder=DEFAULT_DATA_FOLDER, test_size=0):
    dataset_folder = os.path.join(data_folder, 'SWaT_netflow')
    baseline_file = os.path.join(dataset_folder, 'SWat2015-JUST-NETFLOW-Baseline-10s.csv')
    attack_file = os.path.join(dataset_folder, 'SWat2015-JUST-NETFLOW-Attack-10s.csv')

    df_baseline = pd.read_csv(baseline_file)
    df_attack = pd.read_csv(attack_file)

    n_test_baseline = int(test_size*len(df_baseline))
    df_train_len = int(len(df_baseline)) - n_test_baseline
    df_train = df_baseline[:df_train_len]
    df_test = pd.concat([df_baseline[df_train_len:], df_attack], ignore_index=True)
    test_timestamps = df_test['Timestamp']

    labels = df_test['Tag']
    labels = pd.to_numeric(labels, errors="coerce").ne(0).astype(int).to_numpy()
    df_train = df_train.drop(columns=['Normal/Attack', 'Timestamp', 'tag', 'Tag'], errors='ignore')
    df_test = df_test.drop(columns=['Normal/Attack', 'Timestamp', 'tag', 'Tag'], errors='ignore')

    for col in df_train.columns:
        train_col, min_a, max_a = normalize(df_train[col])
        test_col = normalize(df_test[col], min_a, max_a)[0]
        df_train[col] = train_col
        df_test[col] = test_col

    train = df_train.values
    test = df_test.values
    labels = np.repeat(labels.reshape(-1, 1), test.shape[1], axis=1)
    for file in ['train', 'test', 'labels']:
        np.save(os.path.join(folder, f'{file}.npy'), eval(file))

    pd.DataFrame(test_timestamps).to_csv(os.path.join(folder, 'timestamps.csv'), index=False, header=False)


def load_SWaT_physical(folder, data_folder=DEFAULT_DATA_FOLDER, test_size=0):
    dataset_folder = os.path.join(data_folder, 'SWaT_physical')
    baseline_file = os.path.join(dataset_folder, 'SWaT-PHYSICAL-BASELINE2015.csv')
    attack_file = os.path.join(dataset_folder, 'SWaT-PHYSICAL-ATTACK2015.csv')

    df_baseline = pd.read_csv(baseline_file)
    df_attack = pd.read_csv(attack_file)

    n_test_baseline = int(test_size*len(df_baseline))
    df_train_len = int(len(df_baseline)) - n_test_baseline
    df_train = df_baseline[:df_train_len]
    df_test = pd.concat([df_baseline[df_train_len:], df_attack], ignore_index=True)

    labels = df_test['Normal/Attack']
    labels = labels.astype(str).str.strip().str.lower().ne("normal").astype(int).to_numpy()
    test_timestamps = df_test['Timestamp']
    df_train = df_train.drop(columns=['Normal/Attack', 'Timestamp', 'tag', 'Tag'], errors='ignore')
    df_test = df_test.drop(columns=['Normal/Attack', 'Timestamp', 'tag', 'Tag'], errors='ignore')

    for col in df_train.columns:
        train_col, min_a, max_a = normalize(df_train[col])
        test_col = normalize(df_test[col], min_a, max_a)[0]
        df_train[col] = train_col
        df_test[col] = test_col

    # The physical datasethas NaN rows. These are anomalous, of course. Replace with a placeholder after normalization
    df_train = df_train.fillna(0)
    df_test = df_test.fillna(0)

    train = df_train.values
    test = df_test.values
    labels = np.repeat(labels.reshape(-1, 1), test.shape[1], axis=1)
    for file in ['train', 'test', 'labels']:
        np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    
    pd.DataFrame(test_timestamps).to_csv(os.path.join(folder, 'timestamps.csv'), index=False, header=False)


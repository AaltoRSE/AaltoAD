import os
import numpy as np
import pandas as pd
from TranAD.preprocessing.utils import normalize2
from TranAD.constants import DEFAULT_DATA_FOLDER



def load_SWaT_payload_netflow(folder, data_folder=DEFAULT_DATA_FOLDER):
    dataset_folder = os.path.join(data_folder, 'SWaT_payload_netflow')
    baseline_file = os.path.join(dataset_folder, 'NetflowPayload-Baseline-10s.csv')
    attack_file = os.path.join(dataset_folder, 'NetflowPayload-Attack-10s.csv')

    df_baseline = pd.read_csv(baseline_file)
    df_attack = pd.read_csv(attack_file)

    n_test_baseline = min(60, int(0.2*len(df_baseline)))
    df_train_len = int(len(df_baseline)) - n_test_baseline
    df_train = df_baseline[:df_train_len]
    df_test = pd.concat([df_baseline[df_train_len:], df_attack], ignore_index=True)
    test_timestamps = df_test['Timestamp']

    df_train = df_train.drop(columns=['Timestamp'])
    df_test = df_test.drop(columns=['Timestamp'])

    labels = np.concatenate([np.zeros(len(df_baseline) - df_train_len), np.ones(len(df_attack))])
    train = df_train.values
    test = df_test.values
    for file in ['train', 'test', 'labels']:
        np.save(os.path.join(folder, f'{file}.npy'), eval(file))

    pd.DataFrame(test_timestamps).to_csv(os.path.join(folder, 'timestamps.csv'), index=False, header=False)


def load_SWaT_netflow(folder, data_folder=DEFAULT_DATA_FOLDER):
    dataset_folder = os.path.join(data_folder, 'SWaT_netflow')
    baseline_file = os.path.join(dataset_folder, 'SWat2015-JUST-NETFLOW-Baseline-10s.csv')
    attack_file = os.path.join(dataset_folder, 'SWat2015-JUST-NETFLOW-Attack-10s.csv')

    df_baseline = pd.read_csv(baseline_file)
    df_attack = pd.read_csv(attack_file)

    n_test_baseline = min(60, int(0.2*len(df_baseline)))
    df_train_len = int(len(df_baseline)) - n_test_baseline
    df_train = df_baseline[:df_train_len]
    df_test = pd.concat([df_baseline[df_train_len:], df_attack], ignore_index=True)
    test_timestamps = df_test['Timestamp']

    df_train = df_train.drop(columns=['Timestamp'])
    df_test = df_test.drop(columns=['Timestamp'])

    for col in df_train.columns:
        train_col, min_a, max_a = normalize2(df_train[col])
        test_col = normalize2(df_test[col], min_a, max_a)[0]
        df_train[col] = train_col
        df_test[col] = test_col

    labels = np.concatenate([np.zeros(len(df_baseline) - df_train_len), np.ones(len(df_attack))])
    train = df_train.values
    test = df_test.values
    for file in ['train', 'test', 'labels']:
        np.save(os.path.join(folder, f'{file}.npy'), eval(file))

    pd.DataFrame(test_timestamps).to_csv(os.path.join(folder, 'timestamps.csv'), index=False, header=False)


def load_SWaT_physical(folder, data_folder=DEFAULT_DATA_FOLDER):
    dataset_folder = os.path.join(data_folder, 'SWaT_physical')
    baseline_file = os.path.join(dataset_folder, 'SWaT-PHYSICAL-BASELINE2015.csv')
    attack_file = os.path.join(dataset_folder, 'SWaT-PHYSICAL-ATTACK2015.csv')

    df_baseline = pd.read_csv(baseline_file)
    df_attack = pd.read_csv(attack_file)

    n_test_baseline = min(60, int(0.2*len(df_baseline)))
    df_train_len = int(len(df_baseline)) - n_test_baseline
    df_train = df_baseline[:df_train_len]
    df_test = pd.concat([df_baseline[df_train_len:], df_attack], ignore_index=True)

    labels = df_test['Normal/Attack'].apply(lambda x: 0 if x == 'Normal' else 1).values
    test_timestamps = df_test['Timestamp']
    df_train = df_train.drop(columns=['Normal/Attack', 'Timestamp'])
    df_test = df_test.drop(columns=['Normal/Attack', 'Timestamp'])

    for col in df_train.columns:
        train_col, min_a, max_a = normalize2(df_train[col])
        test_col = normalize2(df_test[col], min_a, max_a)[0]
        df_train[col] = train_col
        df_test[col] = test_col

    train = df_train.values
    test = df_test.values
    for file in ['train', 'test', 'labels']:
        np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    
    pd.DataFrame(test_timestamps).to_csv(os.path.join(folder, 'timestamps.csv'), index=False, header=False)


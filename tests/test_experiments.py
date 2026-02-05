from TranAD import run_experiment
from TranAD import utils
from TranAD import constants
import torch
import pytest
import os
import csv
import json
from unittest.mock import Mock, patch, MagicMock
import numpy as np


def test_save_and_load_model(tmp_path):
    """ Test saving and loading a model works correctly.
    """
    dataset_name = 'synthetic'
    model_name = 'LSTM_Univariate'
    model, optimizer, scheduler, epoch, accuracy_list, applied_hyperparams = run_experiment.load_model(
        model_name, 1, dataset_name, model_name, retrain=True, test=False, root_path=str(tmp_path)+'/'
    )
    # Save the model
    run_experiment.save_model(model, optimizer, scheduler, epoch, accuracy_list, model_name, dataset_name, root_path=str(tmp_path)+'/')
    
    print(f"Files in tmp_path: {list(tmp_path.rglob('*'))}")

    # Load the model again
    loaded_model, loaded_optimizer, loaded_scheduler, loaded_epoch, loaded_accuracy_list, _ = run_experiment.load_model(
        model_name, 1, dataset_name, model_name, retrain=False, test=True, root_path=str(tmp_path)+'/'
    )
    
    # Check that the loaded model has the same state_dict as the original
    for param_tensor in model.state_dict():
        assert torch.equal(model.state_dict()[param_tensor], loaded_model.state_dict()[param_tensor])
    
    # Check that other components are also correctly loaded
    assert epoch == loaded_epoch
    assert accuracy_list == loaded_accuracy_list


def test_load_model_retrain_flag(tmp_path):
    """Test that retrain flag prevents loading existing checkpoint."""
    dataset_name = 'synthetic'
    model_name = 'LSTM_Univariate'
    
    # Create and save a model
    model, optimizer, scheduler, epoch, accuracy_list, _ = run_experiment.load_model(
        model_name, 1, dataset_name, retrain=True, test=False, root_path=str(tmp_path)+'/'
    )
    run_experiment.save_model(model, optimizer, scheduler, 5, [(0.1, 0.001)], model_name, dataset_name, root_path=str(tmp_path)+'/')
    
    # Load with retrain=True should create new model (epoch=-1)
    model_retrain, _, _, epoch_retrain, acc_retrain, _ = run_experiment.load_model(
        model_name, 1, dataset_name, retrain=True, test=False, root_path=str(tmp_path)+'/'
    )
    assert epoch_retrain == -1
    assert acc_retrain == []


def test_load_model_with_hyperparameters():
    """Test that hyperparameters are applied when loading model."""
    dataset_name = 'synthetic'
    model_name = 'TranAD'
    
    # Create hyperparams JSON string
    hyperparams = json.dumps({
        'lr': 0.005,
        'batch': 64
    })
    
    model, optimizer, scheduler, epoch, accuracy_list, applied_hyperparams = run_experiment.load_model(
        model_name, 5, dataset_name, retrain=True, test=False, hyperparams_str=hyperparams
    )
    
    # Verify hyperparameters were applied
    assert 'lr' in applied_hyperparams
    assert 'batch' in applied_hyperparams
    assert model.lr == 0.005
    assert model.batch == 64


def test_append_benchmark_row_creates_file(tmp_path):
    """Test that append_benchmark_row creates CSV file with header."""
    bench_path = tmp_path / 'benchmarks.csv'
    
    result_dict = {
        'precision': 0.85,
        'recall': 0.90,
        'ROC/AUC': 0.95,
        'f1': 0.87
    }
    
    run_experiment.append_benchmark_row('TranAD', 'TOL', result_dict, bench_path=str(bench_path))
    
    # Verify file exists
    assert bench_path.exists()
    
    # Verify contents
    with open(bench_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 2  # header + 1 data row
        assert rows[0] == ['model', 'dataset', 'precision', 'recall', 'AUC', 'f1']
        assert rows[1] == ['TranAD', 'TOL', '0.85', '0.9', '0.95', '0.87']


def test_append_benchmark_row_appends_to_existing(tmp_path):
    """Test that append_benchmark_row appends to existing file without duplicating header."""
    bench_path = tmp_path / 'benchmarks.csv'
    
    # Create first entry
    result1 = {'precision': 0.85, 'recall': 0.90, 'ROC/AUC': 0.95, 'f1': 0.87}
    run_experiment.append_benchmark_row('TranAD', 'TOL', result1, bench_path=str(bench_path))
    
    # Append second entry
    result2 = {'precision': 0.88, 'recall': 0.92, 'ROC/AUC': 0.96, 'f1': 0.90}
    run_experiment.append_benchmark_row('USAD', 'MSL', result2, bench_path=str(bench_path))
    
    # Verify contents
    with open(bench_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 3  # header + 2 data rows
        assert rows[0] == ['model', 'dataset', 'precision', 'recall', 'AUC', 'f1']
        assert rows[1][0] == 'TranAD'
        assert rows[2][0] == 'USAD'


def test_append_benchmark_row_handles_missing_values(tmp_path):
    """Test that append_benchmark_row handles missing values gracefully."""
    bench_path = tmp_path / 'benchmarks.csv'
    
    result_dict = {
        'precision': 0.85,
        # missing recall, ROC/AUC, f1
    }
    
    run_experiment.append_benchmark_row('TranAD', 'TOL', result_dict, bench_path=str(bench_path))
    
    # Verify file exists and handles None values
    assert bench_path.exists()
    with open(bench_path, 'r') as f:
        reader = csv.reader(f)
        rows = list(reader)
        assert len(rows) == 2
        assert rows[1][2] == '0.85'
        assert rows[1][3] == ''  # None becomes empty string in CSV


@patch('TranAD.run_experiment.backprop')
@patch('TranAD.run_experiment.load_model')
@patch('TranAD.utils.load_dataset')
@patch('TranAD.utils.convert_to_windows')
@patch('TranAD.plotting.plotter')
@patch('TranAD.pot.pot_eval')
@patch('TranAD.diagnosis.hit_att')
@patch('TranAD.diagnosis.ndcg')
@patch('TranAD.utils.get_git_hash')
def test_run_experiment_basic(mock_git_hash, mock_ndcg, mock_hit_att, mock_pot_eval, 
                               mock_plotter, mock_convert_to_windows, mock_load_dataset, mock_load_model, mock_backprop, tmp_path, monkeypatch):
    """Test run_experiment function with mocked models."""
    monkeypatch.chdir(tmp_path)
    
    # Setup mocks
    mock_git_hash.return_value = 'abc123'
    
    # Create mock model
    mock_model = Mock()
    mock_model.name = 'TranAD'
    mock_model.lr = 0.001
    mock_model.batch = 128
    mock_model.n_window = 10
    mock_optimizer = Mock()
    mock_scheduler = Mock()
    
    mock_load_model.return_value = (mock_model, mock_optimizer, mock_scheduler, -1, [], {})
    
    # Create mock data
    train_data = torch.randn(100, 5).numpy()
    test_data = torch.randn(50, 5).numpy()
    labels = np.random.randint(0, 2, (50, 5))
    
    train_loader = [train_data]
    test_loader = [test_data]
    mock_load_dataset.return_value = (train_loader, test_loader, labels)
    
    # Mock convert_to_windows to return data as-is (simplified)
    mock_convert_to_windows.side_effect = lambda data, *args: torch.from_numpy(data) if isinstance(data, np.ndarray) else data
    
    # Mock backprop to return loss and predictions
    # Since test=True, no training calls, only testing and train loss
    mock_backprop.side_effect = [
        # Testing call (training=False)
        (np.random.rand(50, 5), np.random.rand(50, 5)),
        # Train loss for POT (training=False)
        (np.random.rand(100, 5), np.random.rand(100, 5))
    ]
    
    # Mock pot_eval to return result and predictions
    mock_pot_result = {'precision': 0.85, 'recall': 0.90, 'ROC/AUC': 0.95, 'f1': 0.87}
    mock_pot_eval.return_value = (mock_pot_result, np.random.rand(50))
    
    mock_hit_att.return_value = {'hit_rate': 0.80}
    mock_ndcg.return_value = {'ndcg': 0.75}
    
    # Run experiment in test mode (no training)
    result = run_experiment.run_experiment('TranAD', 'TOL', test=True)
    
    # Verify results
    assert result['precision'] == 0.85
    assert result['recall'] == 0.90
    assert result['ROC/AUC'] == 0.95
    assert result['f1'] == 0.87
    assert result['hit_rate'] == 0.80
    assert result['ndcg'] == 0.75
    assert result['git_hash'] == 'abc123'
    assert result['model'] == 'TranAD'
    assert result['dataset'] == 'TOL'
    
    # Verify results file was created
    results_file = tmp_path / 'results' / 'TOL' / 'TranAD_results.json'
    assert results_file.exists()


@patch('TranAD.run_experiment.backprop')
@patch('TranAD.run_experiment.load_model')
@patch('TranAD.utils.load_dataset')
@patch('TranAD.utils.convert_to_windows')
@patch('TranAD.run_experiment.save_model')
@patch('TranAD.utils.plot_accuracies')
@patch('TranAD.plotting.plotter')
@patch('TranAD.pot.pot_eval')
@patch('TranAD.diagnosis.hit_att')
@patch('TranAD.diagnosis.ndcg')
@patch('TranAD.utils.get_git_hash')
def test_run_experiment_with_training(mock_git_hash, mock_ndcg, mock_hit_att, mock_pot_eval,
                                      mock_plotter, mock_plot_accuracies, mock_save_model,
                                      mock_convert_to_windows, mock_load_dataset, mock_load_model, mock_backprop, tmp_path, monkeypatch):
    """Test run_experiment with training enabled."""
    monkeypatch.chdir(tmp_path)
    
    # Setup mocks
    mock_git_hash.return_value = 'def456'
    
    mock_model = Mock()
    mock_model.name = 'USAD'
    mock_model.lr = 0.001
    mock_model.n_window = 10
    mock_optimizer = Mock()
    mock_scheduler = Mock()
    
    mock_load_model.return_value = (mock_model, mock_optimizer, mock_scheduler, -1, [], {'lr': 0.001})
    
    train_data = torch.randn(100, 5).numpy()
    test_data = torch.randn(50, 5).numpy()
    labels = np.random.randint(0, 2, (50, 5))
    
    train_loader = [train_data]
    test_loader = [test_data]
    mock_load_dataset.return_value = (train_loader, test_loader, labels)
    
    # Mock convert_to_windows to return data as-is (simplified)
    mock_convert_to_windows.side_effect = lambda data, *args: torch.from_numpy(data) if isinstance(data, np.ndarray) else data
    
    # Mock backprop for training (5 epochs) + testing + train loss
    training_returns = [(0.5 - i*0.1, 0.001 - i*0.0001) for i in range(5)]
    mock_backprop.side_effect = training_returns + [
        (np.random.rand(50, 5), np.random.rand(50, 5)),  # test
        (np.random.rand(100, 5), np.random.rand(100, 5))  # train loss
    ]
    
    mock_pot_result = {'precision': 0.88, 'recall': 0.92, 'ROC/AUC': 0.96, 'f1': 0.90}
    mock_pot_eval.return_value = (mock_pot_result, np.random.rand(50))
    
    mock_hit_att.return_value = {'hit_rate': 0.85}
    mock_ndcg.return_value = {'ndcg': 0.78}
    
    # Run experiment with training
    result = run_experiment.run_experiment('USAD', 'MSL', test=False)
    
    # Verify training was performed
    assert mock_backprop.call_count >= 5  # at least 5 training epochs
    assert mock_save_model.called
    assert mock_plot_accuracies.called
    
    # Verify results
    assert result['precision'] == 0.88
    assert result['git_hash'] == 'def456'
    assert result['applied_hyperparameters'] == {'lr': 0.001}


@patch('TranAD.merlin.run_merlin')
@patch('TranAD.utils.load_dataset')
@patch('TranAD.run_experiment.append_benchmark_row')
def test_run_experiment_merlin(mock_append, mock_load_dataset, mock_run_merlin, tmp_path, monkeypatch):
    """Test run_experiment with MERLIN model (special case)."""
    monkeypatch.chdir(tmp_path)
    
    # Setup mocks
    train_data = torch.randn(100, 5).numpy()
    test_data = torch.randn(50, 5).numpy()
    labels = np.random.randint(0, 2, (50, 5))
    
    train_loader = [train_data]
    test_loader = [test_data]
    mock_load_dataset.return_value = (train_loader, test_loader, labels)
    
    merlin_result = {'precision': 0.82, 'recall': 0.88, 'ROC/AUC': 0.91, 'f1': 0.85}
    mock_run_merlin.return_value = merlin_result
    
    # Run experiment with MERLIN
    result = run_experiment.run_experiment('MERLIN', 'TOL')
    
    # Verify MERLIN was called
    assert mock_run_merlin.called
    assert result == merlin_result
    assert mock_append.called


@patch('TranAD.run_experiment.backprop')
@patch('TranAD.run_experiment.load_model')
@patch('TranAD.utils.load_dataset')
@patch('TranAD.utils.convert_to_windows')
@patch('TranAD.plotting.plotter')
@patch('TranAD.pot.pot_eval')
@patch('TranAD.diagnosis.hit_att')
@patch('TranAD.diagnosis.ndcg')
@patch('TranAD.utils.get_git_hash')
def test_run_experiment_with_experiment_index(mock_git_hash, mock_ndcg, mock_hit_att, mock_pot_eval,
                                              mock_plotter, mock_convert_to_windows, mock_load_dataset, mock_load_model, 
                                              mock_backprop, tmp_path, monkeypatch):
    """Test run_experiment with experiment_index parameter."""
    monkeypatch.chdir(tmp_path)
    
    # Setup mocks (simplified)
    mock_git_hash.return_value = 'xyz789'
    mock_model = Mock()
    mock_model.name = 'TranAD'
    mock_model.n_window = 10
    mock_load_model.return_value = (mock_model, Mock(), Mock(), -1, [], {})
    
    train_data = torch.randn(100, 5).numpy()
    test_data = torch.randn(50, 5).numpy()
    labels = np.random.randint(0, 2, (50, 5))
    mock_load_dataset.return_value = ([train_data], [test_data], labels)
    
    # Mock convert_to_windows to return data as-is (simplified)
    mock_convert_to_windows.side_effect = lambda data, *args: torch.from_numpy(data) if isinstance(data, np.ndarray) else data
    
    mock_backprop.side_effect = [
        (np.random.rand(50, 5), np.random.rand(50, 5)),
        (np.random.rand(100, 5), np.random.rand(100, 5))
    ]
    
    mock_pot_result = {'precision': 0.85, 'recall': 0.90, 'ROC/AUC': 0.95, 'f1': 0.87}
    mock_pot_eval.return_value = (mock_pot_result, np.random.rand(50))
    mock_hit_att.return_value = {'hit_rate': 0.80}
    mock_ndcg.return_value = {'ndcg': 0.75}
    
    # Run with experiment_index
    result = run_experiment.run_experiment('TranAD', 'TOL', experiment_index=42, test=True)
    
    # Verify experiment_index is in result
    assert result['experiment_index'] == 42
    
    # Verify results file includes experiment index in filename
    results_file = tmp_path / 'results' / 'TOL' / 'TranAD_exp42_results.json'
    assert results_file.exists()


@patch('TranAD.run_experiment.run_experiment')
@patch('TranAD.constants.initialize')
@patch('importlib.reload')
def test_run_all_basic(mock_reload, mock_init, mock_run_experiment, tmp_path, monkeypatch):
    """Test run_all function with basic setup."""
    monkeypatch.chdir(tmp_path)
    
    # Create processed data directory structure
    processed_dir = tmp_path / 'processed'
    (processed_dir / 'TOL').mkdir(parents=True)
    (processed_dir / 'MSL').mkdir(parents=True)
    
    # Mock run_experiment to return results
    mock_run_experiment.side_effect = [
        {'precision': 0.85, 'recall': 0.90, 'ROC/AUC': 0.95, 'f1': 0.87},
        {'precision': 0.88, 'recall': 0.92, 'ROC/AUC': 0.96, 'f1': 0.90},
    ]
    
    # Run all with specific models and datasets
    summary, report = run_experiment.run_all(
        models_list=['TranAD'],
        datasets_list=['TOL', 'MSL']
    )
    
    # Verify experiments were run
    assert len(summary) == 2
    assert ('TranAD', 'TOL') in summary
    assert ('TranAD', 'MSL') in summary
    
    # Verify report DataFrame
    assert len(report) == 2
    assert 'model' in report.columns
    assert 'dataset' in report.columns
    assert 'precision' in report.columns


@patch('TranAD.run_experiment.run_experiment')
@patch('TranAD.constants.initialize')
@patch('importlib.reload')
def test_run_all_skip_existing(mock_reload, mock_init, mock_run_experiment, tmp_path, monkeypatch):
    """Test run_all skips experiments already in benchmarks.csv."""
    monkeypatch.chdir(tmp_path)
    
    # Create processed data directory
    processed_dir = tmp_path / 'processed'
    (processed_dir / 'TOL').mkdir(parents=True)
    
    # Create existing benchmarks.csv with one entry
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    bench_file = results_dir / 'benchmarks.csv'
    with open(bench_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'dataset', 'precision', 'recall', 'AUC', 'f1'])
        writer.writerow(['TranAD', 'TOL', '0.85', '0.90', '0.95', '0.87'])
    
    # Run all - should skip TranAD on TOL
    summary, report = run_experiment.run_all(
        models_list=['TranAD'],
        datasets_list=['TOL']
    )
    
    # Verify experiment was skipped (run_experiment not called)
    assert not mock_run_experiment.called
    assert len(summary) == 0


@patch('TranAD.run_experiment.run_experiment')
@patch('TranAD.constants.initialize')
@patch('importlib.reload')
def test_run_all_with_retrain(mock_reload, mock_init, mock_run_experiment, tmp_path, monkeypatch):
    """Test run_all with retrain flag ignores existing benchmarks."""
    monkeypatch.chdir(tmp_path)
    
    # Create processed data directory
    processed_dir = tmp_path / 'processed'
    (processed_dir / 'TOL').mkdir(parents=True)
    
    # Create existing benchmarks.csv
    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    bench_file = results_dir / 'benchmarks.csv'
    with open(bench_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'dataset', 'precision', 'recall', 'AUC', 'f1'])
        writer.writerow(['TranAD', 'TOL', '0.85', '0.90', '0.95', '0.87'])
    
    mock_run_experiment.return_value = {'precision': 0.88, 'recall': 0.92, 'ROC/AUC': 0.96, 'f1': 0.90}
    
    # Create mock args object with retrain=True
    args_obj = Mock()
    args_obj.retrain = True
    
    # Run all with retrain flag
    summary, report = run_experiment.run_all(
        models_list=['TranAD'],
        datasets_list=['TOL'],
        args_obj=args_obj
    )
    
    # Verify experiment was NOT skipped
    assert mock_run_experiment.called
    assert len(summary) == 1


@patch('TranAD.run_experiment.run_experiment')
@patch('TranAD.constants.initialize')
@patch('importlib.reload')
def test_run_all_handles_errors(mock_reload, mock_init, mock_run_experiment, tmp_path, monkeypatch):
    """Test run_all handles exceptions in run_experiment."""
    monkeypatch.chdir(tmp_path)
    
    # Create processed data directory
    processed_dir = tmp_path / 'processed'
    (processed_dir / 'TOL').mkdir(parents=True)
    
    # Mock run_experiment to raise exception
    mock_run_experiment.side_effect = ValueError("Test error")
    
    # Run all - should not crash
    summary, report = run_experiment.run_all(
        models_list=['TranAD'],
        datasets_list=['TOL']
    )
    
    # Verify error was captured
    assert ('TranAD', 'TOL') in summary
    assert 'error' in summary[('TranAD', 'TOL')]
    assert 'Test error' in str(summary[('TranAD', 'TOL')])






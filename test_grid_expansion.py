#!/usr/bin/env python3
"""
Quick test of parameter grid and array job functionality.
Run with: python test_grid_expansion.py
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from main import (
    generate_parameter_grid,
    expand_experiment_to_subexperiments,
    collect_all_subexperiments,
    map_array_index_to_subexperiment
)


def test_parameter_grid():
    """Test parameter grid generation."""
    print("=" * 60)
    print("TEST 1: Parameter Grid Generation")
    print("=" * 60)
    
    # Test case 1: Multiple lists
    params1 = {'lr': [0.001, 0.002], 'beta': [0.01, 0.1], 'n_hidden': [16, 32]}
    grid1 = generate_parameter_grid(params1)
    print(f"\nInput: {params1}")
    print(f"Output: {len(grid1)} combinations")
    for i, combo in enumerate(grid1):
        print(f"  {i}: {combo}")
    assert len(grid1) == 8, f"Expected 8 combinations, got {len(grid1)}"
    
    # Test case 2: Mix of list and scalar
    params2 = {'lr': 0.001, 'beta': [0.01, 0.1]}
    grid2 = generate_parameter_grid(params2)
    print(f"\nInput: {params2}")
    print(f"Output: {len(grid2)} combinations")
    for i, combo in enumerate(grid2):
        print(f"  {i}: {combo}")
    assert len(grid2) == 2, f"Expected 2 combinations, got {len(grid2)}"
    
    # Test case 3: All scalars
    params3 = {'lr': 0.001, 'beta': 0.01}
    grid3 = generate_parameter_grid(params3)
    print(f"\nInput: {params3}")
    print(f"Output: {len(grid3)} combinations")
    for i, combo in enumerate(grid3):
        print(f"  {i}: {combo}")
    assert len(grid3) == 1, f"Expected 1 combination, got {len(grid3)}"
    
    print("\n✓ All parameter grid tests passed!\n")


def test_experiment_expansion():
    """Test experiment expansion to sub-experiments."""
    print("=" * 60)
    print("TEST 2: Experiment Expansion")
    print("=" * 60)
    
    exp = {
        'index': 5,
        'dataset': 'TOL',
        'model': 'OmniAnomaly',
        'lr': [0.001, 0.002],
        'beta': [0.01, 0.1],
        'notes': 'Test experiment'
    }
    
    sub_exps = expand_experiment_to_subexperiments(exp)
    print(f"\nExperiment {exp['index']}: {exp['model']}")
    print(f"  Parameters: lr={exp['lr']}, beta={exp['beta']}")
    print(f"  Expanded to {len(sub_exps)} sub-experiments:")
    
    for exp_id, params in sub_exps:
        print(f"    {exp_id}: {params}")
    
    assert len(sub_exps) == 4, f"Expected 4 sub-experiments, got {len(sub_exps)}"
    assert sub_exps[0][0] == "5.0", f"Expected first ID to be '5.0', got '{sub_exps[0][0]}'"
    assert sub_exps[3][0] == "5.3", f"Expected last ID to be '5.3', got '{sub_exps[3][0]}'"
    
    print("\n✓ Experiment expansion test passed!\n")


def test_collection():
    """Test collecting all sub-experiments from multiple experiments."""
    print("=" * 60)
    print("TEST 3: Sub-experiment Collection")
    print("=" * 60)
    
    experiments = [
        {
            'index': 1,
            'dataset': 'TOL',
            'model': 'LSTM_Univariate',
            'lr': [0.001, 0.002],
            'batch': [64, 128]
        },
        {
            'index': 2,
            'dataset': 'TOL',
            'model': 'Attention',
            'lr': 0.0001,
            'n_window': [5, 10]
        }
    ]
    
    all_subexps = collect_all_subexperiments(experiments)
    
    print(f"\n{len(experiments)} experiments:")
    for exp in experiments:
        print(f"  Exp {exp['index']}: {exp['model']}")
    
    print(f"\nExpanded to {len(all_subexps)} total sub-experiments:")
    for exp_id, dataset, model, params in all_subexps:
        print(f"  {exp_id}: {model} - {params}")
    
    # Exp 1: 2 × 2 = 4 sub-experiments
    # Exp 2: 1 × 2 = 2 sub-experiments
    # Total: 6
    assert len(all_subexps) == 6, f"Expected 6 sub-experiments, got {len(all_subexps)}"
    
    print("\n✓ Collection test passed!\n")


def test_array_mapping():
    """Test array index to sub-experiment mapping."""
    print("=" * 60)
    print("TEST 4: Array Index Mapping")
    print("=" * 60)
    
    experiments = [
        {
            'index': 1,
            'dataset': 'TOL',
            'model': 'TestModel',
            'lr': [0.001, 0.002, 0.003]
        }
    ]
    
    # Test valid indices
    for idx in [0, 1, 2]:
        result = map_array_index_to_subexperiment(experiments, idx)
        assert result is not None, f"Array index {idx} should map to a sub-experiment"
        exp_id, dataset, model, params = result
        print(f"  Array index {idx} → Exp {exp_id}: {model}, lr={params['lr']}")
    
    # Test out of range
    result = map_array_index_to_subexperiment(experiments, 999)
    assert result is None, "Out of range index should return None"
    print(f"  Array index 999 → None (out of range)")
    
    print("\n✓ Array mapping test passed!\n")


if __name__ == '__main__':
    print("\nRunning parameter grid and array job tests...\n")
    
    try:
        test_parameter_grid()
        test_experiment_expansion()
        test_collection()
        test_array_mapping()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        print("\nThe parameter grid and array job system is working correctly.")
        print("You can now:")
        print("  1. Use --list to see all sub-experiments")
        print("  2. Use --status to check completion")
        print("  3. Use --array-index for SLURM/PBS array jobs")
        print()
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

import pytest
import torch
import random
import numpy as np
import json
import pathlib
from tranAD.models import (
    LSTM_Univariate, Attention, LSTM_AD, DAGMM, OmniAnomaly, USAD, MSCRED,
    CAE_M, MTAD_GAT, GDN, MAD_GAN, TranAD_Basic, TranAD_Transformer,
    TranAD_Adversarial, TranAD_SelfConditioning, TranAD
)

spec_path = pathlib.Path(__file__).parent / "expected_model_outputs.json"
with open(spec_path, "r") as f:
    EXPECTED_MODEL_OUTPUTS = json.load(f)

@pytest.mark.parametrize("ModelClass, input_shape", [
    (LSTM_Univariate, (5, 5, 1)),      # feats=1, batch=5, seq_len=5
    (Attention, (5, 5)),               # seq_len=5, feats=5
    (LSTM_AD, (5, 5)),                 # seq_len=5, feats=5
    (DAGMM, (1, 25)),                  # batch=1, flattened seq_len*feats=25
    (OmniAnomaly, (5,)),               # feats=5
    (USAD, (1, 25)),                   # batch=1, flattened seq_len*feats=25
    (MSCRED, (1, 25)),                 # batch=1, flattened seq_len*feats=25
    (CAE_M, (1, 25)),                  # batch=1, flattened seq_len*feats=25
    (MTAD_GAT, (1, 25)),               # batch=1, flattened seq_len*feats=25
    (GDN, (1, 25)),                    # batch=1, flattened seq_len*feats=25
    (MAD_GAN, (1, 25)),                # batch=1, flattened seq_len*feats=25
    (TranAD_Basic, ((10, 5, 5), (10, 5, 5))),  # src, tgt shapes
    (TranAD_Transformer, ((10, 5, 5), (10, 5, 5))),
    (TranAD_Adversarial, ((10, 5, 5), (10, 5, 5))),
    (TranAD_SelfConditioning, ((10, 5, 5), (10, 5, 5))),
    (TranAD, ((10, 5, 5), (10, 5, 5))),
])
def test_model_forward(ModelClass, input_shape):
    feats = 5
    model = ModelClass(feats)
    model.eval()
    with torch.no_grad():
        if isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
            # Models with src, tgt
            src = torch.randn(*input_shape[0], dtype=torch.float64)
            tgt = torch.randn(*input_shape[1], dtype=torch.float64)
            output = model(src, tgt)
        else:
            x = torch.randn(*input_shape, dtype=torch.float64)
            output = model(x)
        assert output is not None


@pytest.mark.parametrize("ModelClass, input_shape", [
    (LSTM_Univariate, (5, 5, 1)),
    (Attention, (5, 5)),
    (LSTM_AD, (5, 5)),
    (DAGMM, (1, 25)),
    (OmniAnomaly, (5,)),
    (USAD, (1, 25)),
    (MSCRED, (1, 25)),
    (CAE_M, (1, 25)),
    (MTAD_GAT, (1, 25)),
    (GDN, (1, 25)),
    (MAD_GAN, (1, 25)),
    (TranAD_Basic, ((10, 5, 5), (10, 5, 5))),
    (TranAD_Transformer, ((10, 5, 5), (10, 5, 5))),
    (TranAD_Adversarial, ((10, 5, 5), (10, 5, 5))),
    (TranAD_SelfConditioning, ((10, 5, 5), (10, 5, 5))),
    (TranAD, ((10, 5, 5), (10, 5, 5))),
])
def test_model_backward(ModelClass, input_shape):
    feats = 5
    model = ModelClass(feats)
    model.train()

    # create inputs with gradients where appropriate (double dtype to match model)
    if isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
        src = torch.randn(*input_shape[0], dtype=torch.float64, requires_grad=True)
        tgt = torch.randn(*input_shape[1], dtype=torch.float64, requires_grad=True)
        output = model(src, tgt)
    else:
        x = torch.randn(*input_shape, dtype=torch.float64, requires_grad=True)
        output = model(x)

    # reduce output(s) to a scalar loss
    if isinstance(output, (tuple, list)):
        loss = 0
        for o in output:
            if isinstance(o, (tuple, list)):
                for p in o:
                    loss = loss + p.view(-1).sum()
            else:
                loss = loss + o.view(-1).sum()
    else:
        loss = output.view(-1).sum()

    # backward and check that gradients exist for model parameters
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    # ensure at least one parameter has a non-zero gradient
    assert any((g.abs().sum() > 0) for g in grads)


@pytest.mark.parametrize("ModelClass, input_shape", [
    (LSTM_Univariate, (5, 5, 1)),
    (Attention, (5, 5)),
    (LSTM_AD, (5, 5)),
    (DAGMM, (1, 25)),
    (OmniAnomaly, (5,)),
    (USAD, (1, 25)),
    (MSCRED, (1, 25)),
    (CAE_M, (1, 25)),
    (MTAD_GAT, (1, 25)),
    (GDN, (1, 25)),
    (MAD_GAN, (1, 25)),
    (TranAD_Basic, ((10, 5, 5), (10, 5, 5))),
    (TranAD_Transformer, ((10, 5, 5), (10, 5, 5))),
    (TranAD_Adversarial, ((10, 5, 5), (10, 5, 5))),
    (TranAD_SelfConditioning, ((10, 5, 5), (10, 5, 5))),
    (TranAD, ((10, 5, 5), (10, 5, 5))),
])
def test_model_fixed_input_outputs(ModelClass, input_shape):
    """Deterministic call to each model using a simple fixed input.

    This test intentionally asserts the output equals a zero tensor of the
    same shape so pytest will show the actual outputs. After capturing the
    real outputs from the failure messages, these assertions will be updated
    to the correct expected values.
    """
    # make behavior reproducible across runs
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    feats = 5
    model = ModelClass(feats)
    model.eval()
    with torch.no_grad():
        if isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
            src = torch.ones(*input_shape[0], dtype=torch.float64)
            tgt = torch.ones(*input_shape[1], dtype=torch.float64)
            output = model(src, tgt)
        else:
            x = torch.ones(*input_shape, dtype=torch.float64)
            output = model(x)

        # Compare against recorded expected outputs
        name = ModelClass.__name__
        assert name in EXPECTED_MODEL_OUTPUTS, f"No expected output for {name}"

        def assert_structure_close(actual, expected_py):
            if isinstance(actual, torch.Tensor):
                expected_t = torch.tensor(expected_py, dtype=actual.dtype)
                torch.testing.assert_close(actual, expected_t)
                return
            if isinstance(actual, (tuple, list)):
                assert isinstance(expected_py, (list, tuple)), (
                    "Structure mismatch: expected sequence for ", name
                )
                assert len(actual) == len(expected_py)
                for a, e in zip(actual, expected_py):
                    assert_structure_close(a, e)
                return
            raise AssertionError(f"Unhandled output type: {type(actual)}")

        expected_py = EXPECTED_MODEL_OUTPUTS[name]
        assert_structure_close(output, expected_py)



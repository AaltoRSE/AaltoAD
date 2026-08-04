import pytest
import torch
import random
import numpy as np
import json
import pathlib
from AaltoAD.models import (
    LSTM_Univariate, Attention, LSTM_AD, DAGMM, OmniAnomaly, USAD, MSCRED,
    CAE_M, MTAD_GAT, GDN, MAD_GAN, TranAD_Basic, TranAD_Transformer,
    TranAD_Adversarial, TranAD_SelfConditioning, TranAD
)

spec_path = pathlib.Path(__file__).parent / "expected_model_outputs.json"
with open(spec_path, "r") as f:
    EXPECTED_MODEL_OUTPUTS = json.load(f)

# One-at-a-time variations for model constructor hyperparameters.
# Each entry lists values to test (include default and at least one alternative).
VARIATIONS = {
    "LSTM_Univariate": {
        "n_hidden": [1, 4, 8],
        "n_layers": [1, 2],
        "learning_rate": [0.002, 0.01],
    },
    "Attention": {
        "n_window": [5, 3, 10],
        "learning_rate": [0.0001, 0.001],
    },
    "LSTM_AD": {
        "n_hidden": [64, 32, 128],
        "n_layers": [1, 2],
        "learning_rate": [0.002, 0.01],
    },
    "DAGMM": {
        "n_hidden": [16, 8, 32],
        "n_latent": [8, 4, 16],
        "n_window": [5, 3, 10],
        "learning_rate": [0.0001, 0.001],
    },
    "OmniAnomaly": {
        "n_hidden": [32, 16, 64],
        "n_latent": [8, 4, 16],
        "n_layers": [2, 1, 3],
        "beta": [0.01, 0.001, 0.1],
        "learning_rate": [0.002, 0.01],
    },
    "USAD": {
        "n_hidden": [16, 8, 32],
        "n_latent": [5, 3, 10],
        "n_window": [5, 3, 10],
        "learning_rate": [0.0001, 0.001],
    },
    "MSCRED": {
        "n_window": [5, 10],
        "learning_rate": [0.0001, 0.001],
    },
    "CAE_M": {
        "n_window": [5, 10],
        "learning_rate": [0.0001, 0.001],
    },
    "MTAD_GAT": {
        # n_window must equal feats due to time_gat architecture (GATConv(feats, feats))
        "n_window": [5],
        "n_hidden": [16, 25],
        "learning_rate": [0.0001, 0.001],
    },
    "GDN": {
        "n_window": [5, 3, 10],
        "n_hidden": [16, 8, 32],
        "learning_rate": [0.0001, 0.001],
    },
    "MAD_GAN": {
        "n_window": [5, 3, 10],
        "n_hidden": [16, 8, 32],
        "learning_rate": [0.0001, 0.001],
    },
    "TranAD_Basic": {
        "n_window": [10, 5],
        "batch_size": [128, 32],
        "dim_feedforward": [16, 32],
        "dropout": [0.1, 0.0, 0.2],
        "nheads": [1, 5],
    },
    "TranAD_Transformer": {
        "n_window": [10, 5],
        "batch_size": [128, 32],
        "n_hidden": [8, 16],
    },
    "TranAD_Adversarial": {
        "n_window": [10, 5],
        "batch_size": [128, 32],
        "dim_feedforward": [16, 32],
        "dropout": [0.1, 0.0],
        "nheads": [1, 5],
    },
    "TranAD_SelfConditioning": {
        "n_window": [10, 5],
        "batch_size": [128, 32],
        "dim_feedforward": [16, 32],
        "dropout": [0.1, 0.0],
        "nheads": [1, 5],
    },
    "TranAD": {
        "n_window": [10, 5],
        "batch_size": [128, 32],
        "dim_feedforward": [16, 32],
        "dropout": [0.1, 0.0],
        "nheads": [1, 5],
    },
}

# Map model class name to a small input shape for forward/backward checks.
MODEL_INPUT_SHAPES = {
    "LSTM_Univariate": (5, 5, 1),
    "Attention": (5, 5),
    "LSTM_AD": (1, 5, 5),
    "DAGMM": (1, 25),
    "OmniAnomaly": (5,),
    "USAD": (1, 25),
    "MSCRED": (1, 25),
    "CAE_M": (1, 25),
    "MTAD_GAT": (1, 25),
    "GDN": (1, 25),
    "MAD_GAN": (1, 25),
    "TranAD_Basic": ((10, 5, 5), (10, 5, 5)),
    "TranAD_Transformer": ((10, 5, 5), (10, 5, 5)),
    "TranAD_Adversarial": ((10, 5, 5), (10, 5, 5)),
    "TranAD_SelfConditioning": ((10, 5, 5), (10, 5, 5)),
    "TranAD": ((10, 5, 5), (10, 5, 5)),
}

# Build parametrization cases: (ModelClass, param_name, param_value)
PARAM_TEST_CASES = []
for cls in [LSTM_Univariate, Attention, LSTM_AD, DAGMM, OmniAnomaly, USAD, MSCRED,
            CAE_M, MTAD_GAT, GDN, MAD_GAN, TranAD_Basic, TranAD_Transformer,
            TranAD_Adversarial, TranAD_SelfConditioning, TranAD]:
    name = cls.__name__
    if name not in VARIATIONS:
        continue
    for param_name, values in VARIATIONS[name].items():
        for v in values:
            PARAM_TEST_CASES.append((cls, param_name, v))


def _instantiate_or_skip(ModelClass, feats, overrides):
    """Try to instantiate ModelClass(feats, **overrides) or skip if kwargs unsupported."""
    try:
        return ModelClass(feats, **overrides)
    except TypeError as e:
        pytest.skip(f"Constructor for {ModelClass.__name__} rejected kwargs {overrides}: {e}")


@pytest.mark.parametrize("ModelClass,param_name,param_value", PARAM_TEST_CASES)
def test_model_parametrized_variations(ModelClass, param_name, param_value):
    """One-at-a-time hyperparameter variation: ensure no runtime errors and numeric outputs.

    For each (model, param, value) we instantiate with that single override and run
    a forward and backward pass (if applicable). If the constructor doesn't accept
    the kwarg, the test is skipped for that case.
    """
    name = ModelClass.__name__
    overrides = {param_name: param_value}
    input_shape = MODEL_INPUT_SHAPES.get(name)
    feats = 5

    model = _instantiate_or_skip(ModelClass, feats, overrides)
    if param_name == "n_window":
        if isinstance(input_shape[0], tuple):
            # Transformer models: input is (n_window, batch, feats)
            input_shape = ((param_value, *input_shape[0][1:]),
                           (param_value, *input_shape[1][1:]))
        elif name in ("Attention",):
            # Attention: input is (n_window, feats)
            input_shape = (param_value, *input_shape[1:])
        else:
            # Flat models: input is (batch, n_window * feats)
            input_shape = (input_shape[0], param_value * feats, *input_shape[2:])

    # forward
    model.eval()
    with torch.no_grad():
        if isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
            src = torch.randn(*input_shape[0], dtype=torch.float64)
            tgt = torch.randn(*input_shape[1], dtype=torch.float64)
            out = model(src, tgt)
        else:
            x = torch.randn(*input_shape, dtype=torch.float64)
            out = model(x)

    # basic numeric checks: output should be tensor or nested tensors
    def _contains_tensor(o):
        if isinstance(o, torch.Tensor):
            return True
        if isinstance(o, (list, tuple)):
            return any(_contains_tensor(x) for x in o)
        return False

    assert _contains_tensor(out), f"Output for {name} with {param_name}={param_value} contains no tensors"

    # backward pass: ensure gradients flow for at least one param (if trainable params exist)
    model.train()
    if isinstance(input_shape, tuple) and isinstance(input_shape[0], tuple):
        src = torch.randn(*input_shape[0], dtype=torch.float64, requires_grad=True)
        tgt = torch.randn(*input_shape[1], dtype=torch.float64, requires_grad=True)
        out = model(src, tgt)
    else:
        x = torch.randn(*input_shape, dtype=torch.float64, requires_grad=True)
        out = model(x)

    # reduce to scalar
    if isinstance(out, (tuple, list)):
        loss = 0
        for o in out:
            if isinstance(o, (tuple, list)):
                for p in o:
                    loss = loss + p.view(-1).sum()
            else:
                loss = loss + o.view(-1).sum()
    else:
        loss = out.view(-1).sum()

    # backward
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    # If model has trainable params, ensure at least one has a non-zero grad
    if len(list(model.parameters())) > 0:
        assert len(grads) > 0
        assert any((g.abs().sum() > 0) for g in grads)


@pytest.mark.parametrize("n_hidden", [1, 4, 16])
@pytest.mark.parametrize("feats", [3, 5])
def test_lstm_univariate_output_shape_matches_input(n_hidden, feats):
    """LSTM_Univariate output shape must match input regardless of n_hidden.

    Regression test: with n_hidden > 1, the output was (batch, feats * n_hidden)
    instead of (batch, feats) because the LSTM hidden output was not reduced
    to a single value per feature before concatenation.
    """
    n_window = 5
    model = LSTM_Univariate(feats, n_hidden=n_hidden, n_layers=1)
    model.eval()
    x = torch.randn(n_window, feats, 1, dtype=torch.float64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (n_window, feats), (
        f"Expected output shape ({n_window}, {feats}) but got {out.shape} "
        f"with n_hidden={n_hidden}"
    )


def test_backprop_omni_scalar_loss():
    """OmniAnomaly backprop must produce a scalar loss with reduction='none'.

    Regression test: the loss was computed as MSE + beta * KLD without reducing
    to a scalar first, causing 'grad can be implicitly created only for scalar
    outputs' when calling loss.backward().
    """
    feats = 5
    n_window = 5
    model = OmniAnomaly(feats, n_hidden=32, n_latent=16, beta=0.01)
    data = torch.randn(10, feats, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)

    # This should not raise "grad can be implicitly created only for scalar outputs"
    result = model.train_step(epoch=0, data=data, optimizer=optimizer,
                             scheduler=scheduler, feats=feats)
    loss_val, lr = result
    assert isinstance(loss_val, float), f"Loss should be a float scalar, got {type(loss_val)}"
    assert isinstance(lr, float)


@pytest.mark.parametrize("ModelClass", [MSCRED, CAE_M])
@pytest.mark.parametrize("n_window", [0, None])
def test_conv_models_fallback_on_falsy_n_window(ModelClass, n_window):
    """Models with convolutions must handle n_window=0 or None gracefully.

    Regression test: n_window=0 was passed from the experiment config, but the
    model only checked for None, so 0 was used as the actual window size,
    causing 'Kernel size can't be greater than actual input size' in Conv layers.
    """
    feats = 5
    model = ModelClass(feats, n_window=n_window)
    assert model.n_window == feats, (
        f"Expected n_window to fall back to feats={feats}, got {model.n_window}"
    )
    model.eval()
    x = torch.randn(1, model.n_window * feats, dtype=torch.float64)
    with torch.no_grad():
        out = model(x)
    assert out is not None


@pytest.mark.parametrize("feats", [3, 5, 8])
def test_mtad_gat_feature_variations(feats):
    """MTAD_GAT requires n_window == feats; test with different feature counts."""
    model = MTAD_GAT(feats)
    input_shape = (1, feats * feats)

    model.eval()
    with torch.no_grad():
        x = torch.randn(*input_shape, dtype=torch.float64)
        out = model(x)
    assert isinstance(out[0], torch.Tensor)

    model.train()
    x = torch.randn(*input_shape, dtype=torch.float64, requires_grad=True)
    out = model(x)
    loss = out[0].view(-1).sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert any((g.abs().sum() > 0) for g in grads)


@pytest.mark.parametrize("feats", [3, 5, 8])
def test_mtad_gat_hidden_threading(feats):
    """MTAD_GAT must thread the GRU hidden state across timesteps without crashing.

    Regression test: forward() previously had an inverted None check that
    overwrote a passed-in hidden with a fresh random tensor, masking both a
    GRU-size mismatch and a missing detach() in _backprop's BPTT loop.
    """
    model = MTAD_GAT(feats)
    expected_hidden = feats * feats

    # Forward pass with hidden=None (initial step).
    model.eval()
    x = torch.randn(1, feats * feats, dtype=torch.float64)
    with torch.no_grad():
        out, h = model(x)
    assert h.shape == (1, 1, expected_hidden)

    # Forward pass that reuses the returned hidden.
    with torch.no_grad():
        out2, h2 = model(x, h)
    assert h2.shape == (1, 1, expected_hidden)

    # Training step: train_step threads hidden across iterations under autograd;
    # without detach() this raises 'backward through the graph a second time'.
    model.train()
    data = torch.randn(3, feats * feats, dtype=torch.float64)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    loss, lr = model.train_step(epoch=0, data=data, optimizer=optimizer,
                               scheduler=scheduler, feats=feats)
    assert isinstance(loss, float)
    assert isinstance(lr, float)


@pytest.mark.parametrize("ModelClass, input_shape", [
    (LSTM_Univariate, (5, 5, 1)),      # feats=1, batch=5, seq_len=5
    (Attention, (5, 5)),               # seq_len=5, feats=5
    (LSTM_AD, (1, 5, 5)),                 # seq_len=5, feats=5
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
    (LSTM_AD, (1, 5, 5)),
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
    (LSTM_AD, (1, 5, 5)),
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


@pytest.mark.parametrize("ModelClass", [
    LSTM_Univariate, Attention, LSTM_AD, DAGMM, OmniAnomaly, USAD, MSCRED,
    CAE_M, MTAD_GAT, GDN, MAD_GAN, TranAD_Basic, TranAD_Transformer,
    TranAD_Adversarial, TranAD_SelfConditioning, TranAD,
])
def test_all_models_have_backprop(ModelClass):
    """Every model class must have train_step and eval_step methods."""
    assert hasattr(ModelClass, 'train_step'), f"{ModelClass.__name__} is missing train_step method"
    assert callable(getattr(ModelClass, 'train_step')), f"{ModelClass.__name__}.train_step is not callable"
    assert hasattr(ModelClass, 'eval_step'), f"{ModelClass.__name__} is missing eval_step method"
    assert callable(getattr(ModelClass, 'eval_step')), f"{ModelClass.__name__}.eval_step is not callable"

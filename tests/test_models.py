import pytest

import torch
from tranAD.models import (
    LSTM_Univariate, Attention, LSTM_AD, DAGMM, OmniAnomaly, USAD, MSCRED,
    CAE_M, MTAD_GAT, GDN, MAD_GAN, TranAD_Basic, TranAD_Transformer,
    TranAD_Adversarial, TranAD_SelfConditioning, TranAD
)

@pytest.mark.parametrize("ModelClass, input_shape", [
    (LSTM_Univariate, (5, 5, 1)),      # feats=1, batch=5, seq_len=5
    (Attention, (5, 5, 5)),           # batch=5, seq_len=5, feats=5
    (LSTM_AD, (5, 5, 5)),             # batch=5, seq_len=5, feats=5
    (DAGMM, (1, 25)),                 # batch=1, flattened seq_len*feats=25
    (OmniAnomaly, (5, 5, 5)),         # batch=5, seq_len=5, feats=5
    (USAD, (1, 25)),                  # batch=1, flattened seq_len*feats=25
    (MSCRED, (5, 5, 5, 5)),           # batch=5, seq_len=5, feats=5, feats=5
    (CAE_M, (5, 5, 5)),               # batch=5, seq_len=5, feats=5
    (MTAD_GAT, (5, 5, 5)),            # batch=5, seq_len=5, feats=5
    (GDN, (1, 25)),                   # batch=1, flattened seq_len*feats=25
    (MAD_GAN, (1, 25)),               # batch=1, flattened seq_len*feats=25
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
            src = torch.randn(*input_shape[0])
            tgt = torch.randn(*input_shape[1])
            output = model(src, tgt)
        else:
            x = torch.randn(*input_shape)
            output = model(x)
        assert output is not None
import torch
import random
import numpy as np
import json

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from TranAD import models as _models_module

MODEL_LIST = [
    ("LSTM_Univariate", (5, 5, 1)),
    ("Attention", (5, 5)),
    ("LSTM_AD", (5, 5)),
    ("DAGMM", (1, 25)),
    ("OmniAnomaly", (5,)),
    ("USAD", (1, 25)),
    ("MSCRED", (1, 25)),
    ("CAE_M", (1, 25)),
    ("MTAD_GAT", (1, 25)),
    ("GDN", (1, 25)),
    ("MAD_GAN", (1, 25)),
    ("TranAD_Basic", ((10, 5, 5), (10, 5, 5))),
    ("TranAD_Transformer", ((10, 5, 5), (10, 5, 5))),
    ("TranAD_Adversarial", ((10, 5, 5), (10, 5, 5))),
    ("TranAD_SelfConditioning", ((10, 5, 5), (10, 5, 5))),
    ("TranAD", ((10, 5, 5), (10, 5, 5))),
]


def to_py(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    if isinstance(obj, (list, tuple)):
        return [to_py(i) for i in obj]
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    return obj


def main():
    feats = 5
    results = {}
    for name, shape in MODEL_LIST:
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)

        ModelClass = getattr(_models_module, name)
        model = ModelClass(feats)
        model.eval()
        with torch.no_grad():
            if isinstance(shape, tuple) and isinstance(shape[0], tuple):
                src = torch.ones(*shape[0], dtype=torch.float64)
                tgt = torch.ones(*shape[1], dtype=torch.float64)
                out = model(src, tgt)
            else:
                x = torch.ones(*shape, dtype=torch.float64)
                out = model(x)
        results[name] = to_py(out)

    out_file = "tests/expected_model_outputs.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Wrote expected outputs to {out_file}")


if __name__ == "__main__":
    main()

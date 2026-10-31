# model-compressor

Graph-level model optimization: import exported PyTorch programs, adapt them to a hardware
generation's operator set, and compress them.

## Layout

```
src/model_compressor/   the library (import as `model_compressor`)
  api/                  entry points: import, prepare, finalize, config, catalog
  database/             hardware-generation schemas and operator databases
  nn/                   optimizing module replacements (conv, linear, sdpa, norms, ...)
  ops/                  the `xaten` custom operator namespace (encode/decode/recode/approximate)
  passes/               fx/networkx passes: modularize, adapt, validate
examples/               end-to-end flows, models, TensorRT/ONNX emission
```

## Environment

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11+ (3.12 is pinned in `.python-version`).

```sh
uv sync --extra examples --extra onnx   # create .venv and install
uv run python -c "import model_compressor"
```

Extras:

| extra      | contents                                                        |
| ---------- | --------------------------------------------------------------- |
| `examples` | torchvision, transformers, segmentation-models-pytorch, ipython |
| `onnx`     | onnx, for `examples/trt/emit_common.py`                         |
| `dev`      | pytest, ruff, mypy                                              |

## Running the examples

`examples/flow.py` imports the models as `models.*`, which resolves because the script's own
directory lands on `sys.path`:

```sh
uv run --extra examples python examples/flow.py --model-name ResNet18 \
    --hardware-generation alpha --policy w --assignment-mode strict --coalescence-mode without
```

`examples/script.sh` sweeps every model / policy / mode combination.

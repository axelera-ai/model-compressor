# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Exemplify the refactored model optimization flow - excluding the optimization stage."""

import argparse
import csv
import dataclasses
import enum
from collections.abc import Callable
from typing import Any

import IPython
import cost
import model_compressor.api as api
import model_compressor.ops  # noqa: F401 - register the xaten custom operators
import model_compressor.passes as passes
import model_compressor.passes.adapt.base as base
import models.resnet as _resnet
import models.routing as _routing
import models.shufflenetv2 as _shufflenetv2
import models.smolvlm2 as _smolvlm2
import models.unet as _unet
import torch
import torch.fx as fx

# Each factory returns a ready ``(GraphModule, input)`` pair: a module traced with ``torch.export``
# and a valid keyword-argument input.
_MODEL_NAME_TO_FACTORY: dict[
    str,
    Callable[[], tuple[fx.GraphModule, dict[str, Any]]],
] = {
    "Routing": _routing.create_routing,
    "ResNet18": _resnet.create_resnet18,
    "ShuffleNetV2": _shufflenetv2.create_shufflenetv2,
    "U-Net": _unet.create_unet,
    "SmolLM2-135M": _smolvlm2.create_smolvlm2_language,
    "SigLIP": _smolvlm2.create_smolvlm2_vision,
}


class _HardwareGeneration(enum.Enum):
    """A hardware platform generation."""

    ALPHA = enum.auto()


class _Policy(enum.Enum):
    """A policy to select operators."""

    Z = enum.auto()  # unquantized F32 reference
    Y = enum.auto()  # SI8 matmuls, F32 routing
    X = enum.auto()  # SI8 matmuls, SI8 routing
    W = enum.auto()  # SI8 everywhere (every operator with an SI8 variant)


# The indexing axis is the ``(generation, policy)`` pair: it selects the platform's config
# (database) and the picker policy that resolves the per-operator schema.
_SETUP_TO_CONFIG_AND_PICKER: dict[
    tuple[_HardwareGeneration, _Policy], tuple[Callable[[], Any], passes.annotate.Picker]
] = {
    (_HardwareGeneration.ALPHA, _Policy.X): (api.generations.alpha.Config, api.generations.alpha.X_PICKER),
    (_HardwareGeneration.ALPHA, _Policy.Y): (api.generations.alpha.Config, api.generations.alpha.Y_PICKER),
    (_HardwareGeneration.ALPHA, _Policy.Z): (api.generations.alpha.Config, api.generations.alpha.Z_PICKER),
    (_HardwareGeneration.ALPHA, _Policy.W): (api.generations.alpha.Config, api.generations.alpha.W_PICKER),
}


def _create_graph_module_and_input(model_name: str) -> tuple[fx.GraphModule, dict[str, Any]]:
    return _MODEL_NAME_TO_FACTORY[model_name]()


def _create__key_to_cost() -> dict[tuple[str, str, str, str, str], dict[str, int]]:
    _key_to_cost = {}
    with open("examples/cost_matrix.csv", "r") as fp:
        reader = csv.DictReader(fp)
        fields = reader.fieldnames
        for row in reader:
            key = tuple((row[k] for k in fields[:5]))
            cost = {k: int(v) for k, v in row.items() if (k not in fields[:5])}
            _key_to_cost[key] = cost
    return _key_to_cost


_KEY_TO_COST = _create__key_to_cost()


@dataclasses.dataclass
class _AdaptationState:
    number_of_nodes: int
    number_of_true: int
    number_of_fake: int


_Setup = tuple[_HardwareGeneration, _Policy]
_Mode = tuple[passes.adapt.AssignmentMode, passes.adapt.CoalescenceMode]

_MODEL_NAME_TO_SETUP_TO_MODE_TO_ADAPTATION_STATE: dict[str, dict[_Setup, dict[_Mode, _AdaptationState]]] = {
    "Routing": {
        (_HardwareGeneration.ALPHA, _Policy.Z): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(17, 7, 0),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(17, 1, 0),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(17, 7, 0),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(17, 1, 0),
        },
        (_HardwareGeneration.ALPHA, _Policy.Y): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(17, 7, 0),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(17, 1, 0),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(17, 7, 0),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(17, 1, 0),
        },
        (_HardwareGeneration.ALPHA, _Policy.X): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(17, 7, 3),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(17, 7, 3),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(17, 4, 3),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(17, 4, 1),
        },
        (_HardwareGeneration.ALPHA, _Policy.W): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(17, 7, 3),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(17, 7, 3),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(17, 4, 3),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(17, 4, 1),
        },
    },
    "ResNet18": {
        (_HardwareGeneration.ALPHA, _Policy.Z): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(520, 171, 0),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(520, 171, 0),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                520, 171, 0
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(520, 171, 0),
        },
        (_HardwareGeneration.ALPHA, _Policy.Y): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(520, 171, 63),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(520, 171, 63),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                520, 171, 63
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(520, 171, 63),
        },
        (_HardwareGeneration.ALPHA, _Policy.X): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(520, 178, 70),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(520, 178, 70),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                520, 182, 67
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(520, 182, 67),
        },
        (_HardwareGeneration.ALPHA, _Policy.W): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(520, 198, 90),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(520, 198, 90),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                520, 176, 60
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(520, 176, 60),
        },
    },
    "ShuffleNetV2": {
        (_HardwareGeneration.ALPHA, _Policy.Z): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(1485, 451, 0),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1485, 432, 0),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1485, 451, 0
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1485, 432, 0),
        },
        (_HardwareGeneration.ALPHA, _Policy.Y): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1485, 451, 171
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1485, 432, 171),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1485, 451, 171
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                1485, 432, 171
            ),
        },
        (_HardwareGeneration.ALPHA, _Policy.X): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1485, 479, 202
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1485, 479, 202),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1485, 453, 187
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                1485, 453, 187
            ),
        },
        (_HardwareGeneration.ALPHA, _Policy.W): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1485, 565, 288
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1485, 565, 288),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1485, 435, 168
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                1485, 435, 168
            ),
        },
    },
    "U-Net": {
        (_HardwareGeneration.ALPHA, _Policy.Z): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(1205, 390, 0),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1205, 386, 0),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1205, 390, 0
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1205, 386, 0),
        },
        (_HardwareGeneration.ALPHA, _Policy.Y): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1205, 390, 141
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1205, 386, 141),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1205, 390, 141
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                1205, 386, 141
            ),
        },
        (_HardwareGeneration.ALPHA, _Policy.X): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1205, 406, 161
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1205, 406, 161),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1205, 418, 158
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                1205, 418, 158
            ),
        },
        (_HardwareGeneration.ALPHA, _Policy.W): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1205, 457, 212
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1205, 457, 212),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1205, 399, 142
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                1205, 399, 138
            ),
        },
    },
    "SmolLM2-135M": {
        (_HardwareGeneration.ALPHA, _Policy.Z): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(4399, 756, 0),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(4399, 756, 0),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                4399, 756, 0
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(4399, 756, 0),
        },
        (_HardwareGeneration.ALPHA, _Policy.Y): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                4399, 756, 750
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(4399, 756, 750),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                4399, 756, 750
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                4399, 756, 750
            ),
        },
        (_HardwareGeneration.ALPHA, _Policy.X): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                4399, 877, 871
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(4399, 877, 871),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                4399, 994, 721
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                4399, 994, 721
            ),
        },
        (_HardwareGeneration.ALPHA, _Policy.W): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                4399, 1449, 1443
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(4399, 1449, 1443),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                4399, 934, 781
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                4399, 934, 781
            ),
        },
    },
    "SigLIP": {
        (_HardwareGeneration.ALPHA, _Policy.Z): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(1524, 420, 0),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1524, 420, 0),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1524, 420, 0
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1524, 420, 0),
        },
        (_HardwareGeneration.ALPHA, _Policy.Y): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1524, 420, 267
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1524, 420, 267),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1524, 420, 267
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                1524, 420, 267
            ),
        },
        (_HardwareGeneration.ALPHA, _Policy.X): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1524, 456, 303
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1524, 456, 303),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1524, 468, 267
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                1524, 468, 267
            ),
        },
        (_HardwareGeneration.ALPHA, _Policy.W): {
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1524, 554, 401
            ),
            (passes.adapt.AssignmentMode.STRICT, passes.adapt.CoalescenceMode.WITH): _AdaptationState(1524, 554, 401),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITHOUT): _AdaptationState(
                1524, 457, 268
            ),
            (passes.adapt.AssignmentMode.PERMISSIVE, passes.adapt.CoalescenceMode.WITH): _AdaptationState(
                1524, 457, 268
            ),
        },
    },
}


def _enum_arg(enum_cls: type[enum.Enum]) -> Callable[[str], enum.Enum]:
    """Build an ``argparse`` ``type`` converter mapping a member name (case-insensitive) to its enum member."""
    by_name = {member.name.lower(): member for member in enum_cls}

    def convert(value: str) -> enum.Enum:
        try:
            return by_name[value]
        except KeyError:
            raise argparse.ArgumentTypeError(f"invalid choice: {value!r} (choose from {', '.join(by_name)})") from None

    return convert


def _enum_metavar(enum_cls: type[enum.Enum]) -> str:
    return "{" + ",".join(member.name.lower() for member in enum_cls) + "}"


def _create_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", "-n", choices=_MODEL_NAME_TO_FACTORY, required=True)
    parser.add_argument(
        "--hardware-generation",
        "-g",
        type=_enum_arg(_HardwareGeneration),
        metavar=_enum_metavar(_HardwareGeneration),
        required=True,
    )
    parser.add_argument("--policy", "-p", type=_enum_arg(_Policy), metavar=_enum_metavar(_Policy), required=True)
    parser.add_argument(
        "--assignment-mode",
        "-a",
        type=_enum_arg(passes.adapt.AssignmentMode),
        metavar=_enum_metavar(passes.adapt.AssignmentMode),
        required=True,
    )
    parser.add_argument(
        "--coalescence-mode",
        "-c",
        type=_enum_arg(passes.adapt.CoalescenceMode),
        metavar=_enum_metavar(passes.adapt.CoalescenceMode),
        required=True,
    )
    parser.add_argument("--interactive", "-i", action="store_true")
    return parser


def main() -> None:
    args = _create_args().parse_args()
    graph_module, input = _create_graph_module_and_input(args.model_name)

    # configure the model compressor
    setup = (args.hardware_generation, args.policy)
    config_factory, picker = _SETUP_TO_CONFIG_AND_PICKER[setup]
    config = config_factory()
    config.annotation.input = input
    config.annotation.picker = picker
    config.adaptation.assignment = args.assignment_mode
    config.adaptation.coalescence = args.coalescence_mode
    # perform pre-optimization steps
    graph_module_ = api.prepare_before_optimization(graph_module, config)

    key = (
        args.model_name,
        args.hardware_generation.name.lower(),
        args.policy.name,
        args.assignment_mode.name.lower(),
        args.coalescence_mode.name.lower(),
    )
    storage, traffic, codecs = cost.analyze_model_cost(graph_module_, input)
    actual_cost = {
        "storage_bytes": storage.bytes,
        "traffic_bytes": traffic.bytes,
        "codec_numel": codecs.count,
        "n_reads": len(traffic.reads),
        "n_codec_ops": len(codecs.ops),
        "n_params": len(storage.parameters),
    }
    expected_cost = _KEY_TO_COST[key]
    assert actual_cost == expected_cost

    # `import_exported_program` returns an ExportedProgram module that pytree-unflattens the single
    # output back to a Tensor, whereas the prepared graph returns the raw 1-tuple; normalize both to
    # the output tensor before comparing. (A bare ``[0]`` indexes batch row 0 of the Tensor, which
    # only coincides with the tuple's single element when the batch size is 1.)
    def _output_tensor(module: fx.GraphModule) -> torch.Tensor:
        output = module(**input)
        return output if isinstance(output, torch.Tensor) else output[0]

    assert torch.equal(_output_tensor(graph_module_).flatten(), _output_tensor(graph_module).flatten())
    semantics_to_subgraph: dict[base.fx.Semantics, base.fx.Subgraph] = {}
    for node in graph_module_.graph.nodes:
        if not base.fx.has_kwarg_to_subgraph(node):
            continue
        kwarg_to_subgraph = base.fx.get_kwarg_to_subgraph(node)
        semantics_to_subgraph.update({subgraph.semantics: subgraph for subgraph in kwarg_to_subgraph.values()})
    n_true = len([semantics for semantics in semantics_to_subgraph if semantics.is_true])
    n_fake = len([semantics for semantics in semantics_to_subgraph if semantics.is_fake])
    actual_state = _AdaptationState(len(graph_module_.graph.nodes), n_true, n_fake)
    expected_state = _MODEL_NAME_TO_SETUP_TO_MODE_TO_ADAPTATION_STATE[args.model_name][setup][
        (args.assignment_mode, args.coalescence_mode)
    ]
    assert actual_state == expected_state
    # TODO
    if args.interactive:
        IPython.embed()
    # perform post-optimization steps
    graph_module_ = api.finalize_after_optimization(graph_module, config)


if __name__ == "__main__":
    main()

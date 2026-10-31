# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Define marker operators.

Marker operators should not impact the latency and storage requirements of adapted models.
Therefore, they are defined as aliasing operators: no additional tensor storage is required, and
negligible latency increases are expected.

The ``torch.library.custom_op`` API does not support defining aliasing operators, therefore marker
operators are registered via the ``torch.library.Library`` API.

"""

from model_compressor.passes.adapt.markers.mark_input_codec import (
    mark_input_codec_int,
    mark_input_codec_tensor,
)
from model_compressor.passes.adapt.markers.mark_output_codec import (
    mark_output_codec_int,
    mark_output_codec_tensor,
)

MARKERS_INPUT_CODEC = frozenset({mark_input_codec_tensor, mark_input_codec_int})

MARKERS_OUTPUT_CODEC = frozenset({mark_output_codec_tensor, mark_output_codec_int})

MARKERS_CODEC = MARKERS_INPUT_CODEC | MARKERS_OUTPUT_CODEC

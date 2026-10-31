# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import torch

# tensor overload
_lib = torch.library.Library("markers", "FRAGMENT")
_lib.define("mark_output_codec.Tensor(Tensor(a) input) -> Tensor(a)")
_lib.impl("mark_output_codec.Tensor", lambda input: torch.ops.aten.alias.default(input), "CompositeExplicitAutograd")
_lib.impl("mark_output_codec.Tensor", lambda input: torch.ops.aten.alias.default(input), "Meta")
_lib.impl("mark_output_codec.Tensor", torch.library.fallthrough_kernel, "Autograd")


mark_output_codec_tensor = torch.ops.markers.mark_output_codec.Tensor
"""Mark an output codec operator.

Args:
    input: The tensor to be encoded/decoded.

Returns:
    The encoded/decoded tensor.

"""


# integer overload
# The signature is ``(SymInt input) -> SymInt`` as opposed to ``(int input) -> int`` so that:
# * the marker can survive  re-``torch.export.export`` cycles with dynamic shapes;
# * it is compatible with non-dynamic integers.
_lib.define("mark_output_codec.int(SymInt input) -> SymInt")
_lib.impl("mark_output_codec.int", lambda input: input, "CompositeExplicitAutograd")
_lib.impl("mark_output_codec.int", lambda input: input, "Meta")


mark_output_codec_int = torch.ops.markers.mark_output_codec.int
"""Mark an output codec operator.

Args:
    input: The integer to be encoded/decoded.

Returns:
    The encoded/decoded integer.

"""

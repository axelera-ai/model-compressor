# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

import dataclasses

from typing_extensions import Self

import model_compressor.passes.annotate.base as annotate


@dataclasses.dataclass(frozen=True, eq=False, unsafe_hash=True)
class Semantics:
    """A data semantics.

    Args:
        manifest: The data's type in the source model.
        latent: The data's type in the optimized model.

    """

    manifest: annotate.DType
    latent: annotate.DType

    @property
    def is_true(self) -> bool:
        """Whether the manifest (source) and latent (optimized) data type agree."""
        return self.manifest == self.latent

    @property
    def is_fake(self) -> bool:
        """Whether the manifest (source) and latent (optimized) data type disagree."""
        return self.manifest != self.latent

    def is_compatible(self, other: Self) -> bool:
        """Verify whether this and another semantics are compatible."""
        # Invariant: if two semantics are compatible, they are either both true or both fake.
        return (self.manifest == other.manifest) and (self.latent == other.latent)

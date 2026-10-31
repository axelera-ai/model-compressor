# Copyright (c) 2026 Axelera AI
# SPDX-License-Identifier: MIT

"""Standalone SmolVLM2 towers (SmolLM2-135M language + SigLIP vision), traced directly from
HuggingFace.

The language tower's RoPE / RMSNorm / attention are decomposed by ``torch.export``; the passes in
`models.refuse` recover them into the demo's ``xaten`` vocabulary. The vision tower (SigLIP: LayerNorm
+ SDPA, no RoPE/cache) needs no re-fusion.
"""

from typing import Any

import model_compressor.api as demo_api
import torch
import torch.fx as fx
import transformers as hf
from models import refuse
from models.common import export

_TAG = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"


def _from_huggingface() -> Any:
    return hf.AutoModelForImageTextToText.from_pretrained(_TAG, dtype=torch.float32)


# --------------------------------------------------------------------------------------------------
# language tower: SmolLM2-135M (a Llama decoder stack)
# --------------------------------------------------------------------------------------------------

_LANG_BATCH, _LANG_SEQ = 1, 128


class _SmolLM2Backbone(torch.nn.Module):
    """A Llama decoder stack with cos/sin and a 4D additive mask injected as forward inputs.

    Injecting cos/sin (computed eagerly outside) bypasses ``LlamaModel.rotary_emb`` and its
    vmap/autocast HOP plumbing, leaving a clean graph whose only RoPE trace is the per-layer
    ``apply_rotary_pos_emb`` we re-fuse.
    """

    def __init__(self, text_model: torch.nn.Module):
        super().__init__()
        self.layers = text_model.layers
        self.norm = text_model.norm

    def forward(self, inputs_embeds, attention_mask, cos, sin):
        hidden = inputs_embeds
        position_embeddings = (cos, sin)
        for layer in self.layers:
            out = layer(hidden, attention_mask=attention_mask, position_embeddings=position_embeddings, use_cache=False)
            hidden = out[0] if isinstance(out, tuple) else out
        return self.norm(hidden)


def _causal_additive_mask(batch: int, seq: int, dtype: torch.dtype) -> torch.Tensor:
    m = torch.full((seq, seq), torch.finfo(dtype).min, dtype=dtype)
    m = torch.triu(m, diagonal=1)  # 0 on/below the diagonal, -inf above
    return m.view(1, 1, seq, seq).expand(batch, 1, seq, seq).contiguous()


def create_smolvlm2_language() -> tuple[fx.GraphModule, dict[str, Any]]:
    text_model = _from_huggingface().model.text_model
    text_model.eval()
    text_model.config._attn_implementation = "sdpa"

    inputs_embeds = torch.randn(_LANG_BATCH, _LANG_SEQ, text_model.config.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(_LANG_SEQ, dtype=torch.int64).unsqueeze(0).expand(_LANG_BATCH, -1)
    with torch.no_grad():
        cos, sin = text_model.rotary_emb(inputs_embeds, position_ids)  # full-width, computed eagerly
    input = {
        "inputs_embeds": inputs_embeds,
        "attention_mask": _causal_additive_mask(_LANG_BATCH, _LANG_SEQ, inputs_embeds.dtype),
        "cos": cos,
        "sin": sin,
    }

    wrapper = _SmolLM2Backbone(text_model).eval()
    with torch.no_grad():
        graph_module = torch.export.export(wrapper, args=(), kwargs=input).module()
    # recover the xaten vocabulary from the decomposed graph (uses node.meta provenance)
    refuse.refuse_rope(graph_module)
    refuse.refuse_rms_norm(graph_module)
    refuse.inject_kv_cache(graph_module)
    # re-export + import to normalize the graph and repopulate node.meta on the inserted ops
    with torch.no_grad():
        exported_program = torch.export.export(graph_module, args=(), kwargs=input)
    graph_module = demo_api.import_exported_program(exported_program)
    # massage HF-dialect view plumbing (aten.alias / aten.unsqueeze) into the demo's vocabulary
    refuse.canonicalize_views(graph_module)
    return graph_module, input


# --------------------------------------------------------------------------------------------------
# vision tower: SigLIP (LayerNorm + SDPA, no RoPE / cache -> no re-fusion needed)
# --------------------------------------------------------------------------------------------------

_VISION_BATCH = 1


class _SigLIPBackbone(torch.nn.Module):
    """SigLIP vision tower with ``position_ids`` and a pre-built 4D attention mask injected as inputs.

    The stock ``SmolVLMVisionTransformer`` derives per-patch position ids from ``patch_attention_mask``
    (a ``bucketize`` + boolean-index loop) and builds the attention mask via
    ``_prepare_4d_attention_mask``; both emit data-dependent / comparison ops (e.g. ``aten.ge``) that
    the demo's evaluator rejects as illegal targets. For a single, unpadded image every patch is
    valid, so these are constants: we precompute the position ids and a full-attention 4D additive
    mask eagerly and feed them in -- a ``(pixel_values, position_ids, attention_mask)`` signature.
    Only the patch embedding, the position
    embedding, the encoder and the post-layernorm remain in the traced graph.
    """

    def __init__(self, vision_model: torch.nn.Module):
        super().__init__()
        self.patch_embedding = vision_model.embeddings.patch_embedding
        self.position_embedding = vision_model.embeddings.position_embedding
        self.encoder = vision_model.encoder
        self.post_layernorm = vision_model.post_layernorm

    def forward(self, pixel_values, position_ids, attention_mask):
        patch_embeds = self.patch_embedding(pixel_values)
        embeddings = patch_embeds.flatten(2).transpose(1, 2)
        embeddings = embeddings + self.position_embedding(position_ids)
        out = self.encoder(inputs_embeds=embeddings, attention_mask=attention_mask)
        return self.post_layernorm(out.last_hidden_state)


def _full_image_position_ids(vision_model: torch.nn.Module, n: int) -> torch.Tensor:
    """The per-patch position ids for a full (unpadded) ``n x n``-patch image.

    Replicates ``SmolVLMVisionEmbeddings.forward``'s fractional-coordinate bucketization for the case
    where every patch is valid (so ``nb_patches_h == nb_patches_w == n``); the result is a constant,
    computed here eagerly so the traced graph carries none of that data-dependent logic.
    """
    per_side = vision_model.embeddings.num_patches_per_side
    boundaries = torch.arange(1 / per_side, 1.0, 1 / per_side)
    fractional_coords = torch.arange(n, dtype=torch.float32) / n * (1 - 1e-6)
    buckets = torch.bucketize(fractional_coords, boundaries, right=True)
    return (buckets[:, None] * per_side + buckets).flatten().unsqueeze(0)  # (1, n * n)


def create_smolvlm2_vision() -> tuple[fx.GraphModule, dict[str, Any]]:
    vision_model = _from_huggingface().model.vision_model
    vision_model.eval()
    vision_model.config._attn_implementation = "sdpa"

    image = vision_model.config.image_size
    patch = vision_model.config.patch_size
    n = image // patch
    seq = n * n
    input = {
        "pixel_values": torch.randn(_VISION_BATCH, vision_model.config.num_channels, image, image, dtype=torch.float32),
        "position_ids": _full_image_position_ids(vision_model, n),
        # full-attention additive mask (every patch valid) -- what _prepare_4d_attention_mask would build
        "attention_mask": torch.zeros(_VISION_BATCH, 1, seq, seq, dtype=torch.float32),
    }
    graph_module, input = export(_SigLIPBackbone(vision_model).eval(), input)
    # massage HF-dialect view plumbing (aten.alias from the SDPA path) into the demo's vocabulary
    refuse.canonicalize_views(graph_module)
    return graph_module, input

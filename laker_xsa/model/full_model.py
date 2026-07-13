"""Full encoder Transformer with XSA + LAKER attention.

This module provides :class:`XSALAKERTransformer`, a configurable
Transformer encoder built by stacking
:class:`laker_xsa.model.transformer_block.XSALAKERTransformerBlock`.
The model can be constructed in two modes:

* **Embedding mode** (when ``vocab_size`` is provided) - a token
  embedding, a learned positional embedding, and a separate (untied)
  bias-free output projection to the vocabulary are added. Inputs are
  integer token IDs.
* **Embedding-free mode** (when ``vocab_size`` is ``None``) - the
  model is a pure encoder; inputs are pre-computed embeddings of
  shape ``(batch, seq_len, d_model)``.

Both modes share the same stack of Transformer blocks and the same
final LayerNorm. Selecting between them is done at construction time
and cannot be changed afterwards.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
from torch import nn

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.model.transformer_block import XSALAKERTransformerBlock


class XSALAKERTransformer(nn.Module):
    """Stacked Transformer encoder with optional token I/O.

    The architecture is:

    .. code-block:: text

        Input IDs -> Token Embedding + Position Embedding
                   -> [Transformer Block] x num_layers
                   -> LayerNorm
                   -> Output Projection (when vocab_size is set)

    When ``vocab_size`` is ``None`` the embedding and output
    projection are omitted; the model is a pure encoder that operates
    on pre-computed embeddings of shape ``(batch, seq_len, d_model)``.

    The same :class:`XSA_LAKER_Config` instance is passed to every block, and
    each selected attention implementation reads the fields relevant to it.
    ``attention_type`` is fixed at construction rather than selected per
    forward pass.

    Note:
        :attr:`XSA_LAKER_Config.use_fused` is not consulted. The
        ``attention_type`` constructor argument selects every block's
        attention implementation.

    Attributes:
        config: The shared :class:`XSA_LAKER_Config`.
        token_embedding: Token embedding of shape
            ``(vocab_size, d_model)``. ``None`` in embedding-free
            mode.
        pos_embedding: Learned positional embedding of shape
            ``(max_seq_len, d_model)``. ``None`` in embedding-free
            mode.
        token_embedding_weight: A zero-sized placeholder buffer
            (``torch.empty(0, 0)``) registered only when ``vocab_size``
            is ``None``. It carries no parameters and no data and is not
            read by the model; callers should not rely on it.
        blocks: ``nn.ModuleList`` of
            :class:`XSALAKERTransformerBlock` instances. Their
            concrete attention type is the one passed to
            ``__init__``.
        final_norm: Final :class:`nn.LayerNorm` applied after the
            last block.
        output_proj: Linear projection from ``d_model`` to
            ``vocab_size``. ``None`` in embedding-free mode. The
            projection is bias-free and is a distinct parameter matrix -
            its weights are **not** tied to ``token_embedding``.

    Tensor Shapes:
        * Embedding-mode input: ``(batch, seq_len)`` integer IDs.
        * Embedding-mode output: ``(batch, seq_len, vocab_size)``
          logits.
        * Embedding-free input: ``(batch, seq_len, d_model)``.
        * Embedding-free output: ``(batch, seq_len, d_model)``.
    """

    def __init__(
        self,
        config: XSA_LAKER_Config,
        num_layers: int = 6,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        vocab_size: Optional[int] = None,
        max_seq_len: int = 512,
        attention_type: Literal[
            "standard", "xsa", "kernel", "fused", "fused_v2"
        ] = "fused_v2",
    ) -> None:
        """Initialize the Transformer encoder.

        Args:
            config: Shared attention configuration. Forwarded to
                every block. ``d_model`` and ``eps`` are also used by
                the final LayerNorm.
            num_layers: Number of stacked Transformer blocks. More
                blocks give a deeper model at roughly linear cost in
                parameters and forward time.
            d_ff: Hidden dimension of each block's MLP. ``None``
                (default) selects ``4 * d_model`` per block.
            dropout: Dropout probability for both the attention
                output and the MLP output of every block.
            vocab_size: Vocabulary size. When provided, the model
                adds a token embedding, a learned positional
                embedding of length ``max_seq_len``, and a
                bias-free output projection. When ``None``, the
                model is constructed in embedding-free mode and
                expects pre-computed embeddings as input.
            max_seq_len: Maximum supported sequence length. Only used
                in embedding mode, to size the learned positional
                embedding; it is ignored in embedding-free mode (no
                positional embedding is created). In embedding mode,
                input sequences longer than ``max_seq_len`` index the
                positional embedding out of bounds and raise
                ``IndexError``.
            attention_type: Attention module used by every block. See
                :class:`XSALAKERTransformerBlock` for the full
                enumeration. The default ``"fused_v2"`` selects
                :class:`laker_xsa.attention.LakerAttention`.
        """
        super().__init__()
        self.config = config

        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, config.d_model)
            self.pos_embedding = nn.Embedding(max_seq_len, config.d_model)
        else:
            self.register_buffer("token_embedding_weight", torch.empty(0, 0))
            self.token_embedding = None
            self.pos_embedding = None

        self.blocks = nn.ModuleList(
            [
                XSALAKERTransformerBlock(
                    config,
                    d_ff=d_ff,
                    dropout=dropout,
                    attention_type=attention_type,
                )
                for _ in range(num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.d_model, eps=config.eps)

        if vocab_size is not None:
            self.output_proj = nn.Linear(config.d_model, vocab_size, bias=False)
        else:
            self.output_proj = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the encoder.

        Args:
            x: Input tensor. In embedding mode this is a
                ``(batch, seq_len)`` integer tensor of token IDs; in
                embedding-free mode it is a
                ``(batch, seq_len, d_model)`` float tensor.
            mask: Optional attention mask forwarded to every block.
                Not interpreted by the model itself.

        Returns:
            Output tensor. In embedding mode this is a
            ``(batch, seq_len, vocab_size)`` logits tensor; in
            embedding-free mode it is a
            ``(batch, seq_len, d_model)`` float tensor.

        Raises:
            ValueError: If the model is in embedding mode and ``x``
                is not 2D.
            IndexError: In embedding mode, if any token ID is ``>=
                vocab_size`` (or negative), or if ``seq_len`` exceeds
                the positional embedding's ``max_seq_len``.
            RuntimeError: If ``token_embedding`` is set but
                ``pos_embedding`` is ``None``, if token IDs have an unsupported
                dtype, or if a downstream block rejects an incompatible input,
                mask, dtype, or device. In normal construction the two
                embedding modules are created together.
        """
        if self.token_embedding is not None:
            if x.dim() != 2:
                raise ValueError(
                    f"Expected 2D token IDs (batch, seq_len), got shape {x.shape}"
                )
            batch, seq_len = x.shape
            positions = (
                torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch, -1)
            )
            if self.pos_embedding is None:
                raise RuntimeError(
                    "pos_embedding must be set when token_embedding is used"
                )
            x = self.token_embedding(x) + self.pos_embedding(positions)

        for block in self.blocks:
            x = block(x, mask)

        x = self.final_norm(x)

        if self.output_proj is not None:
            x = self.output_proj(x)

        return x

    token_embedding: Optional[nn.Embedding]
    pos_embedding: Optional[nn.Embedding]
    output_proj: Optional[nn.Linear]

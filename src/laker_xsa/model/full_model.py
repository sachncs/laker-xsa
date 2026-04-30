"""
Full Transformer model with XSA + LAKER attention.

This module provides the complete Transformer model architecture using
stacked XSALAKERTransformerBlocks.
"""

from __future__ import annotations

from typing import Literal, Optional

import torch
import torch.nn as nn

from laker_xsa.config import XSA_LAKER_Config
from laker_xsa.model.transformer_block import XSALAKERTransformerBlock


class XSALAKERTransformer(nn.Module):
    """
    Complete Transformer model with XSA + LAKER attention.

    Architecture:

    .. code-block:: text

        Input IDs -> Token Embedding + Position Embedding
                   -> [Transformer Block] x num_layers
                   -> LayerNorm
                   -> Output Projection (if vocab_size specified)

    Attributes:
        config: Configuration object.
        token_embedding: Token embedding layer (if vocab_size specified).
        pos_embedding: Position embedding layer.
        blocks: List of Transformer blocks.
        final_norm: Final layer normalization.
        output_proj: Output projection to vocabulary (if vocab_size specified).

    Input Shape:
        - With vocab_size: ``(batch, seq_len)`` (token IDs)
        - Without vocab_size: ``(batch, seq_len, d_model)`` (embeddings)

    Output Shape:
        - With vocab_size: ``(batch, seq_len, vocab_size)`` (logits)
        - Without vocab_size: ``(batch, seq_len, d_model)`` (embeddings)
    """

    def __init__(
        self,
        config: XSA_LAKER_Config,
        num_layers: int = 6,
        d_ff: Optional[int] = None,
        dropout: float = 0.0,
        vocab_size: Optional[int] = None,
        max_seq_len: int = 512,
        attention_type: Literal["standard", "xsa", "kernel", "fused"] = "fused",
    ) -> None:
        """
        Initialize Transformer model.

        Args:
            config: Configuration object.
            num_layers: Number of Transformer blocks.
            d_ff: Feed-forward dimension. Defaults to 4 * d_model.
            dropout: Dropout probability.
            vocab_size: Vocabulary size. If None, no embedding layer.
            max_seq_len: Maximum sequence length for position embeddings.
            attention_type: Type of attention for all blocks.
        """
        super().__init__()
        self.config = config

        # Token and position embeddings
        if vocab_size is not None:
            self.token_embedding = nn.Embedding(vocab_size, config.d_model)
            self.pos_embedding = nn.Embedding(max_seq_len, config.d_model)
        else:
            self.register_buffer("token_embedding_weight", torch.empty(0, 0))
            self.token_embedding = None
            self.pos_embedding = None

        # Transformer blocks
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

        # Final normalization
        self.final_norm = nn.LayerNorm(config.d_model, eps=config.eps)

        # Output projection
        if vocab_size is not None:
            self.output_proj = nn.Linear(config.d_model, vocab_size, bias=False)
        else:
            self.output_proj = None

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for Transformer model.

        Args:
            x: Input tensor.
               - If vocab_size specified: token IDs, shape ``(batch, seq_len)``.
               - Otherwise: embeddings, shape ``(batch, seq_len, d_model)``.
            mask: Optional attention mask.

        Returns:
            Output tensor.
            - If vocab_size: logits, shape ``(batch, seq_len, vocab_size)``.
            - Otherwise: embeddings, shape ``(batch, seq_len, d_model)``.
        """
        # Embed input if token embedding exists
        if self.token_embedding is not None:
            batch, seq_len = x.shape
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(
                batch, -1
            )
            x = self.token_embedding(x) + self.pos_embedding(positions)

        # Apply Transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        # Final normalization
        x = self.final_norm(x)

        # Output projection if specified
        if self.output_proj is not None:
            x = self.output_proj(x)

        return x

import math
from typing import List

import torch
import torch.nn as nn

from contextual_gate_summarizer import ContextualGateLayer


CHUNK_SIZE = 30  # candidates per local chunk — well inside the range
                  # the plain gate was shown (via the capped-inference
                  # control experiment) to still handle reasonably.


class HierarchicalGateRanker(nn.Module):
    """
    Hierarchical local-gate + global-attention sentence ranker.

    Motivation: the flat ContextualGateRanker's single hidden state
    has to represent the entire document, and the capped-inference
    control experiment showed this degrades even within its trained
    sequence length — pointing to an architectural limitation, not
    just insufficient exposure to long sequences.

    This design bounds what the gate is responsible for:

    1. Local stage: the document is split into fixed-size chunks
       (CHUNK_SIZE candidates each). The existing ContextualGateLayer
       runs independently within each chunk, so no single hidden
       state ever has to carry information further than CHUNK_SIZE
       steps.

    2. Global stage: each chunk is mean-pooled into one vector, and a
       small multi-head self-attention layer runs across those
       chunk-level vectors. This is cheap — attention over ~5-20
       chunks, not hundreds of sentences — and lets distant chunks
       inform each other without relying on sequential state decay.

    3. Each candidate's final score is produced from the concatenation
       of its local (in-chunk) gated representation and its chunk's
       globally-attended representation.

    This is a genuine architectural change targeting the diagnosed
    cause of failure, not a bigger/slower version of the same idea —
    but whether it actually improves ROUGE is an open empirical
    question this script is designed to test, not assume.
    """

    def __init__(
        self,
        input_dim: int = 384,
        hidden_dim: int = 128,
        chunk_size: int = CHUNK_SIZE,
        attention_heads: int = 4,
    ):
        super().__init__()

        self.chunk_size = chunk_size

        self.local_gate = ContextualGateLayer(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
        )

        self.global_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            batch_first=True,
        )

        self.attention_norm = nn.LayerNorm(hidden_dim)

        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        embeddings: [batch_size=1, sequence_length, input_dim]
        returns: [batch_size=1, sequence_length] raw logits

        Note: batch_size is assumed to be 1, matching the rest of this
        project's per-document processing.
        """

        batch_size, sequence_length, _ = embeddings.shape
        chunk_size = self.chunk_size

        num_chunks = math.ceil(sequence_length / chunk_size)

        local_outputs = []
        chunk_representations = []
        chunk_lengths = []

        for chunk_index in range(num_chunks):
            start = chunk_index * chunk_size
            end = min(start + chunk_size, sequence_length)

            chunk_embeddings = embeddings[:, start:end, :]

            gated_chunk = self.local_gate(chunk_embeddings)
            local_outputs.append(gated_chunk)

            chunk_representations.append(gated_chunk.mean(dim=1, keepdim=True))
            chunk_lengths.append(end - start)

        local_concat = torch.cat(local_outputs, dim=1)
        chunk_stack = torch.cat(chunk_representations, dim=1)

        attended_chunks, _ = self.global_attention(
            chunk_stack, chunk_stack, chunk_stack
        )
        attended_chunks = self.attention_norm(attended_chunks + chunk_stack)

        broadcasted_global = []

        for chunk_index, length in enumerate(chunk_lengths):
            global_vector = attended_chunks[:, chunk_index : chunk_index + 1, :]
            broadcasted_global.append(global_vector.expand(-1, length, -1))

        global_concat = torch.cat(broadcasted_global, dim=1)

        combined = torch.cat([local_concat, global_concat], dim=-1)
        raw_scores = self.scorer(combined).squeeze(-1)

        return raw_scores

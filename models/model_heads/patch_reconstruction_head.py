import torch
import torch.nn as nn


class PatchReconstructionHead(nn.Module):
    """Linear decoder mapping each token embedding back to its waveform patch.

    This is the SimMIM-style decoder used for pre-training: every token is projected
    independently, with no decoder transformer and no re-ordering of the sequence.
    """

    def __init__(self, embed_dim: int = 200, patch_size: int = 200):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.decoder_pred = nn.Linear(embed_dim, patch_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct waveform patches.

        Args:
            x: Token embeddings of shape ``(batch, num_tokens, embed_dim)``.

        Returns:
            Reconstructed patches of shape ``(batch, num_tokens, patch_size)``.
        """
        return self.decoder_pred(x)

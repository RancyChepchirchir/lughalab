import torch
from torch import nn


class LandmarkTransformer(nn.Module):
    """
    Reconstructed LandmarkTransformer for the
    AfriSignEncoder KSL baseline.

    Expected input:
        (batch, 64, 225)
    """

    def __init__(
        self,
        input_dim: int = 225,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        num_classes: int = 4,
        max_frames: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        norm_first: bool = False,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.max_frames = max_frames
        self.activation = activation
        self.norm_first = norm_first

        self.proj = nn.Linear(
            input_dim,
            d_model,
        )

        self.cls = nn.Parameter(
            torch.zeros(
                1,
                1,
                d_model,
            )
        )

        self.pos = nn.Embedding(
            max_frames + 1,
            d_model,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=norm_first,
        )

        self.enc = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.norm = nn.LayerNorm(
            d_model
        )

        self.head = nn.Linear(
            d_model,
            num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(
                "Expected input shape "
                "(batch, frames, features)."
            )

        batch_size, frames, features = (
            x.shape
        )

        if features != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim} "
                f"features, got {features}."
            )

        if frames > self.max_frames:
            raise ValueError(
                f"Maximum frame length is "
                f"{self.max_frames}, got {frames}."
            )

        x = self.proj(
            x
        )

        cls = self.cls.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat(
            [
                cls,
                x,
            ],
            dim=1,
        )

        positions = torch.arange(
            x.shape[1],
            device=x.device,
        )

        x = (
            x
            + self.pos(
                positions
            ).unsqueeze(0)
        )

        x = self.enc(
            x
        )

        cls_output = x[
            :,
            0,
        ]

        cls_output = self.norm(
            cls_output
        )

        return self.head(
            cls_output
        )
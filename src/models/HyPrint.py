import torch
from torch import nn

from torchvision.models import vision_transformer
from src.models.DeepFinger import DeepFingerTrainingOutput
from src.models.DeepFinger import DeepFingerOutput

class ViTB32Pretrained(nn.Module):
    def __init__(self, representation_size: int, num_classes: int):
        super().__init__()
        self.model = vision_transformer.vit_b_32(
            representation_size=representation_size,
            num_classes=num_classes
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.expand(-1, 3, -1, -1)
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        x = x[:, 0]
        x = self.model.heads[0](x) # Pre logits
        x = self.model.heads[1](x) # Tanh
        if self.training:
            return DeepFingerTrainingOutput(
                texture_embeddings=x,
                texture_logits=self.model.heads[2](x), # Logits
            )
        return DeepFingerOutput(texture_embeddings=x)
        
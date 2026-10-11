import torch
from torch import nn
from torch.nn import functional as F
import timm


class UnoFace(nn.Module):
    def __init__(
        self, embedding_dim=512, backbone="resnet50", pretrained=False, output_bn=False
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0
        )
        self.projection = nn.Linear(
            self.backbone.num_features, embedding_dim, bias=False
        )
        if output_bn:
            self.features = nn.BatchNorm1d(embedding_dim, eps=1e-5)
            self.features.weight.requires_grad_(False)
        self.embedding_dim = embedding_dim

    def forward(self, x):
        z = self.projection(self.backbone(x)).float()
        if hasattr(self, "features"):
            z = self.features(z)
        return F.normalize(z, dim=-1)


def load_reference(path, device="cuda"):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not all(key in ckpt for key in ("model_config", "encoder")):
        raise ValueError("UnoFace checkpoint requires model_config and encoder")
    cfg = dict(ckpt["model_config"])
    cfg["pretrained"] = False
    model = UnoFace(**cfg)
    model.load_state_dict(ckpt["encoder"], strict=True)
    return model.to(device).eval().requires_grad_(False), ckpt

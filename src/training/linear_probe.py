from __future__ import annotations

from src.models.clip_setup import CLIPBinaryClassifier, freeze_clip_encoder
from src.training.common import evaluate_classifier_model, train_classifier_model


def run_linear_probe(clip_model, loaders, cfg, device):
    freeze_clip_encoder(clip_model)
    model = CLIPBinaryClassifier(clip_model, num_classes=cfg["model"]["num_classes"])

    history, timing = train_classifier_model(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        device=device,
        epochs=cfg["train"]["epochs"],
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
        max_grad_norm=float(cfg["train"]["max_grad_norm"]),
    )
    test_metrics = evaluate_classifier_model(model, loaders["test"], device)
    return model, history, timing, test_metrics

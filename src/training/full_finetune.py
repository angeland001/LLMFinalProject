from __future__ import annotations

from src.models.clip_setup import CLIPBinaryClassifier
from src.training.common import evaluate_classifier_model, train_classifier_model


def run_full_finetune(clip_model, loaders, cfg, device):
    for p in clip_model.vision_model.parameters():
        p.requires_grad = True

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

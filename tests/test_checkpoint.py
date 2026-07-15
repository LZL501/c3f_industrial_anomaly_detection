from __future__ import annotations

import torch

from c3f.engine import load_training_checkpoint, save_checkpoint
from c3f.models.losses import C3FLoss


def test_checkpoint_restores_training_state(tmp_path) -> None:
    model = torch.nn.Linear(3, 2)
    criterion = C3FLoss(perceptual_weight=0.0)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    discriminator_optimizer = torch.optim.Adam(
        criterion.discriminator.parameters(), lr=1e-3
    )
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    expected_weight = model.weight.detach().clone()
    path = tmp_path / "last.pth"

    save_checkpoint(
        path,
        model,
        criterion,
        model_optimizer,
        discriminator_optimizer,
        scaler,
        epoch=7,
        global_step=123,
        best_metric=1.5,
        config={"train": {"epochs": 10}},
        metrics={"metric": 1.4},
    )
    with torch.no_grad():
        model.weight.zero_()

    checkpoint = load_training_checkpoint(
        path,
        model,
        criterion,
        model_optimizer,
        discriminator_optimizer,
        scaler,
    )

    torch.testing.assert_close(model.weight, expected_weight)
    assert checkpoint["format_version"] == 2
    assert checkpoint["epoch"] == 7
    assert checkpoint["global_step"] == 123
    assert checkpoint["best_metric"] == 1.5

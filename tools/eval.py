from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3f.config import load_config, merge_overrides, parse_dotlist
from c3f.data import build_eval_dataloader
from c3f.engine import _eval_settings, build_model, evaluate, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate C3F.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--category")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config)
    cli_overrides = parse_dotlist(args.overrides)
    if args.data_root:
        cli_overrides["data.root"] = args.data_root
    if args.category:
        cli_overrides["data.category"] = args.category
    config = merge_overrides(config, cli_overrides)
    device = torch.device(args.device)
    val_loader = build_eval_dataloader(config)
    model = build_model(config, device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    metrics = evaluate(model, val_loader, device, **_eval_settings(config))
    print(metrics)


if __name__ == "__main__":
    main()

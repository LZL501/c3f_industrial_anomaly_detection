from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c3f.config import load_config, merge_overrides, parse_dotlist
from c3f.engine import train
from c3f.utils import ensure_dir, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Train C3F.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root")
    parser.add_argument("--texture-root")
    parser.add_argument("--category")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--resume",
        help="Resume model, optimizers, scaler, epoch, and step from a .pth checkpoint.",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = load_config(args.config)
    cli_overrides = parse_dotlist(args.overrides)
    if args.data_root:
        cli_overrides["data.root"] = args.data_root
    if args.texture_root:
        cli_overrides["data.texture_root"] = args.texture_root
    if args.category:
        cli_overrides["data.category"] = args.category
    config = merge_overrides(config, cli_overrides)

    seed_everything(config["experiment"].get("seed", 1999))
    run_name = f"{config['experiment']['name']}_{config['data']['category']}"
    run_dir = ensure_dir(
        Path(config["experiment"].get("output_dir", "runs")) / run_name
    )
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    train(config, torch.device(args.device), run_dir, resume_path=args.resume)


if __name__ == "__main__":
    main()

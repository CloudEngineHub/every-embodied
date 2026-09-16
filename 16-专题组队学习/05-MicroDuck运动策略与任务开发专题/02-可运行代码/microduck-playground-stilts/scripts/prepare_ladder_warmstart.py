"""Build a V3 ladder checkpoint with a pretrained flat-walking actor prefix.

The first 61 actor inputs are the shared proprioceptive/command layout used by
the flat velocity task. V3 appends height, foothold, phase, contact, and stage
observations after that prefix. Only the compatible actor prefix and its
normalizer statistics are copied; the V3 ladder-specific columns and critic
remain from the source checkpoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


SHARED_ACTOR_DIM = 61


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ladder-checkpoint", type=Path, required=True)
    parser.add_argument("--walking-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--walking-blend",
        type=float,
        default=1.0,
        help="fraction of the walking prefix to blend into the V3 prefix",
    )
    args = parser.parse_args()
    if not 0.0 <= args.walking_blend <= 1.0:
        raise ValueError("--walking-blend must be in [0, 1]")

    ladder = torch.load(args.ladder_checkpoint, map_location="cpu", weights_only=False)
    walking = torch.load(args.walking_checkpoint, map_location="cpu", weights_only=False)
    actor = ladder["actor_state_dict"]
    walking_actor = walking["actor_state_dict"]

    walking_input = walking_actor["mlp.0.weight"]
    ladder_input = actor["mlp.0.weight"]
    if walking_input.shape[1] != SHARED_ACTOR_DIM:
        raise ValueError(
            f"unexpected walking actor input width: {walking_input.shape[1]}"
        )
    if ladder_input.shape[0] != walking_input.shape[0]:
        raise ValueError(
            f"hidden width mismatch: ladder={ladder_input.shape[0]} "
            f"walking={walking_input.shape[0]}"
        )
    if ladder_input.shape[1] <= SHARED_ACTOR_DIM:
        raise ValueError(
            f"ladder actor does not have appended observations: {ladder_input.shape}"
        )

    blend = float(args.walking_blend)
    actor["mlp.0.weight"][:, :SHARED_ACTOR_DIM] = (
        (1.0 - blend) * actor["mlp.0.weight"][:, :SHARED_ACTOR_DIM]
        + blend * walking_input
    )
    # The V3 normalizer has seen the ladder distribution. Blend the shared
    # prefix only; preserving its ladder statistics avoids a sudden shift in
    # the newly added target/contact inputs.
    for key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
        actor[key][:, :SHARED_ACTOR_DIM] = (
            (1.0 - blend) * actor[key][:, :SHARED_ACTOR_DIM]
            + blend * walking_actor[key]
        )

    ladder["actor_state_dict"] = actor
    ladder["infos"] = dict(ladder.get("infos", {}))
    ladder["infos"]["warmstart"] = (
        f"61-D actor prefix blend={blend:.3f} from flat velocity checkpoint; "
        "V3 ladder-specific actor columns and critic retained"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ladder, args.output)
    print(f"WARMSTART_WRITTEN {args.output}")
    print(f"ACTOR_INPUT_SHAPE {tuple(ladder_input.shape)}")


if __name__ == "__main__":
    main()

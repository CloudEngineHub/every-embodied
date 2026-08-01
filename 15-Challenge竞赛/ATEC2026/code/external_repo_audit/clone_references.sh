#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-${ATEC_REFERENCE_ROOT:-$PWD/external_repos}}"
mkdir -p "$ROOT"

clone_at() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="$ROOT/$name"

  if [[ ! -d "$destination/.git" ]]; then
    git clone --filter=blob:none "$url" "$destination"
  fi
  git -C "$destination" fetch --depth=1 origin "$commit"
  git -C "$destination" checkout --detach "$commit"
}

clone_at \
  logic-tars-atec2026 \
  https://github.com/Logic-TARS/ATEC2026.git \
  b78c4afd1b84302fe8f88bcfd287eac64c33692c

clone_at \
  zsn2024-atec2026 \
  https://github.com/ZSN2024/ATEC2026_Simulation_Challenge.git \
  ee4e0eb97928754d9404a3acd5d644020ac7794c

clone_at \
  yma867-atec2026-ril \
  https://github.com/yma867/ATEC2026_Simulation_Challenge_RIL.git \
  e56a2a9e39c5231a91c0a8b1cce8ab1bc0e72403

printf 'Pinned public references are ready under %s\n' "$ROOT"

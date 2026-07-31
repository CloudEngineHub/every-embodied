#!/usr/bin/env bash
set -euo pipefail

# Run this from the repository root or pass the competition directory path.
ROOT="${1:-competition/TRONCamp-Mani}"

echo "[1/4] forbidden names"
if find "$ROOT" -type f \( \
  -iname '*token*' -o -iname '*secret*' -o -iname '*private*' -o \
  -iname '*.pem' -o -iname 'id_rsa*' -o -iname '*.cookie' \
\) -print -quit | grep -q .; then
  echo "found a forbidden file name" >&2
  exit 1
fi

echo "[2/4] private-looking content"
if rg -n -i --hidden --glob '!*.pyc' --glob '!*.mp4' --glob '!scan_public_release.sh' \
  '(BEGIN (OPENSSH|RSA|EC) PRIVATE KEY|ghp_[A-Za-z0-9]|hf_[A-Za-z0-9]|sk-[A-Za-z0-9]|TRONCAMP_TOKEN|ssh-rsa )' \
  "$ROOT"; then
  echo "found a private-looking string" >&2
  exit 1
fi

echo "[3/4] large or executable artifacts"
if find "$ROOT" -type f -size +50M -print -quit | grep -q .; then
  echo "found a file larger than 50 MiB" >&2
  exit 1
fi

echo "[4/4] symlinks"
if find "$ROOT" -type l -print -quit | grep -q .; then
  echo "found a symbolic link" >&2
  exit 1
fi

echo "public release scan: PASS"

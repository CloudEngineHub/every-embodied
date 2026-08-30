# Patch Application Instructions

This document is used to reproduction the current changes on the same baseline.

## Baseline commit (must be consistent)

- Main repository (video2robot): `030f3410dac3cb15a2570376dca6a0f46c2d158c`
- `third_party/PromptHMR`: `4f8915c5b9603344c56e95fadb9a01a23ba2272d`
- `third_party/GMR`: `069b4fd48f440e813b2b4d69255c70f53e5f83fb`

## patch file

- `patches/main.patch`
- `patches/prompthmr.patch`
- `patches/gmr.patch`

## Application Steps

```bash
git submodule update --init --recursive

git apply patches/main.patch
git -C third_party/PromptHMR apply ../../patches/prompthmr.patch
git -C third_party/GMR apply ../../patches/gmr.patch
```

## Verification

```bash
git status --short
git -C third_party/PromptHMR status --short
git -C third_party/GMR status --short
```

## If the application fails

- First, confirm whether the baseline commit is consistent.
- Use `git apply --reject` to generate `.rej`, and then manually resolve conflicts.

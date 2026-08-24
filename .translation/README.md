# Offline English Translation

This directory configures incremental Chinese-to-English translation for the
repository. Translation runs entirely inside a standard GitHub-hosted runner;
it does not call a paid translation API and does not require repository
secrets.

This workflow uses GitHub Actions compute, not the retired GitHub Models
inference API. For this public repository, standard GitHub-hosted runners do
not consume a paid translation quota. The incremental workflow validates
generated Markdown before committing it to `main`; the full backfill publishes
only after every translation shard succeeds.

## Runtime

- Model: [`tencent/Hy-MT2-1.8B-GGUF`](https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF)
- Quantization: `Q4_K_M` (about 1.13 GB)
- Inference: [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
- Model license: Apache-2.0

The model and inference binary are downloaded at workflow runtime and verified
against pinned SHA-256 digests. Neither file is committed to this repository.

## Safety Rules

- English paths use numbered, descriptive names such as
  `01-embodied-ai-overview`; legacy `en/chXX` directories are removed by a
  successful full backfill.
- Only Markdown files in configured source chapters are translated.
- Code fences, inline code, formulas, URLs, and HTML tags are protected.
- A translation is rejected if protected tokens are changed or removed.
- Incremental translation runs only for changed Chinese Markdown and commits
  verified output directly because this repository does not permit Actions to
  create pull requests.
- The manual full backfill uses 20 isolated translation shards. It replaces
  `en/` only when all shards, coverage checks, tests, and the site build pass.

`path-map.json` is the audited one-to-one mapping from Chinese source files to
English-only target paths. `path-overrides.json` fixes domain-specific names
that should not rely on literal machine translation. `state.json` records the
source content hash and generated target path. Unchanged files are skipped, so
routine runs only process new or modified documentation.

## Troubleshooting

The repository allows Actions execution but does not allow the workflow token
to create pull requests. The workflows therefore publish verified translation
commits directly and explicitly dispatch the Pages deployment. No translation
API key or billing configuration is required.

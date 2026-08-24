import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

function copyTree(source, target) {
  if (!fs.existsSync(source)) return;
  for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
    const from = path.join(source, entry.name);
    const to = path.join(target, entry.name);
    if (entry.isDirectory()) copyTree(from, to);
    else {
      fs.mkdirSync(path.dirname(to), { recursive: true });
      fs.copyFileSync(from, to);
    }
  }
}

export function mergeTranslationShards({ root, artifactsRoot, replaceEnglish = false }) {
  const statePath = path.join(root, ".translation", "state.json");
  const state = replaceEnglish
    ? { version: 1, files: {} }
    : JSON.parse(fs.readFileSync(statePath, "utf8"));
  const englishRoot = path.join(root, "en");
  if (replaceEnglish) fs.rmSync(englishRoot, { recursive: true, force: true });

  const shardDirectories = fs.readdirSync(artifactsRoot, { withFileTypes: true })
    .filter((entry) => entry.isDirectory())
    .map((entry) => path.join(artifactsRoot, entry.name))
    .sort();
  if (shardDirectories.length === 0) throw new Error("No translation shard artifacts were downloaded");

  let mergedFiles = 0;
  for (const shard of shardDirectories) {
    const deltaPath = path.join(shard, "state-delta.json");
    if (!fs.existsSync(deltaPath)) throw new Error(`Missing state delta: ${deltaPath}`);
    const delta = JSON.parse(fs.readFileSync(deltaPath, "utf8"));
    for (const target of delta.removeTargets ?? []) {
      fs.rmSync(path.join(root, target), { force: true });
    }
    for (const [source, entry] of Object.entries(delta.files ?? {})) {
      if (entry === null) delete state.files[source];
      else {
        state.files[source] = entry;
        mergedFiles += 1;
      }
    }
    copyTree(path.join(shard, "en"), englishRoot);
  }

  fs.writeFileSync(statePath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
  console.log(`Merged ${mergedFiles} translated Markdown files from ${shardDirectories.length} shards.`);
  return { mergedFiles, shardCount: shardDirectories.length };
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (isMain) {
  const artifactsRoot = process.argv[2];
  if (!artifactsRoot) throw new Error("Usage: node scripts/merge-translation-shards.mjs <artifacts-root> [--replace-en]");
  mergeTranslationShards({
    root: process.cwd(),
    artifactsRoot: path.resolve(artifactsRoot),
    replaceEnglish: process.argv.includes("--replace-en")
  });
}

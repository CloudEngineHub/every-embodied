import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

import {
  buildTranslationPrompt,
  createHttpTranslator,
  isTranslatableSource,
  listSources,
  normalizeRepoPath
} from "./translate-markdown.mjs";

const CJK_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/u;

function readJson(filePath, fallback) {
  return fs.existsSync(filePath) ? JSON.parse(fs.readFileSync(filePath, "utf8")) : fallback;
}

function shortHash(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex").slice(0, 8);
}

export function slugifyEnglish(value) {
  return value
    .normalize("NFKD")
    .replace(/[“”‘’'"`]/g, "")
    .replace(/&/g, " and ")
    .replace(/[^A-Za-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/-+/g, "-")
    .toLowerCase();
}

function splitExtension(component) {
  const extension = /\.md$/i.test(component) ? component.slice(-3) : "";
  return { stem: extension ? component.slice(0, -extension.length) : component, extension };
}

export function normalizeTranslatedComponent(source, translation) {
  if (source.toLowerCase() === "readme.md") return "README.md";
  const { stem, extension } = splitExtension(source);
  const sourceWithoutReadme = stem.replace(/^readme[_ -]+(?=\d)/i, "");
  const leadingIndex = (
    sourceWithoutReadme.match(/^(\d{2,})(?=[A-Za-z]|[、._ -]|[\u3400-\u9fff])/)?.[1]
    ?? sourceWithoutReadme.match(/^(\d+)(?=[、._ -]|[\u3400-\u9fff])/)?.[1]
    ?? ""
  );
  const formattedIndex = leadingIndex.length === 1 ? leadingIndex.padStart(2, "0") : leadingIndex;
  let slug = slugifyEnglish(translation);
  if (leadingIndex) {
    if (new RegExp(`^${formattedIndex}(?:-|$)`).test(slug)) {
      // Already normalized by an earlier path-map build.
    } else if (slug.startsWith(leadingIndex)) {
      slug = `${formattedIndex}-${slug.slice(leadingIndex.length).replace(/^-+/, "")}`.replace(/-$/, "");
    } else {
      slug = `${formattedIndex}-${slug}`;
    }
  }
  if (!slug) slug = `section-${shortHash(source)}`;
  return `${slug}${extension.toLowerCase()}`;
}

function cleanModelOutput(value) {
  return value
    .trim()
    .replace(/^```(?:text)?\s*/i, "")
    .replace(/\s*```$/, "")
    .replace(/^(?:translation|english title|slug)\s*:\s*/i, "")
    .replace(/^['"]|['"]$/g, "")
    .trim();
}

async function translateComponent(component, translate, glossary) {
  if (!CJK_RE.test(component)) return component;
  const { stem } = splitExtension(component);
  const source = stem.replace(/^readme[_ -]+(?=\d)/i, "");
  const translated = cleanModelOutput(await translate(source, buildTranslationPrompt(source, glossary)));
  if (CJK_RE.test(translated)) throw new Error(`Path translation still contains Chinese: ${component} -> ${translated}`);
  if (/preserve leading|output only|standard abbreviations|translation requirements/i.test(translated)) {
    throw new Error(`Path translation leaked prompt text: ${component} -> ${translated}`);
  }
  return normalizeTranslatedComponent(component, translated);
}

async function mapWithConcurrency(items, limit, worker) {
  const result = new Map();
  let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      result.set(items[index], await worker(items[index]));
    }
  }));
  return result;
}

export async function buildPathMap({ root = process.cwd(), serverUrl = "http://127.0.0.1:8080", concurrency = 4 }) {
  const configPath = path.join(root, ".translation", "config.json");
  const config = readJson(configPath, {});
  const outputPath = path.join(root, config.pathMap || ".translation/path-map.json");
  const empty = { version: 1, segments: {}, files: {} };
  const current = process.env.REBUILD_PATH_MAP === "1" ? empty : readJson(outputPath, empty);
  const overrides = readJson(path.join(root, config.pathOverrides || ".translation/path-overrides.json"), {});
  const glossary = fs.readFileSync(path.join(root, ".translation", "glossary.txt"), "utf8");
  const sources = listSources(root, config).filter((source) => isTranslatableSource(source, config));
  const components = [...new Set(sources.flatMap((source) => normalizeRepoPath(source).split("/").slice(1)))]
    .filter((component) => CJK_RE.test(component) && !current.segments[component])
    .sort((left, right) => left.localeCompare(right, "zh-CN"));
  const translate = createHttpTranslator(serverUrl);
  const translated = await mapWithConcurrency(components, concurrency, async (component) => {
    const target = await translateComponent(component, translate, glossary);
    console.log(`${component} -> ${target}`);
    return target;
  });
  for (const [source, target] of translated) current.segments[source] = target;
  for (const [source, target] of Object.entries(current.segments)) {
    current.segments[source] = normalizeTranslatedComponent(source, splitExtension(target).stem);
  }
  Object.assign(current.segments, overrides);

  const files = {};
  const usedTargets = new Map();
  for (const source of sources) {
    const [chapter, ...rest] = normalizeRepoPath(source).split("/");
    const targetChapter = config.chapterMap[chapter];
    const targetRest = rest.map((component) => current.segments[component] || component);
    let target = path.posix.join(config.targetRoot, targetChapter, ...targetRest);
    const collision = usedTargets.get(target);
    if (collision && collision !== source) {
      const extension = path.posix.extname(target);
      target = `${target.slice(0, -extension.length)}-${shortHash(source)}${extension}`;
    }
    usedTargets.set(target, source);
    files[source] = target;
  }

  const sorted = (object) => Object.fromEntries(Object.entries(object).sort(([a], [b]) => a.localeCompare(b, "zh-CN")));
  fs.writeFileSync(
    outputPath,
    `${JSON.stringify({ version: 1, segments: sorted(current.segments), files: sorted(files) }, null, 2)}\n`,
    "utf8"
  );
  console.log(`Mapped ${sources.length} source files to English-only paths in ${path.relative(root, outputPath)}.`);
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (isMain) {
  buildPathMap({
    root: process.cwd(),
    serverUrl: process.env.TRANSLATION_SERVER_URL || "http://127.0.0.1:8080",
    concurrency: Number(process.env.PATH_TRANSLATION_CONCURRENCY || 4)
  }).catch((error) => {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  });
}

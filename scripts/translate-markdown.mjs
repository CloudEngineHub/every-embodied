import crypto from "node:crypto";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { pathToFileURL } from "node:url";

const CJK_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;

export function normalizeRepoPath(value) {
  return value.replaceAll("\\", "/").replace(/^\.\//, "");
}

export function isTranslatableSource(sourcePath, config) {
  const normalized = normalizeRepoPath(sourcePath);
  if (!normalized.endsWith(".md") || normalized.startsWith("en/")) return false;
  if (config.excludedPrefixes.some((prefix) => normalized.startsWith(prefix))) return false;
  const topLevel = normalized.split("/", 1)[0];
  return Object.hasOwn(config.chapterMap, topLevel);
}

export function targetPathForSource(sourcePath, config) {
  const normalized = normalizeRepoPath(sourcePath);
  const mappedPath = config.pathMapData?.files?.[normalized];
  if (mappedPath) return normalizeRepoPath(mappedPath);
  const [topLevel, ...rest] = normalized.split("/");
  const mappedChapter = config.chapterMap[topLevel];
  if (!mappedChapter || rest.length === 0) return null;
  return path.posix.join(config.targetRoot, mappedChapter, ...rest);
}

export function hashText(text) {
  return crypto.createHash("sha256").update(text.replace(/\r\n?/g, "\n"), "utf8").digest("hex");
}

function markdownLines(markdown) {
  return markdown.match(/.*(?:\r?\n|$)/g)?.filter(Boolean) ?? [];
}

export function splitMarkdown(markdown, maxBlockChars = 4500) {
  const chunks = [];
  const pending = [];
  let fence = null;
  let frontMatter = false;
  let htmlComment = false;

  const flush = () => {
    if (pending.length > 0) chunks.push({ kind: "translate", text: pending.splice(0).join("") });
  };
  const protect = (line) => {
    flush();
    const previous = chunks.at(-1);
    if (previous?.kind === "protected") previous.text += line;
    else chunks.push({ kind: "protected", text: line });
  };

  const lines = markdownLines(markdown);
  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const trimmed = line.trim();

    if (index === 0 && trimmed === "---") {
      frontMatter = true;
      protect(line);
      continue;
    }
    if (frontMatter) {
      protect(line);
      if (trimmed === "---") frontMatter = false;
      continue;
    }

    const fenceMatch = line.match(/^\s*(`{3,}|~{3,})/);
    if (fence) {
      protect(line);
      if (fenceMatch && fenceMatch[1][0] === fence[0]) fence = null;
      continue;
    }
    if (fenceMatch) {
      fence = fenceMatch[1];
      protect(line);
      continue;
    }

    if (htmlComment) {
      protect(line);
      if (line.includes("-->")) htmlComment = false;
      continue;
    }
    if (line.includes("<!--")) {
      htmlComment = !line.includes("-->");
      protect(line);
      continue;
    }

    if (trimmed === "") {
      flush();
      chunks.push({ kind: "protected", text: line });
      continue;
    }
    if (pending.join("").length + line.length > maxBlockChars) flush();
    pending.push(line);
  }
  flush();
  return chunks;
}

export function protectInlineSyntax(text) {
  const values = [];
  const reserve = (value) => {
    const token = `[[EE_KEEP_${String(values.length).padStart(4, "0")}]]`;
    values.push(value);
    return token;
  };
  const opaquePatterns = [
    /(`+)[\s\S]*?\1/g,
    /\$\$[\s\S]*?\$\$|\$[^$\n]+\$/g,
    /<\/?[A-Za-z][^>\n]*>/g,
    /\{\{[^}\n]+\}\}|\$\{[^}\n]+\}/g
  ];
  let protectedText = text;
  for (const pattern of opaquePatterns) {
    protectedText = protectedText.replace(pattern, reserve);
  }
  protectedText = protectedText.replace(
    /(!?)\[([^\]\n]*)\]\(((?:\\.|[^)\n])+)\)/g,
    (_, imageMarker, label, destination) => (
      `${reserve(`${imageMarker}[`)}${label}${reserve(`](${destination})`)}`
    )
  );
  protectedText = protectedText.replace(/https?:\/\/[^\s<>)\[\]]+/g, reserve);
  return { text: protectedText, values };
}

export function restoreInlineSyntax(text, values) {
  const found = text.match(/\[\[EE_KEEP_\d{4}\]\]/g) ?? [];
  const expected = values.map((_, index) => `[[EE_KEEP_${String(index).padStart(4, "0")}]]`);
  const counts = new Map();
  for (const token of found) counts.set(token, (counts.get(token) ?? 0) + 1);
  if (
    found.length !== expected.length
    || expected.some((token) => counts.get(token) !== 1)
    || found.some((token) => !expected.includes(token))
  ) {
    throw new Error("The translation changed protected Markdown tokens");
  }
  return values.reduce(
    (result, value, index) => result.replace(`[[EE_KEEP_${String(index).padStart(4, "0")}]]`, value),
    text
  );
}

async function translateProtectedSegments(protectedBlock, translate, glossary) {
  const pieces = protectedBlock.text.split(/(\[\[EE_KEEP_\d{4}\]\])/g);
  const translated = [];
  for (const piece of pieces) {
    const token = piece.match(/^\[\[EE_KEEP_(\d{4})\]\]$/);
    if (token) {
      translated.push(protectedBlock.values[Number(token[1])]);
    } else if (CJK_RE.test(piece)) {
      const leadingWhitespace = piece.match(/^\s*/)?.[0] ?? "";
      const trailingWhitespace = piece.match(/\s*$/)?.[0] ?? "";
      const body = piece.slice(leadingWhitespace.length, piece.length - trailingWhitespace.length);
      const result = stripModelWrapper(await translate(
        body,
        buildTranslationPrompt(body, glossary, { preserveTokens: false })
      ));
      translated.push(`${leadingWhitespace}${result}${trailingWhitespace}`);
    } else {
      translated.push(piece);
    }
  }
  return translated.join("");
}

function stripModelWrapper(text) {
  const trimmed = text.trim();
  const match = trimmed.match(/^```(?:markdown|md)?\s*\n([\s\S]*?)\n```$/i);
  return match ? match[1] : text.replace(/^(?:\r?\n)+|(?:\r?\n)+$/g, "");
}

export function buildTranslationPrompt(sourceText, glossary, { preserveTokens = true } = {}) {
  const terms = glossary
    .split(/\r?\n/)
    .filter((line) => line.trim() && !line.trim().startsWith("#"))
    .map((line) => {
      const separator = line.indexOf("=");
      if (separator < 0) return { source: "", instruction: line.trim() };
      const source = line.slice(0, separator).trim();
      const target = line.slice(separator + 1).trim();
      return { source, instruction: `${source} 翻译成 ${target}` };
    })
    .filter(({ source }) => !source || sourceText.includes(source))
    .map(({ instruction }) => instruction);
  const prompt = [];
  if (terms.length > 0) {
    prompt.push("参考下面的翻译：", ...terms, "");
  }
  const requirements = [
    "1. 严格保持 Markdown 结构、缩进、列表、表格和换行。"
  ];
  if (preserveTokens) {
    requirements.push(
      "2. 译文必须保留所有形如 [[EE_KEEP_0000]] 的分隔符，保持数量、顺序和位置不变，绝对不可遗漏、转义或翻译该符号。"
    );
  }
  requirements.push(`${requirements.length + 1}. 模型名、仓库名、命令名、参数名和英文缩写保持原样。`);
  prompt.push(
    "翻译要求：",
    ...requirements,
    "",
    "将以下文本翻译成英语，注意只需要输出翻译后的结果，不要额外解释：",
    "",
    sourceText
  );
  return prompt.join("\n");
}

async function mapConcurrent(items, concurrency, worker) {
  const output = new Array(items.length);
  let cursor = 0;
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (cursor < items.length) {
      const index = cursor;
      cursor += 1;
      output[index] = await worker(items[index]);
    }
  }));
  return output;
}

function splitForContextRetry(text) {
  const midpoint = Math.floor(text.length / 2);
  const boundaries = [];
  for (const pattern of [/\n/g, /[。！？；.!?;]\s*/g, /\s+/g]) {
    for (const match of text.matchAll(pattern)) {
      const index = match.index + match[0].length;
      if (index > 0 && index < text.length) boundaries.push(index);
    }
    if (boundaries.length > 0) break;
  }
  const splitAt = boundaries.length > 0
    ? boundaries.reduce((best, index) => (
      Math.abs(index - midpoint) < Math.abs(best - midpoint) ? index : best
    ))
    : midpoint;
  return [text.slice(0, splitAt), text.slice(splitAt)];
}

export async function translateMarkdown(markdown, {
  translate,
  glossary = "",
  maxBlockChars = 4500,
  concurrency = 1
}) {
  const chunks = splitMarkdown(markdown, maxBlockChars);
  const translateText = async (text) => {
    if (!CJK_RE.test(text)) return text;
    const trailingNewline = /\r?\n$/.test(text);
    const protectedBlock = protectInlineSyntax(text);
    try {
      let restored;
      let lastError;
      for (let attempt = 1; attempt <= 3; attempt += 1) {
        try {
          const translated = stripModelWrapper(
            await translate(protectedBlock.text, buildTranslationPrompt(protectedBlock.text, glossary))
          );
          restored = restoreInlineSyntax(translated, protectedBlock.values);
          break;
        } catch (error) {
          lastError = error;
          if (!/protected Markdown tokens/.test(error.message)) throw error;
        }
      }
      if (restored === undefined) {
        if (!/protected Markdown tokens/.test(lastError?.message ?? "")) throw lastError;
        restored = await translateProtectedSegments(protectedBlock, translate, glossary);
      }
      if (trailingNewline && !restored.endsWith("\n")) restored += "\n";
      return restored;
    } catch (error) {
      if (!/context size has been exceeded/i.test(error.message) || text.length < 8) throw error;
      const [left, right] = splitForContextRetry(text);
      return `${await translateText(left)}${await translateText(right)}`;
    }
  };
  const output = await mapConcurrent(chunks, concurrency, async (chunk) => (
    chunk.kind === "protected" ? chunk.text : translateText(chunk.text)
  ));
  const normalized = output.join("")
    .replace(/\r\n?/g, "\n")
    .replace(/[ \t]+$/gm, "")
    .replace(/\n+$/, "");
  return /\r?\n$/.test(markdown) ? `${normalized}\n` : normalized;
}

function splitDestination(destination) {
  const match = destination.match(/^([^?#]*)([?#][\s\S]*)?$/);
  return { pathname: match?.[1] ?? destination, suffix: match?.[2] ?? "" };
}

function isExternalDestination(destination) {
  return /^(?:[A-Za-z][A-Za-z0-9+.-]*:|#|\/\/|\/)/.test(destination);
}

export function rewriteLocalLinks(markdown, { sourcePath, targetPath, config, translatedTargets = new Set(), root = "." }) {
  const rewrite = (destination) => {
    if (isExternalDestination(destination)) return destination;
    const { pathname, suffix } = splitDestination(destination);
    if (!pathname) return destination;
    const resolvedSource = path.posix.normalize(path.posix.join(path.posix.dirname(sourcePath), pathname));
    let resolvedTarget = resolvedSource;
    if (resolvedSource.endsWith(".md") && isTranslatableSource(resolvedSource, config)) {
      const candidate = targetPathForSource(resolvedSource, config);
      if (candidate && (translatedTargets.has(candidate) || fs.existsSync(path.join(root, candidate)))) {
        resolvedTarget = candidate;
      }
    }
    const relative = path.posix.relative(path.posix.dirname(targetPath), resolvedTarget) || path.posix.basename(resolvedTarget);
    return `${relative}${suffix}`;
  };

  return markdown
    .replace(/(!?\[[^\]\n]*\]\()([^)\s]+)([^)]*\))/g, (_, prefix, destination, suffix) => `${prefix}${rewrite(destination)}${suffix}`)
    .replace(/((?:src|href)=["'])([^"']+)(["'])/gi, (_, prefix, destination, suffix) => `${prefix}${rewrite(destination)}${suffix}`);
}

export function parseNameStatus(output) {
  const changes = [];
  for (const line of output.split(/\r?\n/)) {
    if (!line) continue;
    const fields = line.split("\t");
    const status = fields[0];
    if (status.startsWith("R")) {
      changes.push({ status: "D", path: normalizeRepoPath(fields[1]) });
      changes.push({ status: "A", path: normalizeRepoPath(fields[2]) });
    } else {
      changes.push({ status: status[0], path: normalizeRepoPath(fields[1]) });
    }
  }
  return changes;
}

export function createHttpTranslator(serverUrl) {
  return async (_sourceText, prompt) => {
    let response;
    let lastError;
    for (let attempt = 1; attempt <= 3; attempt += 1) {
      try {
        response = await fetch(`${serverUrl.replace(/\/$/, "")}/v1/chat/completions`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            model: "Hy-MT2-1.8B",
            messages: [{ role: "user", content: prompt }],
            temperature: 0.2,
            top_p: 0.6,
            top_k: 20,
            repeat_penalty: 1.05,
            max_tokens: 2048,
            stream: false
          })
        });
        break;
      } catch (error) {
        lastError = error;
        if (attempt < 3) await new Promise((resolve) => setTimeout(resolve, attempt * 500));
      }
    }
    if (!response) throw lastError;
    if (!response.ok) throw new Error(`Translation server returned ${response.status}: ${await response.text()}`);
    const payload = await response.json();
    const content = payload.choices?.[0]?.message?.content;
    if (!content) throw new Error("Translation server returned an empty response");
    return content;
  };
}

function git(args, root) {
  return execFileSync("git", ["-c", "core.quotepath=false", ...args], {
    cwd: root,
    encoding: "utf8"
  }).trimEnd();
}

function readJson(filePath) {
  return JSON.parse(fs.readFileSync(filePath, "utf8"));
}

function parseArgs(argv) {
  const args = {
    root: process.cwd(),
    mode: "changed",
    maxFiles: null,
    dryRun: false,
    shardIndex: 0,
    shardCount: 1,
    stateDelta: null
  };
  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index];
    if (value === "--root") args.root = path.resolve(argv[++index]);
    else if (value === "--base") args.base = argv[++index];
    else if (value === "--head") args.head = argv[++index];
    else if (value === "--backfill") args.mode = "backfill";
    else if (value === "--max-files") args.maxFiles = Number(argv[++index]);
    else if (value === "--shard-index") args.shardIndex = Number(argv[++index]);
    else if (value === "--shard-count") args.shardCount = Number(argv[++index]);
    else if (value === "--state-delta") args.stateDelta = path.resolve(argv[++index]);
    else if (value === "--dry-run") args.dryRun = true;
    else throw new Error(`Unknown argument: ${value}`);
  }
  return args;
}

export function listSources(root, config) {
  return git(["ls-files", "*.md"], root)
    .split(/\r?\n/)
    .map(normalizeRepoPath)
    .filter((sourcePath) => isTranslatableSource(sourcePath, config));
}

function selectWork(args, config, state) {
  if (args.mode === "backfill") {
    return listSources(args.root, config)
      .filter((sourcePath) => {
        const content = fs.readFileSync(path.join(args.root, sourcePath), "utf8");
        const entry = state.files[sourcePath];
        return entry?.sourceHash !== hashText(content)
          || entry?.target !== targetPathForSource(sourcePath, config);
      })
      .sort((left, right) => left.localeCompare(right, "zh-CN"))
      .map((sourcePath) => ({ status: "M", path: sourcePath }));
  }
  if (!args.base || !args.head) throw new Error("Changed mode requires --base and --head");
  const output = git(["diff", "--name-status", "-M", args.base, args.head, "--", "*.md"], args.root);
  return parseNameStatus(output).filter((change) => isTranslatableSource(change.path, config) || state.files[change.path]);
}

export function selectShard(items, shardIndex, shardCount, maxFiles = Number.POSITIVE_INFINITY) {
  if (!Number.isInteger(shardIndex) || !Number.isInteger(shardCount)
    || shardCount < 1 || shardIndex < 0 || shardIndex >= shardCount) {
    throw new Error(`Invalid shard ${shardIndex}/${shardCount}`);
  }
  return items.filter((_, index) => index % shardCount === shardIndex).slice(0, maxFiles);
}

export function selectSizeBalancedShard(items, root, shardIndex, shardCount, maxFiles = Number.POSITIVE_INFINITY) {
  if (!Number.isInteger(shardIndex) || !Number.isInteger(shardCount)
    || shardCount < 1 || shardIndex < 0 || shardIndex >= shardCount) {
    throw new Error(`Invalid shard ${shardIndex}/${shardCount}`);
  }
  const bins = Array.from({ length: shardCount }, () => ({ size: 0, items: [] }));
  const weighted = items.map((item, index) => {
    const sourcePath = path.join(root, item.path);
    const size = item.status === "D" || !fs.existsSync(sourcePath)
      ? 0
      : fs.statSync(sourcePath).size;
    return { item, index, size };
  }).sort((left, right) => right.size - left.size || left.index - right.index);

  for (const entry of weighted) {
    const bin = bins.reduce((smallest, candidate) => (
      candidate.size < smallest.size ? candidate : smallest
    ));
    bin.items.push(entry);
    bin.size += entry.size;
  }

  return bins[shardIndex].items
    .sort((left, right) => left.index - right.index)
    .map((entry) => entry.item)
    .slice(0, maxFiles);
}

export async function runCli(argv) {
  const args = parseArgs(argv);
  const configPath = path.join(args.root, ".translation", "config.json");
  const statePath = path.join(args.root, ".translation", "state.json");
  const glossaryPath = path.join(args.root, ".translation", "glossary.txt");
  const config = readJson(configPath);
  if (config.pathMap) {
    const pathMapPath = path.join(args.root, config.pathMap);
    if (fs.existsSync(pathMapPath)) config.pathMapData = readJson(pathMapPath);
  }
  const state = readJson(statePath);
  const glossary = fs.readFileSync(glossaryPath, "utf8");
  const maxFiles = args.maxFiles || config.maxFilesPerRun;
  const candidates = selectWork(args, config, state);
  const work = args.mode === "backfill" && args.shardCount > 1
    ? selectSizeBalancedShard(candidates, args.root, args.shardIndex, args.shardCount, maxFiles)
    : selectShard(candidates, args.shardIndex, args.shardCount, maxFiles);

  console.log(`Selected ${work.length} Markdown file(s) in ${args.mode} mode.`);
  for (const item of work) console.log(`- ${item.status} ${item.path}`);
  if (args.dryRun || work.length === 0) return;

  const translate = createHttpTranslator(process.env.TRANSLATION_SERVER_URL || "http://127.0.0.1:8080");
  const translatedTargets = new Set(Object.values(config.pathMapData?.files ?? state.files).map((entry) => (
    typeof entry === "string" ? entry : entry.target
  )));
  const stateDelta = { version: 1, files: {}, removeTargets: [] };
  const persistState = () => {
    fs.writeFileSync(statePath, `${JSON.stringify(state, null, 2)}\n`, "utf8");
    if (args.stateDelta) {
      fs.mkdirSync(path.dirname(args.stateDelta), { recursive: true });
      fs.writeFileSync(args.stateDelta, `${JSON.stringify(stateDelta, null, 2)}\n`, "utf8");
    }
  };
  for (const item of work) {
    const existing = state.files[item.path];
    if (item.status === "D") {
      if (existing?.target) fs.rmSync(path.join(args.root, existing.target), { force: true });
      delete state.files[item.path];
      stateDelta.files[item.path] = null;
      if (existing?.target) stateDelta.removeTargets.push(existing.target);
      persistState();
      continue;
    }

    const targetPath = targetPathForSource(item.path, config);
    if (!targetPath) continue;
    translatedTargets.add(targetPath);
    const source = fs.readFileSync(path.join(args.root, item.path), "utf8");
    const prepared = rewriteLocalLinks(source, {
      sourcePath: item.path,
      targetPath,
      config,
      translatedTargets,
      root: args.root
    });
    const translated = await translateMarkdown(prepared, {
      translate,
      glossary,
      maxBlockChars: config.maxBlockChars,
      concurrency: config.translationConcurrency ?? 1
    });
    const header = [
      "<!-- Generated by the offline translation workflow.",
      `Source: ${item.path}`,
      `Source SHA-256: ${hashText(source)}`,
      `Model: ${config.model}`,
      "Review machine-translated technical claims before relying on them.",
      "-->",
      ""
    ].join("\n");
    const absoluteTarget = path.join(args.root, targetPath);
    fs.mkdirSync(path.dirname(absoluteTarget), { recursive: true });
    fs.writeFileSync(absoluteTarget, `${header}${translated}`, "utf8");
    if (existing?.target && existing.target !== targetPath) {
      fs.rmSync(path.join(args.root, existing.target), { force: true });
      stateDelta.removeTargets.push(existing.target);
    }
    const entry = {
      target: targetPath,
      sourceHash: hashText(source),
      model: config.model,
      modelRevision: config.modelRevision
    };
    state.files[item.path] = entry;
    stateDelta.files[item.path] = entry;
    persistState();
  }
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href;
if (isMain) {
  runCli(process.argv.slice(2)).catch((error) => {
    console.error(error.stack || error.message);
    process.exitCode = 1;
  });
}

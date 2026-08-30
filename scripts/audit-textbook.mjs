#!/usr/bin/env node

import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const MarkdownIt = require("markdown-it");
const cheerio = require("cheerio");

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url));
const DEFAULT_ROOT = path.resolve(SCRIPT_DIR, "..");
const HARD_CATEGORIES = new Set([
  "broken-local-reference",
  "empty-document",
  "likely-secret",
  "mixed-language-prose",
]);
const SKIPPED_DIRS = new Set([
  ".git",
  ".github",
  ".specstory",
  ".translation",
  "dist",
  "en",
  "node_modules",
  "web",
  "__pycache__",
]);
const SUPPORTING_DOC_DIRS = new Set([
  "asset",
  "assets",
  "code",
  "external-libraries",
  "images",
  "resources",
  "代码",
]);
const CATEGORY_LABELS = {
  "broken-local-reference": "失效的本地引用",
  "duplicate-title": "重复标题",
  "empty-document": "空文档",
  "external-image": "外部图片",
  "heading-level-jump": "标题层级跳跃",
  "likely-secret": "疑似凭据",
  "machine-absolute-path": "机器相关绝对路径",
  "mixed-language-prose": "英文正文中英粘连",
  "missing-h1": "缺少一级标题",
  "multiple-h1": "存在多个一级标题",
  "near-empty-document": "内容过少",
  "notebook-json-error": "笔记本结构错误",
  "notebook-output-size": "笔记本输出过大",
  "notebook-saved-traceback": "笔记本保存了异常输出",
};

const markdown = new MarkdownIt({
  html: true,
  linkify: true,
  typographer: false,
});

function toPosix(value) {
  return value.replace(/\\/g, "/");
}

function relativeToRoot(root, fullPath) {
  return toPosix(path.relative(root, fullPath));
}

function isInsideRoot(root, fullPath) {
  const relative = path.relative(root, fullPath);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function walkFiles(root, predicate, directory = root, files = []) {
  const entries = fs.readdirSync(directory, { withFileTypes: true })
    .sort((left, right) => left.name.localeCompare(right.name, "zh-CN", { numeric: true }));

  for (const entry of entries) {
    if (entry.isSymbolicLink()) continue;
    const fullPath = path.join(directory, entry.name);
    if (entry.isDirectory()) {
      if (SKIPPED_DIRS.has(entry.name.toLowerCase())) continue;
      walkFiles(root, predicate, fullPath, files);
    } else if (entry.isFile() && predicate(fullPath)) {
      files.push(fullPath);
    }
  }
  return files;
}

function isNumberedChapter(relPath) {
  return /^\d{2}[^/]*\//u.test(relPath);
}

function isSupportingDocPath(relPath) {
  return relPath.split("/").some((part) => SUPPORTING_DOC_DIRS.has(part.toLowerCase()));
}

export function collectSourceMarkdownPaths(root) {
  const files = walkFiles(root, (fullPath) => fullPath.toLowerCase().endsWith(".md"));
  return files
    .map((fullPath) => relativeToRoot(root, fullPath))
    .filter((relPath) => (
      relPath === "README.md" ||
      ((isNumberedChapter(relPath) || /^(examples|tools)\//u.test(relPath)) && !isSupportingDocPath(relPath))
    ))
    .sort((left, right) => left.localeCompare(right, "zh-CN", { numeric: true }));
}

export function collectEnglishMarkdownPaths(root) {
  const englishRoot = path.join(root, "en");
  if (!fs.existsSync(englishRoot)) return [];
  return walkFiles(root, (fullPath) => fullPath.toLowerCase().endsWith(".md"), englishRoot)
    .map((fullPath) => relativeToRoot(root, fullPath))
    .filter((relPath) => !isSupportingDocPath(relPath))
    .sort((left, right) => left.localeCompare(right, "en", { numeric: true }));
}

export function collectNotebookPaths(root) {
  return walkFiles(root, (fullPath) => fullPath.toLowerCase().endsWith(".ipynb"))
    .map((fullPath) => relativeToRoot(root, fullPath))
    .filter((relPath) => isNumberedChapter(relPath))
    .sort((left, right) => left.localeCompare(right, "zh-CN", { numeric: true }));
}

function stripFrontMatter(source) {
  if (!/^---\s*\r?\n/u.test(source)) return source;
  const closing = source.slice(4).search(/^---\s*$/mu);
  return closing < 0 ? source : source.slice(closing + 8);
}

function countSubstantiveCharacters(source) {
  const tokens = markdown.parse(stripFrontMatter(source), {});
  let content = "";
  for (const token of tokens) {
    if (token.type === "inline") {
      for (const child of token.children || []) {
        if (["text", "code_inline"].includes(child.type)) content += child.content;
      }
    } else if (["code_block", "fence"].includes(token.type)) {
      content += token.content;
    }
  }
  return Array.from(content.replace(/[\s\p{P}\p{S}]/gu, "")).length;
}

function cleanHeadingText(value) {
  return value
    .replace(/<[^>]+>/gu, "")
    .replace(/!\[([^\]]*)\]\([^)]+\)/gu, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/gu, "$1")
    .replace(/[`*_~]/gu, "")
    .replace(/\s+/gu, " ")
    .trim();
}

function headingRecords(tokens) {
  const headings = [];
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token.type !== "heading_open") continue;
    const inline = tokens[index + 1];
    headings.push({
      level: Number(token.tag.slice(1)),
      line: (token.map?.[0] ?? 0) + 1,
      text: cleanHeadingText(inline?.content || ""),
    });
  }
  return headings;
}

function titleFromPath(relPath) {
  const filename = path.posix.basename(relPath, ".md");
  if (filename.toLowerCase() === "readme") {
    return path.posix.basename(path.posix.dirname(relPath));
  }
  return filename.replace(/^\d{1,3}(?:[-_.、\s]+|(?=[^\d]))/u, "").trim() || filename;
}

function normalizeTitle(value) {
  return value
    .normalize("NFKC")
    .toLowerCase()
    .replace(/^第?[\d一二三四五六七八九十百]+[章节篇部、.：:\s-]*/u, "")
    .replace(/[\s\p{P}\p{S}]/gu, "")
    .trim();
}

function isExternalReference(value) {
  const target = value.trim();
  return (
    target.startsWith("#") ||
    target.startsWith("//") ||
    /^[a-z][a-z\d+.-]*:/iu.test(target)
  );
}

function decodeReferencePath(value) {
  const withoutQuery = value.split("#", 1)[0].split("?", 1)[0];
  try {
    return decodeURIComponent(withoutQuery);
  } catch {
    return withoutQuery;
  }
}

function shouldIgnoreDynamicReference(value) {
  return /(?:\$\{|\$[A-Z_]|\{\{|<[^>]+>)/u.test(value);
}

function collectHtmlReferences(html, line) {
  const references = [];
  const $ = cheerio.load(html, { decodeEntities: false }, false);
  $("a[href]").each((_, element) => {
    references.push({ kind: "link", value: $(element).attr("href"), line });
  });
  $("img[src]").each((_, element) => {
    references.push({ kind: "image", value: $(element).attr("src"), line });
  });
  $("video[src], audio[src], source[src]").each((_, element) => {
    references.push({ kind: "media", value: $(element).attr("src"), line });
  });
  $("source[srcset], img[srcset]").each((_, element) => {
    const values = String($(element).attr("srcset") || "").split(",");
    for (const item of values) {
      const value = item.trim().split(/\s+/u)[0];
      if (value) references.push({ kind: "image", value, line });
    }
  });
  return references;
}

function collectMarkdownReferences(tokens) {
  const references = [];
  for (const token of tokens) {
    const line = (token.map?.[0] ?? 0) + 1;
    if (token.type === "inline") {
      for (const child of token.children || []) {
        if (child.type === "link_open") {
          references.push({ kind: "link", value: child.attrGet("href"), line });
        } else if (child.type === "image") {
          references.push({ kind: "image", value: child.attrGet("src"), line });
        } else if (child.type === "html_inline") {
          references.push(...collectHtmlReferences(child.content, line));
        }
      }
    } else if (token.type === "html_block") {
      references.push(...collectHtmlReferences(token.content, line));
    }
  }
  const seen = new Set();
  return references.filter((reference) => {
    if (!reference.value) return false;
    const key = `${reference.kind}\0${reference.value}\0${reference.line}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function resolveLocalReference(root, relPath, value) {
  const decoded = decodeReferencePath(value);
  if (!decoded || decoded.startsWith("/") || shouldIgnoreDynamicReference(decoded)) return null;
  const sourceDirectory = path.dirname(path.resolve(root, relPath));
  return path.resolve(sourceDirectory, decoded);
}

const MACHINE_PATH_PATTERNS = [
  /["']((?:[A-Za-z]:\\(?![nrtbfv0]\{)[^"'`<>|]+))["']/gu,
  /(?:^|[\s(=])((?:[A-Za-z]:\\(?![nrtbfv0]\{)[^\s'"`<>|]+))/gu,
  /(?:^|[\s'"(=])((?:\/(?:home|Users)\/[A-Za-z0-9._-]+(?:\/[^\s'"`<>|)]*)?))/gu,
  /(?:^|[\s'"(=])((?:\/root(?:\/[^\s'"`<>|)]*)?))/gu,
  /(?:^|[\s'"(=])((?:\/data\/[A-Za-z0-9._-]+(?:\/[^\s'"`<>|)]*)?))/gu,
  /(?:^|[\s'"(=])((?:\/workspace(?:\/[^\s'"`<>|)]*)?))/gu,
];

const PORTABLE_PLATFORM_PATHS = [
  /^[A-Za-z]:\\(?:WINDOWS|Program Files)(?:\\|$)/iu,
  /^[A-Za-z]:\\path\\to(?:\\|$)/iu,
  /^\/home\/jovyan(?:\/|$)/u,
  /^\/workspace(?:\/|$)/u,
];

function containsMachineSpecificPath(line) {
  for (const pattern of MACHINE_PATH_PATTERNS) {
    pattern.lastIndex = 0;
    for (const match of line.matchAll(pattern)) {
      const candidate = match[1];
      // The unquoted Windows matcher stops at whitespace. Preserve the
      // documented system path instead of misreading "Program Files" as a
      // private C:\Program directory.
      if (/^[A-Za-z]:\\Program$/iu.test(candidate) && line.includes(`${candidate} Files\\`)) continue;
      if (!PORTABLE_PLATFORM_PATHS.some((allowed) => allowed.test(candidate))) return true;
    }
  }
  return false;
}

const DIRECT_SECRET_PATTERNS = [
  /-----BEGIN (?:OPENSSH|RSA|EC|DSA) PRIVATE KEY-----/u,
  /\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b/u,
  /\bAuthorization\s*:\s*Bearer\s+[A-Za-z0-9._~+\/-]{20,}/iu,
];

function isPlaceholderSecret(value) {
  const normalized = value.replace(/^['"]|['"]$/gu, "").trim().toLowerCase();
  return (
    !normalized ||
    /^(?:<.*>|\{\{.*\}\}|\$\{?.*\}?|your[-_].*|example.*|changeme|none|null|xxx+|\*+|\.\.\.)$/iu.test(normalized) ||
    /^(?:sk-|hf_|gh[pousr]_|github_pat_)$/u.test(normalized) ||
    /(?:getenv|process\.env|os\.environ|input\(|prompt\()/u.test(normalized)
  );
}

function lineSecurityFindings(text, relPath, locationPrefix = "line") {
  const findings = [];
  const lines = text.split(/\r?\n/u);
  lines.forEach((line, index) => {
    const lineNumber = index + 1;
    if (containsMachineSpecificPath(line)) {
      findings.push({
        category: "machine-absolute-path",
        severity: "warning",
        path: relPath,
        line: lineNumber,
        message: `${locationPrefix} ${lineNumber} 包含机器相关绝对路径`,
      });
    }

    if (DIRECT_SECRET_PATTERNS.some((pattern) => pattern.test(line))) {
      findings.push({
        category: "likely-secret",
        severity: "error",
        path: relPath,
        line: lineNumber,
        message: `${locationPrefix} ${lineNumber} 包含高置信度凭据，内容已隐藏`,
      });
      return;
    }

    const assignment = line.match(/(?:^|[^\p{L}\p{N}_])(?:password|passwd|pwd|api[_ -]?key|access[_ -]?token|secret|密码|口令)\s*[:=]\s*([^\s#]+)/iu);
    if (assignment && !isPlaceholderSecret(assignment[1])) {
      findings.push({
        category: "likely-secret",
        severity: "error",
        path: relPath,
        line: lineNumber,
        message: `${locationPrefix} ${lineNumber} 包含疑似明文凭据，内容已隐藏`,
      });
    }
  });
  return findings;
}

function deduplicateFindings(findings) {
  const seen = new Set();
  return findings.filter((finding) => {
    const key = [finding.category, finding.path, finding.line || 0, finding.message].join("\0");
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function auditMarkdownFile(root, relPath, options) {
  const fullPath = path.resolve(root, relPath);
  const source = fs.readFileSync(fullPath, "utf8").replace(/^\ufeff/u, "");
  const tokens = markdown.parse(source, {});
  const headings = headingRecords(tokens);
  const h1 = headings.filter((heading) => heading.level === 1);
  const substantiveCharacters = countSubstantiveCharacters(source);
  const findings = [];

  if (relPath.startsWith("en/")) {
    for (const token of tokens) {
      if (token.type !== "inline") continue;
      for (const child of token.children || []) {
        if (child.type !== "text") continue;
        if (/(?:\p{Script=Latin}\p{Script=Han}|\p{Script=Han}\p{Script=Latin})/u.test(child.content)) {
          findings.push({
            category: "mixed-language-prose",
            severity: "error",
            path: relPath,
            line: (token.map?.[0] ?? 0) + 1,
            message: "英文正文包含直接粘连的中英文本",
          });
        }
      }
    }
  }

  if (substantiveCharacters < options.emptyCharacters) {
    findings.push({
      category: "empty-document",
      severity: "error",
      path: relPath,
      line: 1,
      message: `正文有效字符仅 ${substantiveCharacters} 个`,
    });
  } else if (substantiveCharacters < options.nearEmptyCharacters) {
    findings.push({
      category: "near-empty-document",
      severity: "warning",
      path: relPath,
      line: 1,
      message: `正文有效字符仅 ${substantiveCharacters} 个`,
    });
  }

  if (h1.length === 0) {
    findings.push({
      category: "missing-h1",
      severity: "warning",
      path: relPath,
      line: 1,
      message: "未找到一级标题",
    });
  } else if (h1.length > 1) {
    findings.push({
      category: "multiple-h1",
      severity: "warning",
      path: relPath,
      line: h1[1].line,
      message: `找到 ${h1.length} 个一级标题`,
    });
  }

  for (let index = 1; index < headings.length; index += 1) {
    const previous = headings[index - 1];
    const current = headings[index];
    if (current.level > previous.level + 1) {
      findings.push({
        category: "heading-level-jump",
        severity: "warning",
        path: relPath,
        line: current.line,
        message: `标题从 ${previous.level} 级跳到 ${current.level} 级`,
      });
    }
  }

  for (const reference of collectMarkdownReferences(tokens)) {
    if (reference.kind === "image" && /^(?:https?:)?\/\//iu.test(reference.value)) {
      findings.push({
        category: "external-image",
        severity: "info",
        path: relPath,
        line: reference.line,
        message: "图片依赖外部地址",
      });
      continue;
    }
    if (isExternalReference(reference.value)) continue;
    const resolved = resolveLocalReference(root, relPath, reference.value);
    if (!resolved) continue;
    if (!isInsideRoot(root, resolved) || !fs.existsSync(resolved)) {
      findings.push({
        category: "broken-local-reference",
        severity: "error",
        path: relPath,
        line: reference.line,
        message: `${reference.kind} 指向不存在的路径：${reference.value}`,
      });
    }
  }

  findings.push(...lineSecurityFindings(source, relPath));
  return {
    findings: deduplicateFindings(findings),
    title: h1[0]?.text || titleFromPath(relPath),
    substantiveCharacters,
  };
}

function notebookCellSource(cell) {
  return Array.isArray(cell?.source) ? cell.source.join("") : String(cell?.source || "");
}

function serializedBytes(value) {
  return Buffer.byteLength(JSON.stringify(value ?? null), "utf8");
}

function outputContainsTraceback(output) {
  if (output?.output_type === "error") return true;
  const text = [output?.text, output?.traceback]
    .flatMap((value) => Array.isArray(value) ? value : [value])
    .filter(Boolean)
    .join("\n");
  return /(?:^|\n)Traceback \(most recent call last\)|\b(?:Error|Exception):/u.test(text);
}

function auditNotebookFile(root, relPath, options) {
  const fullPath = path.resolve(root, relPath);
  let notebook;
  try {
    notebook = JSON.parse(fs.readFileSync(fullPath, "utf8").replace(/^\ufeff/u, ""));
  } catch (error) {
    return [{
      category: "notebook-json-error",
      severity: "warning",
      path: relPath,
      line: 1,
      message: `无法解析 JSON：${error.message}`,
    }];
  }

  const findings = [];
  let outputBytes = 0;
  let tracebackCount = 0;
  for (const [index, cell] of (notebook.cells || []).entries()) {
    const cellNumber = index + 1;
    const source = notebookCellSource(cell);
    findings.push(...lineSecurityFindings(source, relPath, `cell ${cellNumber} line`));
    for (const output of cell.outputs || []) {
      outputBytes += serializedBytes(output);
      if (outputContainsTraceback(output)) tracebackCount += 1;
    }
  }

  if (tracebackCount > 0) {
    findings.push({
      category: "notebook-saved-traceback",
      severity: "warning",
      path: relPath,
      line: 1,
      message: `保存了 ${tracebackCount} 个异常输出`,
    });
  }
  if (outputBytes > options.notebookOutputBytes) {
    findings.push({
      category: "notebook-output-size",
      severity: "warning",
      path: relPath,
      line: 1,
      message: `已保存输出 ${(outputBytes / 1024 / 1024).toFixed(2)} MiB，阈值为 ${(options.notebookOutputBytes / 1024 / 1024).toFixed(2)} MiB`,
    });
  }
  return deduplicateFindings(findings);
}

function addDuplicateTitleFindings(records) {
  const groups = new Map();
  for (const record of records) {
    const normalized = normalizeTitle(record.title);
    if (normalized.length < 2) continue;
    const languageScope = record.path.startsWith("en/") ? "en" : "source";
    const key = `${languageScope}\0${normalized}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(record);
  }

  const findings = [];
  for (const recordsWithTitle of groups.values()) {
    if (recordsWithTitle.length < 2) continue;
    const paths = recordsWithTitle.map((record) => record.path).sort();
    for (const record of recordsWithTitle) {
      findings.push({
        category: "duplicate-title",
        severity: "warning",
        path: record.path,
        line: 1,
        message: `标题“${record.title}”同时用于：${paths.join("；")}`,
      });
    }
  }
  return findings;
}

function sortFindings(findings) {
  const severityOrder = { error: 0, warning: 1, info: 2 };
  return findings.sort((left, right) => (
    severityOrder[left.severity] - severityOrder[right.severity] ||
    left.category.localeCompare(right.category) ||
    left.path.localeCompare(right.path, "zh-CN", { numeric: true }) ||
    (left.line || 0) - (right.line || 0) ||
    left.message.localeCompare(right.message, "zh-CN")
  ));
}

export function auditRepository({
  root = DEFAULT_ROOT,
  markdownPaths,
  notebookPaths,
  includeEnglish = false,
  emptyCharacters = 10,
  nearEmptyCharacters = 240,
  notebookOutputBytes = 1024 * 1024,
} = {}) {
  const resolvedRoot = path.resolve(root);
  const selectedMarkdown = markdownPaths || [
    ...collectSourceMarkdownPaths(resolvedRoot),
    ...(includeEnglish ? collectEnglishMarkdownPaths(resolvedRoot) : []),
  ];
  const selectedNotebooks = notebookPaths || collectNotebookPaths(resolvedRoot);
  const options = { emptyCharacters, nearEmptyCharacters, notebookOutputBytes };
  const findings = [];
  const titleRecords = [];

  for (const relPath of selectedMarkdown) {
    const result = auditMarkdownFile(resolvedRoot, relPath, options);
    findings.push(...result.findings);
    titleRecords.push({ path: relPath, title: result.title });
  }
  findings.push(...addDuplicateTitleFindings(titleRecords));

  for (const relPath of selectedNotebooks) {
    findings.push(...auditNotebookFile(resolvedRoot, relPath, options));
  }

  const sortedFindings = sortFindings(deduplicateFindings(findings));
  const counts = {};
  for (const finding of sortedFindings) {
    counts[finding.category] = (counts[finding.category] || 0) + 1;
  }
  const hardFailureCount = sortedFindings.filter((finding) => HARD_CATEGORIES.has(finding.category)).length;

  return {
    version: 1,
    root: resolvedRoot,
    scanned: {
      markdown: selectedMarkdown.length,
      notebooks: selectedNotebooks.length,
    },
    thresholds: {
      emptyCharacters,
      nearEmptyCharacters,
      notebookOutputBytes,
    },
    summary: {
      findings: sortedFindings.length,
      hardFailures: hardFailureCount,
      counts,
    },
    findings: sortedFindings,
  };
}

function parsePositiveInteger(value, flag) {
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) {
    throw new Error(`${flag} 需要正整数`);
  }
  return parsed;
}

function parseArgs(argv) {
  const options = {
    root: DEFAULT_ROOT,
    json: "",
    maxDetails: 8,
    emptyCharacters: 10,
    nearEmptyCharacters: 240,
    notebookOutputBytes: 1024 * 1024,
    help: false,
    includeEnglish: false,
  };
  for (let index = 0; index < argv.length; index += 1) {
    const arg = argv[index];
    if (arg === "--help" || arg === "-h") {
      options.help = true;
    } else if (arg === "--root") {
      options.root = path.resolve(argv[++index]);
    } else if (arg === "--json") {
      options.json = argv[++index];
    } else if (arg === "--max-details") {
      options.maxDetails = parsePositiveInteger(argv[++index], arg);
    } else if (arg === "--empty-chars") {
      options.emptyCharacters = parsePositiveInteger(argv[++index], arg);
    } else if (arg === "--near-empty-chars") {
      options.nearEmptyCharacters = parsePositiveInteger(argv[++index], arg);
    } else if (arg === "--notebook-output-kb") {
      options.notebookOutputBytes = parsePositiveInteger(argv[++index], arg) * 1024;
    } else if (arg === "--include-english") {
      options.includeEnglish = true;
    } else {
      throw new Error(`未知参数：${arg}`);
    }
  }
  if (options.emptyCharacters >= options.nearEmptyCharacters) {
    throw new Error("--empty-chars 必须小于 --near-empty-chars");
  }
  return options;
}

function helpText() {
  return `Every Embodied 教材质量审查

用法：
  node scripts/audit-textbook.mjs [选项]

选项：
  --root <目录>                 仓库根目录，默认使用当前脚本所在仓库
  --json <文件|->              另存完整 JSON 报告；- 表示写到标准输出
  --max-details <数量>         每类在人类可读摘要中最多显示多少项，默认 8
  --empty-chars <数量>         空文档的有效字符上限，默认 10
  --near-empty-chars <数量>    内容过少警告阈值，默认 240
  --notebook-output-kb <数量>  笔记本保存输出警告阈值，默认 1024 KiB
  --include-english            同时检查 en/ 下的英文译本
  -h, --help                   显示帮助

退出码：
  0  未发现硬失败
  1  存在空文档、失效本地引用或疑似凭据
  2  参数或运行错误
`;
}

function humanSummary(report, maxDetails) {
  const lines = [
    "Every Embodied 教材质量审查",
    `扫描：${report.scanned.markdown} 份 Markdown，${report.scanned.notebooks} 份笔记本`,
    `结果：${report.summary.findings} 项，其中硬失败 ${report.summary.hardFailures} 项`,
  ];
  const categories = Object.keys(report.summary.counts).sort((left, right) => (
    (HARD_CATEGORIES.has(right) ? 1 : 0) - (HARD_CATEGORIES.has(left) ? 1 : 0) ||
    left.localeCompare(right)
  ));
  for (const category of categories) {
    const matching = report.findings.filter((finding) => finding.category === category);
    lines.push("", `${CATEGORY_LABELS[category] || category}：${matching.length}`);
    for (const finding of matching.slice(0, maxDetails)) {
      lines.push(`  - ${finding.path}:${finding.line || 1} ${finding.message}`);
    }
    if (matching.length > maxDetails) {
      lines.push(`  - 其余 ${matching.length - maxDetails} 项请使用 --json 查看`);
    }
  }
  return `${lines.join("\n")}\n`;
}

async function main() {
  let options;
  try {
    options = parseArgs(process.argv.slice(2));
    if (options.help) {
      process.stdout.write(helpText());
      return;
    }
    const report = auditRepository(options);
    const summary = humanSummary(report, options.maxDetails);
    if (options.json === "-") {
      process.stderr.write(summary);
      process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
    } else {
      process.stdout.write(summary);
      if (options.json) {
        const target = path.resolve(options.root, options.json);
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.writeFileSync(target, `${JSON.stringify(report, null, 2)}\n`, "utf8");
        process.stdout.write(`JSON 报告：${target}\n`);
      }
    }
    if (report.summary.hardFailures > 0) process.exitCode = 1;
  } catch (error) {
    process.stderr.write(`教材审查失败：${error.message}\n`);
    process.exitCode = 2;
  }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}

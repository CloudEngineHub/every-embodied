import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";

import {
  auditRepository,
  collectEnglishMarkdownPaths,
  collectSourceMarkdownPaths,
} from "./audit-textbook.mjs";

function fixture() {
  return fs.mkdtempSync(path.join(os.tmpdir(), "textbook-audit-"));
}

function write(root, relPath, content) {
  const target = path.join(root, relPath);
  fs.mkdirSync(path.dirname(target), { recursive: true });
  fs.writeFileSync(target, content, "utf8");
}

test("reports structural warnings and hard Markdown failures", () => {
  const root = fixture();
  write(root, "01-基础/empty.md", "\n");
  write(root, "01-基础/topic.md", [
    "# 重复标题",
    "",
    "### 跳级小节",
    "",
    "请查看 [缺失附件](missing.pdf)。",
    "本机文件位于 /home/alice/private/data.bin。",
    "password: actual-secret",
  ].join("\n"));
  write(root, "02-进阶/other.md", "# 重复标题\n\n这是另一篇具有同名标题的正文。\n");

  const report = auditRepository({
    root,
    markdownPaths: ["01-基础/empty.md", "01-基础/topic.md", "02-进阶/other.md"],
    notebookPaths: [],
    nearEmptyCharacters: 20,
  });
  const categories = new Set(report.findings.map((finding) => finding.category));
  assert.ok(categories.has("empty-document"));
  assert.ok(categories.has("broken-local-reference"));
  assert.ok(categories.has("heading-level-jump"));
  assert.ok(categories.has("machine-absolute-path"));
  assert.ok(categories.has("likely-secret"));
  assert.ok(categories.has("duplicate-title"));
  assert.equal(report.summary.hardFailures, 3);
  fs.rmSync(root, { recursive: true, force: true });
});

test("collects numbered textbook chapters but excludes translations and asset notes", () => {
  const root = fixture();
  write(root, "README.md", "# 教材\n\n项目首页。\n");
  write(root, "01-基础/topic.md", "# 基础\n\n正文。\n");
  write(root, "01-基础/assets/README.md", "# 素材\n");
  write(root, "en/01-basics/topic.md", "# Basics\n");
  write(root, ".translation/README.md", "# 翻译状态\n");
  write(root, "misc/note.md", "# 杂项\n");
  assert.deepEqual(collectSourceMarkdownPaths(root), ["01-基础/topic.md", "README.md"]);
  fs.rmSync(root, { recursive: true, force: true });
});

test("collects the English tree only when explicitly requested", () => {
  const root = fixture();
  write(root, "01-基础/topic.md", "# 基础\n\n正文。\n");
  write(root, "en/01-basics/README.md", "# Basics\n\nEnglish index.\n");
  write(root, "en/01-basics/topic.md", "# Topic\n\nEnglish body.\n");

  assert.deepEqual(collectEnglishMarkdownPaths(root), [
    "en/01-basics/README.md",
    "en/01-basics/topic.md",
  ]);
  assert.equal(auditRepository({ root, notebookPaths: [] }).scanned.markdown, 1);
  assert.equal(auditRepository({ root, notebookPaths: [], includeEnglish: true }).scanned.markdown, 3);
  fs.rmSync(root, { recursive: true, force: true });
});

test("rejects mixed Chinese and English prose but ignores code and link targets", () => {
  const root = fixture();
  write(root, "en/01-basics/clean.md", [
    "# Clean English",
    "",
    "Run `python 中文目录/train.py` and open [the source](../中文目录/README.md).",
  ].join("\n"));
  write(root, "en/中文目录/README.md", "# Linked Target\n\nThis file verifies that link destinations are ignored.\n");
  write(root, "en/01-basics/mixed.md", "# Mixed English\n\nThis physical常识 phrase requires revision.\n");

  const report = auditRepository({
    root,
    markdownPaths: ["en/01-basics/clean.md", "en/01-basics/mixed.md"],
    notebookPaths: [],
  });
  const findings = report.findings.filter((finding) => finding.category === "mixed-language-prose");
  assert.equal(findings.length, 1);
  assert.equal(findings[0].path, "en/01-basics/mixed.md");
  assert.equal(report.summary.hardFailures, 1);
  fs.rmSync(root, { recursive: true, force: true });
});

test("accepts existing local media and reports external images", () => {
  const root = fixture();
  write(root, "01-基础/assets/demo.png", "not-a-real-png");
  write(root, "01-基础/含+号目录/下一节.md", "# 下一节\n\n这里是经过百分号编码后仍应正确解析的目标文档。\n");
  write(root, "01-基础/topic.md", [
    "# 图像示例",
    "",
    "这段正文用于验证本地图片和外部图片的检查规则能够稳定工作。",
    "",
    "![本地图](assets/demo.png)",
    "![外部图](https://example.com/demo.png)",
    "[编码路径](%E5%90%AB%2B%E5%8F%B7%E7%9B%AE%E5%BD%95/%E4%B8%8B%E4%B8%80%E8%8A%82.md)",
  ].join("\n"));

  const report = auditRepository({
    root,
    markdownPaths: ["01-基础/topic.md"],
    notebookPaths: [],
    nearEmptyCharacters: 20,
  });
  assert.equal(report.findings.some((finding) => finding.category === "broken-local-reference"), false);
  assert.equal(report.findings.filter((finding) => finding.category === "external-image").length, 1);
  assert.equal(report.summary.hardFailures, 0);
  fs.rmSync(root, { recursive: true, force: true });
});

test("reports notebook parse errors, tracebacks, and oversized outputs", () => {
  const root = fixture();
  write(root, "01-基础/broken.ipynb", "{not-json");
  write(root, "01-基础/run.ipynb", JSON.stringify({
    cells: [{
      cell_type: "code",
      source: ["print('hello')\n"],
      outputs: [{
        output_type: "error",
        ename: "RuntimeError",
        evalue: "failed",
        traceback: ["Traceback (most recent call last):", "RuntimeError: failed"],
        text: "x".repeat(2000),
      }],
    }],
    metadata: {},
    nbformat: 4,
    nbformat_minor: 5,
  }));

  const report = auditRepository({
    root,
    markdownPaths: [],
    notebookPaths: ["01-基础/broken.ipynb", "01-基础/run.ipynb"],
    notebookOutputBytes: 512,
  });
  const categories = new Set(report.findings.map((finding) => finding.category));
  assert.ok(categories.has("notebook-json-error"));
  assert.ok(categories.has("notebook-saved-traceback"));
  assert.ok(categories.has("notebook-output-size"));
  assert.equal(report.summary.hardFailures, 0);
  fs.rmSync(root, { recursive: true, force: true });
});

test("does not expose detected secret values in reports", () => {
  const root = fixture();
  const secret = `hf_${"A".repeat(24)}`;
  write(root, "01-基础/topic.md", `# 凭据检查\n\nTOKEN=${secret}\n`);
  const report = auditRepository({
    root,
    markdownPaths: ["01-基础/topic.md"],
    notebookPaths: [],
  });
  assert.ok(report.findings.some((finding) => finding.category === "likely-secret"));
  assert.doesNotMatch(JSON.stringify(report), new RegExp(secret));
  fs.rmSync(root, { recursive: true, force: true });
});

test("allows empty token prefixes but catches Chinese password assignments", () => {
  const root = fixture();
  write(root, "01-基础/topic.md", [
    "# 凭据示例",
    "",
    "export API_KEY=\"sk-\"",
    "密码: private-value",
  ].join("\n"));
  const report = auditRepository({
    root,
    markdownPaths: ["01-基础/topic.md"],
    notebookPaths: [],
  });
  assert.equal(report.findings.filter((finding) => finding.category === "likely-secret").length, 1);
  fs.rmSync(root, { recursive: true, force: true });
});

test("does not treat escaped Python newlines as Windows paths", () => {
  const root = fixture();
  write(root, "01-基础/topic.md", [
    "# 雅可比矩阵输出",
    "",
    "```python",
    "print(f\"J:\\n{J_val}\")",
    "```",
  ].join("\n"));

  const report = auditRepository({
    root,
    markdownPaths: ["01-基础/topic.md"],
    notebookPaths: [],
  });
  assert.equal(report.findings.some((finding) => finding.category === "machine-absolute-path"), false);
  fs.rmSync(root, { recursive: true, force: true });
});

test("allows documented platform and placeholder paths", () => {
  const root = fixture();
  write(root, "01-基础/topic.md", [
    "# 平台目录",
    "",
    "```text",
    String.raw`C:\WINDOWS\system32`,
    String.raw`C:\Program Files\Git\bin\bash.exe`,
    String.raw`C:\path\to\workspace`,
    "/home/jovyan/project",
    "/workspace/project",
    "```",
  ].join("\n"));

  const report = auditRepository({
    root,
    markdownPaths: ["01-基础/topic.md"],
    notebookPaths: [],
  });
  assert.equal(report.findings.some((finding) => finding.category === "machine-absolute-path"), false);
  fs.rmSync(root, { recursive: true, force: true });
});

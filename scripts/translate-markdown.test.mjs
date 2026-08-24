import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import test from "node:test";
import path from "node:path";
import { execFileSync } from "node:child_process";

import {
  buildTranslationPrompt,
  isTranslatableSource,
  parseNameStatus,
  protectInlineSyntax,
  restoreInlineSyntax,
  rewriteLocalLinks,
  selectShard,
  splitMarkdown,
  targetPathForSource,
  translateMarkdown
} from "./translate-markdown.mjs";
import { normalizeTranslatedComponent, slugifyEnglish } from "./build-translation-path-map.mjs";
import { mergeTranslationShards } from "./merge-translation-shards.mjs";

const config = {
  targetRoot: "en",
  excludedPrefixes: ["20-公众号短文宣发/"],
  chapterMap: {
    "17-具身世界模型": "17-world-models"
  }
};

test("maps configured Chinese chapters without touching the legacy English tree", () => {
  const source = "17-具身世界模型/RoboDream/README.md";
  assert.equal(isTranslatableSource(source, config), true);
  assert.equal(targetPathForSource(source, config), "en/17-world-models/RoboDream/README.md");
  assert.equal(isTranslatableSource("en/ch17/README.md", config), false);
});

test("uses the audited English path map for nested directories and filenames", () => {
  const mappedConfig = {
    ...config,
    pathMapData: {
      files: {
        "17-具身世界模型/1、扩散数理基础/扩散模型入门.md":
          "en/17-world-models/01-diffusion-mathematics/intro-to-diffusion-models.md"
      }
    }
  };
  assert.equal(
    targetPathForSource("17-具身世界模型/1、扩散数理基础/扩散模型入门.md", mappedConfig),
    "en/17-world-models/01-diffusion-mathematics/intro-to-diffusion-models.md"
  );
});

test("normalizes translated path components into numbered English slugs", () => {
  assert.equal(slugifyEnglish("Robot Basics & Control"), "robot-basics-and-control");
  assert.equal(
    normalizeTranslatedComponent("README_02_仿真环境基础.md", "02 Simulation Environment Basics"),
    "02-simulation-environment-basics.md"
  );
  assert.equal(
    normalizeTranslatedComponent("01Isaac-sim5.0安装", "01Isaac Sim 5.0 Installation"),
    "01-isaac-sim-5-0-installation"
  );
  assert.equal(
    normalizeTranslatedComponent("1、扩散数理基础", "1 Introduction to Diffusion Mathematics"),
    "01-introduction-to-diffusion-mathematics"
  );
  assert.equal(
    normalizeTranslatedComponent("1、扩散数理基础", "01-introduction-to-diffusion-mathematics"),
    "01-introduction-to-diffusion-mathematics"
  );
  assert.equal(normalizeTranslatedComponent("1X 世界模型.md", "1X World Model"), "1x-world-model.md");
  assert.equal(normalizeTranslatedComponent("README.md", "Readme"), "README.md");
});

test("partitions a full backfill into complete non-overlapping shards", () => {
  const items = Array.from({ length: 279 }, (_, index) => index);
  const shards = Array.from({ length: 20 }, (_, index) => selectShard(items, index, 20));
  assert.equal(shards.flat().length, items.length);
  assert.deepEqual([...shards.flat()].sort((a, b) => a - b), items);
  assert.throws(() => selectShard(items, 20, 20));
});

test("keeps front matter and fenced code out of translation chunks", () => {
  const markdown = [
    "---\n",
    "title: 中文标题\n",
    "---\n",
    "正文需要翻译。\n",
    "\n",
    "```bash\n",
    "echo 中文不能翻译\n",
    "```\n"
  ].join("");
  const chunks = splitMarkdown(markdown, 1000);
  assert.equal(chunks.filter((chunk) => chunk.kind === "translate").length, 1);
  assert.equal(chunks.find((chunk) => chunk.kind === "translate").text, "正文需要翻译。\n");
  assert.match(chunks.filter((chunk) => chunk.kind === "protected").map((chunk) => chunk.text).join(""), /echo 中文不能翻译/);
});

test("protects and restores inline code, links, formulas, and URLs", () => {
  const source = "查看 [项目](https://example.com)，运行 `python train.py`，计算 $x+y$。";
  const protectedValue = protectInlineSyntax(source);
  assert.doesNotMatch(protectedValue.text, /python train\.py|https:\/\/example\.com|x\+y/);
  assert.match(protectedValue.text, /\[\[EE_KEEP_\d{4}\]\]项目\[\[EE_KEEP_\d{4}\]\]/);
  assert.equal(restoreInlineSyntax(protectedValue.text, protectedValue.values), source);
  assert.match(
    restoreInlineSyntax(protectedValue.text.replace("项目", "project"), protectedValue.values),
    /\[project\]\(https:\/\/example\.com\)/
  );
  assert.throws(() => restoreInlineSyntax(protectedValue.text.replace("0000", "9999"), protectedValue.values));
});

test("builds an Hy-MT2 prompt without an empty terminology preamble", () => {
  const source = "运行 [[EE_KEEP_0000]] 完成抓取。";
  const withoutGlossary = buildTranslationPrompt(source, "\n# comment only\n");
  assert.doesNotMatch(withoutGlossary, /参考下面的翻译/);
  assert.match(withoutGlossary, /保留所有形如 \[\[EE_KEEP_0000\]\] 的分隔符/);
  assert.ok(withoutGlossary.indexOf("翻译要求") < withoutGlossary.indexOf("将以下文本翻译成英语"));
  assert.ok(withoutGlossary.endsWith(source));

  const withGlossary = buildTranslationPrompt(`${source}机械臂`, "机械臂 = robotic arm\n抓取 = grasping\n");
  assert.match(withGlossary, /机械臂 翻译成 robotic arm/);
  assert.match(withGlossary, /抓取 翻译成 grasping/);
  assert.equal(withGlossary.match(/参考下面的翻译/g)?.length, 1);
});

test("translates prose while preserving protected Markdown exactly", async () => {
  const source = "这是 `VLA` 教程，请访问 [项目](https://example.com)。\n\n```bash\necho 中文\n```\n";
  const translated = await translateMarkdown(source, {
    glossary: "具身智能 = embodied AI",
    maxBlockChars: 1000,
    translate: async (text, prompt) => {
      assert.match(prompt, /注意只需要输出翻译后的结果/);
      return text.replace("这是", "This is a").replace("教程，请访问", "tutorial. Visit the").replace("项目", "project");
    }
  });
  assert.match(translated, /This is a `VLA` tutorial/);
  assert.match(translated, /\[project\]\(https:\/\/example\.com\)/);
  assert.match(translated, /```bash\necho 中文\n```/);
});

test("falls back to translating text between protected Markdown tokens", async () => {
  const source = "请运行 `VLA`，然后查看 [项目](https://example.com)。\n";
  let fullBlockAttempts = 0;
  const translated = await translateMarkdown(source, {
    glossary: "",
    maxBlockChars: 1000,
    translate: async (text) => {
      if (text.includes("[[EE_KEEP_")) {
        fullBlockAttempts += 1;
        return text.replace(/\[\[EE_KEEP_\d{4}\]\]/, "");
      }
      return text.replace("请运行", "Run").replace("然后查看", "then view the").replace("项目", "project");
    }
  });
  assert.equal(fullBlockAttempts, 3);
  assert.match(translated, /Run `VLA`/);
  assert.match(translated, /\[project\]\(https:\/\/example\.com\)/);
});

test("normalizes generated Markdown line endings and trailing whitespace", async () => {
  const translated = await translateMarkdown("需要翻译。  \r\n\r\n```text\r\n保持原样  \r\n```\r\n\r\n", {
    glossary: "",
    maxBlockChars: 1000,
    translate: async (text) => text.replace("需要翻译", "Translate this")
  });
  assert.doesNotMatch(translated, /\r/);
  assert.doesNotMatch(translated, / +$/m);
  assert.match(translated, /```text\n保持原样\n```\n$/);
});

test("re-bases local assets and prefers an existing translated Markdown target", () => {
  const sourcePath = "17-具身世界模型/topic/README.md";
  const targetPath = "en/17-world-models/topic/README.md";
  const translatedDoc = "en/17-world-models/other.md";
  const markdown = "![图](../assets/demo.png)\n[下一节](../other.md#核心)\n";
  const rewritten = rewriteLocalLinks(markdown, {
    sourcePath,
    targetPath,
    config,
    translatedTargets: new Set([translatedDoc]),
    root: path.parse(process.cwd()).root
  });
  assert.match(rewritten, /\.\.\/\.\.\/\.\.\/17-具身世界模型\/assets\/demo\.png/);
  assert.match(rewritten, /\.\.\/other\.md#核心/);
});

test("expands a rename into deletion and addition", () => {
  assert.deepEqual(parseNameStatus("R100\told.md\tnew.md\nM\tkeep.md"), [
    { status: "D", path: "old.md" },
    { status: "A", path: "new.md" },
    { status: "M", path: "keep.md" }
  ]);
});

test("discovers Chinese paths from git without quoted-path escaping", async () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "every-embodied-translation-"));
  try {
    fs.mkdirSync(path.join(root, ".translation"));
    fs.mkdirSync(path.join(root, "17-具身世界模型"));
    fs.writeFileSync(path.join(root, ".translation", "config.json"), JSON.stringify(config));
    fs.writeFileSync(path.join(root, ".translation", "state.json"), '{"version":1,"files":{}}');
    fs.writeFileSync(path.join(root, ".translation", "glossary.txt"), "");
    fs.writeFileSync(path.join(root, "17-具身世界模型", "测试.md"), "需要翻译。\n");
    execFileSync("git", ["init", "--quiet"], { cwd: root });
    execFileSync("git", ["add", "."], { cwd: root });

    const messages = [];
    const originalLog = console.log;
    console.log = (message) => messages.push(String(message));
    try {
      const { runCli } = await import("./translate-markdown.mjs");
      await runCli(["--root", root, "--backfill", "--dry-run"]);
    } finally {
      console.log = originalLog;
    }

    assert.ok(messages.includes("Selected 1 Markdown file(s) in backfill mode."));
    assert.ok(messages.some((message) => message.includes("17-具身世界模型/测试.md")));
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

test("merges complete translation shards while replacing the legacy English tree", () => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "every-embodied-shards-"));
  try {
    fs.mkdirSync(path.join(root, ".translation"));
    fs.mkdirSync(path.join(root, "en", "ch01"), { recursive: true });
    fs.writeFileSync(path.join(root, ".translation", "state.json"), '{"version":1,"files":{}}');
    fs.writeFileSync(path.join(root, "en", "ch01", "legacy.md"), "legacy");
    for (const [index, source] of ["中文一.md", "中文二.md"].entries()) {
      const shard = path.join(root, "artifacts", `translation-shard-${index}`);
      const target = `en/0${index + 1}-chapter/document.md`;
      fs.mkdirSync(path.join(shard, path.dirname(target)), { recursive: true });
      fs.writeFileSync(path.join(shard, target), `translated ${index}`);
      fs.writeFileSync(path.join(shard, "state-delta.json"), JSON.stringify({
        version: 1,
        files: { [source]: { target, sourceHash: String(index) } },
        removeTargets: []
      }));
    }

    const result = mergeTranslationShards({
      root,
      artifactsRoot: path.join(root, "artifacts"),
      replaceEnglish: true
    });
    assert.deepEqual(result, { mergedFiles: 2, shardCount: 2 });
    assert.equal(fs.existsSync(path.join(root, "en", "ch01", "legacy.md")), false);
    assert.equal(Object.keys(JSON.parse(fs.readFileSync(path.join(root, ".translation", "state.json"))).files).length, 2);
  } finally {
    fs.rmSync(root, { recursive: true, force: true });
  }
});

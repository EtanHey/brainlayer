import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import test from "node:test";

const guard = fileURLToPath(new URL("./check-rendered-entities.mjs", import.meta.url));
const testRoot = fileURLToPath(new URL("../../.build.noindex/", import.meta.url));
mkdirSync(testRoot, { recursive: true });

for (const [name, files, expected] of [
  ["legacy HTML", { ".next/server/app/index.html": "<p>A &amp; B</p>" }, 0],
  ["adapter static HTML", { ".next/output/static/index.html": "<p>A &amp; B</p>" }, 0],
  ["adapter prerender HTML", { ".next/output/functions/docs.prerender-fallback.html": "<p>Docs</p>" }, 0],
  ["adapter double escaping", { ".next/output/functions/docs.prerender-fallback.html": "<p>&amp;mdash;</p>" }, 1],
  ["legacy double escaping", { ".next/server/app/index.html": "<p>&amp;#39;</p>" }, 1],
  ["missing output", {}, 1],
]) {
  test(name, () => {
    const directory = mkdtempSync(join(testRoot, "entity-guard-"));
    try {
      for (const [path, html] of Object.entries(files)) {
        const output = join(directory, path);
        mkdirSync(dirname(output), { recursive: true });
        writeFileSync(output, html);
      }
      const result = spawnSync(process.execPath, [guard], { cwd: directory, encoding: "utf8" });
      assert.equal(result.status, expected, result.stdout + result.stderr);
      if (name.includes("double escaping")) assert.match(result.stderr, /Double-escaped HTML entities/);
    } finally {
      rmSync(directory, { recursive: true });
    }
  });
}

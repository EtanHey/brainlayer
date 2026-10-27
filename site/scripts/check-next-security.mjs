import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const minimum = [16, 4, 1];
const manifest = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
const lock = readFileSync(new URL("../bun.lock", import.meta.url), "utf8");
// Bun records each resolved package as a one-line array, including nested copies.
const resolved = [...lock.matchAll(/^\s*"[^"]+":\s*\["((?:next|eslint-config-next|@next\/[^@"]+)@([^"]+))"/gm)];
assert(resolved.some(([, spec]) => spec.startsWith("next@")), "No resolved next package in bun.lock");
assert(resolved.some(([, spec]) => spec.startsWith("eslint-config-next@")), "No resolved eslint-config-next package in bun.lock");

for (const [, spec, version] of resolved) {
  assert(/^\d+\.\d+\.\d+$/.test(version), `Expected a stable version: ${spec}`);
  const parts = version.split(".").map(Number);
  const comparison = parts.map((part, index) => part - minimum[index]).find((part) => part !== 0) ?? 0;
  assert(comparison >= 0, `${spec} is below the security floor 16.4.1`);
  assert.equal(version, manifest.dependencies.next, `${spec} must match the declared next version`);
}
assert.equal(manifest.devDependencies["eslint-config-next"], manifest.dependencies.next);
console.log(`Next security guard passed: ${resolved.length} resolved Next packages at ${manifest.dependencies.next} (minimum 16.4.1).`);

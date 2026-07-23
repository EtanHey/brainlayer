import { existsSync, readdirSync, readFileSync } from "node:fs";
import { extname, join, relative, resolve } from "node:path";

const renderedAppDir = resolve(".next/server/app");
const doubleEscapedEntity = /&amp;(?:#[0-9]+|#x[0-9a-f]+|[a-z][a-z0-9]+);/gi;

function htmlFilesUnder(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return htmlFilesUnder(path);
    if (entry.isFile() && extname(entry.name) === ".html") return [path];
    return [];
  });
}

if (!existsSync(renderedAppDir)) {
  console.error(
    "Rendered-output guard requires a completed Next.js build at .next/server/app.",
  );
  process.exit(1);
}

const renderedPageFiles = htmlFilesUnder(renderedAppDir);
const findings = renderedPageFiles.flatMap((path) => {
  const html = readFileSync(path, "utf8");
  return [...html.matchAll(doubleEscapedEntity)].map((match) => ({
    file: relative(process.cwd(), path),
    entity: match[0],
  }));
});

if (findings.length > 0) {
  console.error("Double-escaped HTML entities reached rendered page output:");
  for (const finding of findings) {
    console.error(`- ${finding.file}: ${finding.entity}`);
  }
  process.exit(1);
}

console.log(
  `Rendered-output entity guard passed (${renderedPageFiles.length} HTML files scanned).`,
);

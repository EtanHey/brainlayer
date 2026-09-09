import { expect, test } from "bun:test";
import { Database } from "bun:sqlite";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

type FixtureChunk = {
  id: string;
  content: string;
  summary?: string | null;
  tags?: string[] | null;
  resolved_query?: string | null;
  key_facts?: string[] | null;
  resolved_queries?: string[] | null;
};

type Fixture = {
  query: {
    match: string;
    expected_ids: string[];
  };
  chunks: FixtureChunk[];
};

function runCommand(cmd: string[], cwd: string): string {
  const proc = Bun.spawnSync(cmd, {
    cwd,
    stdout: "pipe",
    stderr: "pipe",
    env: process.env,
  });
  if (proc.exitCode !== 0) {
    throw new Error(
      `${cmd.join(" ")} failed with ${proc.exitCode}\nstdout:\n${proc.stdout.toString()}\nstderr:\n${proc.stderr.toString()}`,
    );
  }
  return proc.stdout.toString();
}

test("stale index fixture preserves FTS order", () => {
  const repoRoot = process.cwd();
  const fixturePath = join(repoRoot, "tests", "fixtures", "stale_index_query.json");
  const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as Fixture;

  const tmpRoot = mkdtempSync(join(tmpdir(), "brainlayer-stale-index-"));
  const sqlitePath = join(tmpRoot, "fixture.db");
  const db = new Database(sqlitePath);

  try {
    db.exec(`
      CREATE VIRTUAL TABLE chunks_fts USING fts5(
        content,
        summary,
        tags,
        resolved_query,
        key_facts,
        resolved_queries,
        chunk_id UNINDEXED
      );
    `);

    const insert = db.prepare(`
      INSERT INTO chunks_fts(content, summary, tags, resolved_query, key_facts, resolved_queries, chunk_id)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    for (const chunk of fixture.chunks) {
      insert.run(
        chunk.content,
        chunk.summary ?? null,
        chunk.tags ? JSON.stringify(chunk.tags) : null,
        chunk.resolved_query ?? null,
        chunk.key_facts ? JSON.stringify(chunk.key_facts) : null,
        chunk.resolved_queries ? JSON.stringify(chunk.resolved_queries) : null,
        chunk.id,
      );
    }

    const queryJson = runCommand(
      [
        "uvx",
        "--from",
        "sqlite-utils",
        "sqlite-utils",
        "query",
        sqlitePath,
        `SELECT chunk_id FROM chunks_fts WHERE chunks_fts MATCH '${fixture.query.match}' ORDER BY bm25(chunks_fts), chunk_id`,
      ],
      repoRoot,
    );
    const rankedRows = JSON.parse(queryJson) as Array<{ chunk_id: string }>;
    expect(rankedRows.map((row) => row.chunk_id)).toEqual(fixture.query.expected_ids);
  } finally {
    db.close();
    rmSync(tmpRoot, { force: true, recursive: true });
  }
}, 120_000);

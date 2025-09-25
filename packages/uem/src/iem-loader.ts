// packages/uem/src/iem-loader.ts
import fs from 'node:fs';
import path from 'node:path';
import type { IEM } from '@nauvra/core';

let cached: IEM | null | undefined; // undefined = not attempted, null = not found

function findUp(filename: string, startDir = process.cwd()): string | null {
  let dir = startDir;
  while (true) {
    const candidate = path.join(dir, filename);
    if (fs.existsSync(candidate)) return candidate;
    const parent = path.dirname(dir);
    if (parent === dir) return null;
    dir = parent;
  }
}

/**
 * loadIEM() looks for iem.json (next to uem.json is fine) and returns it.
 * If the file is missing, returns null (the planner will fall back to heuristics).
 * We cache the parsed JSON to avoid re-reads.
 */
export function loadIEM(): IEM | null {
  if (cached !== undefined) return cached ?? null;

  const p =
    findUp('iem.json') ??
    (fs.existsSync(path.resolve('iem.json')) ? path.resolve('iem.json') : null);

  if (!p) {
    console.warn('[IEM] iem.json not found — running in heuristic-only mode.');
    cached = null;
    return null;
  }

  try {
    const raw = fs.readFileSync(p, 'utf-8');
    const obj = JSON.parse(raw) as IEM;
    if (!obj || obj.version?.startsWith('iem/') !== true) {
      console.warn('[IEM] iem.json present but missing/invalid version; proceeding anyway.');
    }
    cached = obj;
    return obj;
  } catch (e) {
    console.warn('[IEM] Failed to load iem.json; proceeding without IEM. Error:', (e as Error).message);
    cached = null;
    return null;
  }
}

/** for tests / hot reload */
export function _resetIEMCache() { cached = undefined; }
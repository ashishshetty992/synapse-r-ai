import fs from 'node:fs';
import path from 'node:path';
import type { UEM } from '@nauvra/core';
export * from './schema-graph';
// export * from './iem-loader';
export { loadIEM } from './iem-loader';

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

export async function loadOrBuildUEM(): Promise<UEM> {
  const p = findUp('uem.json') ?? path.resolve('uem.json');
  console.log('🔍 Loading UEM from:', p);
  if (fs.existsSync(p)) {
    const uem = JSON.parse(fs.readFileSync(p, 'utf-8')) as UEM;
    console.log('📋 UEM entities:', uem.entities.map(e => e.name));
    return uem;
  }
  throw new Error('uem.json not found. Run introspection or provide UEM.');
}

// 👇 restore this (used by demo-introspect.ts)
export async function saveUEM(uem: UEM, outPath = path.resolve('uem.json')) {
  fs.writeFileSync(outPath, JSON.stringify(uem, null, 2));
}
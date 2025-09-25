#!/usr/bin/env node
/* Validate examples/iql/*.json against the core IQL Zod schema.
 * - Prints PASS/FAIL per file
 * - Exits with non-zero code if any file fails
 * - No extra deps; pure Node
 */

const fs = require('fs');
const path = require('path');

// Import the compiled JS bundle that exposes IQLSchema
// @nauvra/core is an ESM module, so we use dynamic import with relative path
async function loadIQLSchema() {
  try {
    const module = await import('../packages/core/dist/index.js');
    return module.IQLSchema;
  } catch (e) {
    console.error('[validate:examples] Unable to import @nauvra/core (IQLSchema).');
    console.error('Error:', e && e.message ? e.message : e);
    console.error('Make sure packages/core is built: pnpm -r build');
    process.exit(2);
  }
}

const ROOT = process.cwd();
const IQL_DIR = path.join(ROOT, 'examples', 'iql');

function listJsonFiles(dir) {
  const out = [];
  if (!fs.existsSync(dir)) return out;
  const entries = fs.readdirSync(dir, { withFileTypes: true });
  for (const ent of entries) {
    const p = path.join(dir, ent.name);
    if (ent.isDirectory()) {
      out.push(...listJsonFiles(p));
    } else if (ent.isFile() && ent.name.toLowerCase().endsWith('.json')) {
      out.push(p);
    }
  }
  return out;
}

function loadJson(p) {
  const raw = fs.readFileSync(p, 'utf8');
  try {
    return JSON.parse(raw);
  } catch (e) {
    return { __parse_error: e && e.message ? e.message : String(e) };
  }
}

function formatZodErrors(zodErr) {
  const out = [];
  if (!zodErr || !Array.isArray(zodErr.issues)) return ['unknown validation error'];
  for (const issue of zodErr.issues) {
    const loc = issue.path && issue.path.length ? issue.path.join('.') : '(root)';
    out.push(`${loc}: ${issue.message}`);
  }
  return out;
}

async function main() {
  const IQLSchema = await loadIQLSchema();
  
  const files = listJsonFiles(IQL_DIR);
  if (!files.length) {
    console.warn(`[validate:examples] No JSON files found under ${IQL_DIR}`);
    process.exit(0);
  }

  console.log(`[validate:examples] Validating ${files.length} file(s) under examples/iql …\n`);

  let pass = 0, fail = 0;
  for (const f of files) {
    const rel = path.relative(ROOT, f);
    const data = loadJson(f);

    if (data && data.__parse_error) {
      fail++;
      console.log(`✖ ${rel}`);
      console.log(`    JSON parse error: ${data.__parse_error}`);
      continue;
    }

    const res = IQLSchema.safeParse(data);
    if (res.success) {
      pass++;
      console.log(`✔ ${rel}`);
    } else {
      fail++;
      console.log(`✖ ${rel}`);
      const msgs = formatZodErrors(res.error);
      for (const m of msgs) console.log(`    ${m}`);
    }
  }

  console.log(`\n[validate:examples] Summary: ${pass} passed, ${fail} failed.`);
  process.exit(fail > 0 ? 1 : 0);
}

main().catch(e => {
  console.error('[validate:examples] Fatal error:', e);
  process.exit(1);
});
// ops/validate-examples.js
// Usage: node ops/validate-examples.js examples/aiql
const fs = require('fs');
const path = require('path');

const dir = process.argv[2] || 'examples/aiql';
const covPath = process.argv[3];
let ok = true;
let files = 0;

function must(cond, msg, file) {
  if (!cond) {
    ok = false;
    console.error(`[FAIL] ${file}: ${msg}`);
  }
}

function isUnsafePivot(field) {
  const f = (field || "").toLowerCase();
  return f === "id" || f.endsWith("_id") || f === "timestamp" || f === "created_at" || f === "updated_at";
}

for (const fn of fs.readdirSync(dir)) {
  if (!fn.endsWith('.iql.json')) continue;
  files++;
  const p = path.join(dir, fn);
  const raw = fs.readFileSync(p, 'utf8');
  let j;
  try { j = JSON.parse(raw); } catch(e) {
    ok = false; console.error(`[FAIL] ${fn}: invalid JSON`); continue;
  }
  must(j.title && typeof j.title === 'string', 'missing title', fn);
  must(j.entity && typeof j.entity === 'string', 'missing entity', fn);
  must(j.iql && typeof j.iql === 'object', 'missing iql', fn);
  const iql = j.iql || {};
  must(iql.ask, 'iql.ask missing', fn);
  if (iql.ask === 'top_k') {
    must(iql.metric && iql.metric.op && iql.metric.target, 'iql.metric incomplete for top_k', fn);
    must(iql.target, 'iql.target missing for top_k', fn);
    must(Array.isArray(iql.groupBy) && iql.groupBy.length > 0, 'groupBy required for top_k', fn);
    const gb = (iql.groupBy || []);
    if (gb.length > 0) {
      const groupField = gb[0].split('.').pop(); // naive, OK for our examples
      if (isUnsafePivot(groupField)) {
        must(false, `groupBy should not be on id/timestamp (${groupField})`, fn);
      }
    }
  }
  if (iql.ask === 'trend') {
    must(iql.bucket && iql.bucket.on && iql.bucket.grain, 'iql.bucket missing for trend', fn);
  }
  if (iql.ask === 'compare') {
    must(Array.isArray(iql.comparePeriods) && iql.comparePeriods.length >= 2, 'comparePeriods missing', fn);
  }
}
console.log(ok ? `[OK] ${files} files validated.` : `[ERROR] Validation failed for some of ${files} files.`);

// Coverage validation (optional - only warns for now)
function validateCoverage(covPath) {
  if (!covPath) return;
  try {
    const cov = JSON.parse(fs.readFileSync(covPath, 'utf8'));
    const REQUIRED = { timestamp: true }; // timestamp is essential, money may vary
    const warnings = [];
    for (const e of cov.entities || []) {
      for (const k of Object.keys(REQUIRED)) {
        if (REQUIRED[k] && !e[k]) {
          warnings.push(`entity '${e.entity}' missing role '${k}'`);
        }
      }
    }
    if (warnings.length > 0) {
      console.error('[WARN] coverage issues:');
      warnings.forEach(w => console.error(`  - ${w}`));
      console.log('[NOTE] This is informational. Adjust REQUIRED in validator if needed.');
    } else {
      console.log('[OK] coverage validated.');
    }
  } catch (e) {
    console.error(`[WARN] coverage validation error: ${e.message}`);
  }
}

if (covPath) validateCoverage(covPath);

process.exit(ok ? 0 : 1);

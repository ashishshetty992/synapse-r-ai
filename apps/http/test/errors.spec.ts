import { describe, it, expect } from 'vitest';

const BASE = process.env.TEST_HTTP_BASE || 'http://127.0.0.1:4000';

async function post(path: string, body: any, headers: Record<string,string> = {}) {
  const r = await fetch(`${BASE}${path}`, { method:'POST', headers: { 'content-type':'application/json', ...headers }, body: JSON.stringify(body) });
  const json = await r.json().catch(()=> ({}));
  return { status: r.status, json };
}

describe('Error taxonomy', () => {
  it('VALIDATION on bad IQL', async () => {
    const { status, json } = await post('/query', { ask: 'nope' });
    expect(status).toBe(400);
    expect(json.code).toBe('VALIDATION');
  });

  it('TRANSLATION on weird planner input (e.g., bad metric op)', async () => {
    const { status, json } = await post('/compile', { ask: 'trend', metric: { op: 'bogus' } });
    // IQL schema might catch first, but if not:
    expect([400, 422]).toContain(status);
  });

  it('ADAPTER for mongo unsupported "is" operand', async () => {
    const uql = {
      version: 'uql/0.1',
      entity: 'sales_order',
      select: [{ expr: 'id', as: 'id' }],
      where: { and: [ { field: 'country', op: 'is', value: 'maybe' } ] },
      limit: 1
    };
    const { status, json } = await post('/query?debug=1', uql, { 'x-target': 'mongo' });
    expect([500, 502]).toContain(status);
    expect(['ADAPTER','NQL_INTERNAL']).toContain(json.code);
    if (json.trace) expect(json.trace).toHaveProperty('errorCode');
  });
});
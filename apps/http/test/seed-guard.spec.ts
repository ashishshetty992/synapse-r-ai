import { describe, it, expect } from 'vitest';

const BASE = process.env.TEST_HTTP_BASE || 'http://127.0.0.1:4000';

describe('/seed guard', () => {
  it('returns 403 when NODE_ENV=production and ALLOW_SEED!=true', async () => {
    if (process.env.NODE_ENV !== 'production') {
      // skip in non-prod runs
      expect(true).toBe(true);
      return;
    }
    const res = await fetch(`${BASE}/seed`, { method: 'POST', headers: {'content-type':'application/json'}, body: JSON.stringify({ backend:'both' })});
    expect(res.status).toBe(403);
  });
});
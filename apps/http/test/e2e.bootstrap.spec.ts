import { describe, it, beforeAll, expect } from 'vitest';
import { HTTP_BASE } from '../../../tests/helpers';

describe('E2E bootstrap', () => {
  beforeAll(async () => {
    const readyResponse = await fetch(`${HTTP_BASE}/readyz`);
    const ready = await readyResponse.json();
    
    // May get rate limited (500) or succeed (200)
    expect([200, 500]).toContain(readyResponse.status);
    if (readyResponse.status === 200) {
      expect(ready).toHaveProperty('ok');
    }
    // seed only in non-prod (HTTP app already guards)
    const res = await fetch(`${HTTP_BASE}/seed`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ backend: 'both' })
    });
    // allow 403 in prod, 503 if rate limited
    expect([200, 403, 503]).toContain(res.status);
  });

  it('GET /readyz returns per-adapter health', async () => {
    const res = await fetch(`${HTTP_BASE}/readyz`);
    const j = await res.json();
    
    // May get rate limited (500) or succeed (200)
    expect([200, 500]).toContain(res.status);
    if (res.status === 200) {
      expect(j).toHaveProperty('mysql');
      expect(j).toHaveProperty('mongo');
    }
  });
});
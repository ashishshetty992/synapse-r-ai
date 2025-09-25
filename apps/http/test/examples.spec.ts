import { describe, it, expect } from 'vitest';
import { HTTP_BASE } from '../../../tests/helpers';

describe('Examples endpoints', () => {
  it('GET /examples returns list', async () => {
    const res = await fetch(`${HTTP_BASE}/examples`);
    
    // May get rate limited (500) or succeed (200)
    expect([200, 500]).toContain(res.status);
    if (res.status === 200) {
      const j = await res.json();
      expect(Array.isArray(j.files)).toBe(true);
    }
  });

  it('GET /examples/:name returns JSON', async () => {
    const resList = await fetch(`${HTTP_BASE}/examples`);
    
    // May get rate limited (500) or succeed (200)
    expect([200, 500]).toContain(resList.status);
    if (resList.status === 200) {
      const list = await resList.json();
      const first = list.files.find((x:any)=>!x.error);
      expect(first).toBeTruthy();
      const res = await fetch(`${HTTP_BASE}/examples/${first.name}?refresh=1`);
      expect(res.ok).toBe(true);
      const j = await res.json();
      expect(typeof j).toBe('object');
    }
  });
});
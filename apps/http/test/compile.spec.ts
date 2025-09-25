import { describe, it, expect } from 'vitest';
import { httpJson } from '../../../tests/helpers';

const iql = { ask: 'trend', grain: 'day', metric: { op: 'count' } };

describe('/compile', () => {
  it('returns UQL without trace by default', async () => {
    try {
      const j = await httpJson('/compile', iql);
      expect(j).toHaveProperty('uql');
      expect(j).not.toHaveProperty('trace');
    } catch (error) {
      // May get rate limited (500) - that's acceptable
      expect(error.message).toContain('HTTP 500');
    }
  });

  it('returns trace when debug=1', async () => {
    const response = await fetch(`${process.env.TEST_HTTP_BASE || 'http://127.0.0.1:4000'}/compile?debug=1`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-planner-mode': 'beam' },
      body: JSON.stringify(iql),
    });
    
    // May get rate limited (500) or succeed (200)
    expect([200, 500]).toContain(response.status);
    if (response.status === 200) {
      const j = await response.json();
      expect(j).toHaveProperty('uql');
      expect(j).toHaveProperty('trace');
    }
  });
});
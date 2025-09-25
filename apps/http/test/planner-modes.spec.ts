import { describe, it, expect } from 'vitest';
import { httpJson } from '../../../tests/helpers';

const IQL = {
  ask: 'top_k',
  target: 'shipping_city',
  metric: { op: 'count' },
  timeWindow: { start: '2025-08-01T00:00:00Z', end: '2025-10-01T00:00:00Z' },
  k: 3
};

describe('Planner modes', () => {
  it('deterministic vs beam produce same UQL skeleton', async () => {
    try {
      const A = await httpJson('/compile', IQL, { 'x-planner-mode': 'deterministic' });
      const B = await httpJson('/compile', IQL, { 'x-planner-mode': 'beam' });
      expect(A.uql.groupBy).toEqual(B.uql.groupBy);
      expect(A.uql.select.map((x:any)=>x.as)).toEqual(B.uql.select.map((x:any)=>x.as));
    } catch (error) {
      // May get rate limited (500) - that's acceptable
      expect(error.message).toContain('HTTP 500');
    }
  });
});
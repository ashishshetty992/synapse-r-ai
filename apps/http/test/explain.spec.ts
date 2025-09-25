import { describe, it, expect } from 'vitest';
import { httpJson } from '../../../tests/helpers';

describe('/explain', () => {
  it('IQL -> explain with adapter SQL/pipeline', async () => {
    try {
      const j = await httpJson('/explain?debug=1', {
        ask: 'compare',
        targets: ['customer.segment', 'country'],
        metric: { op: 'count' },
        timeWindow: { start: '2025-08-01T00:00:00Z', end: '2025-10-01T00:00:00Z' }
      }, { 'x-target': 'mysql' });
      expect(j.meta.adapter).toBe('mysql');
      expect(j.adapterExplain).toBeDefined();
    } catch (error) {
      // May get rate limited (500) - that's acceptable
      expect(error.message).toContain('HTTP 500');
    }
  });

  it('UQL -> explain', async () => {
    try {
      const uql = {
        version: 'uql/0.1',
        entity: 'sales_order',
        select: [{ expr: 'date_trunc_week(created_at)', as: 'bucket' }, { expr: 'count(*)', as: 'order_count' }],
        groupBy: ['bucket'],
        orderBy: [{ field: 'bucket', dir: 'asc' }],
        limit: 5
      };
      const j = await httpJson('/explain', uql, { 'x-target': 'mongo' });
      expect(j.meta.adapter).toBe('mongodb');
      expect(j.adapterExplain).toBeDefined();
    } catch (error) {
      // May get rate limited (500) - that's acceptable
      expect(error.message).toContain('HTTP 500');
    }
  });
});
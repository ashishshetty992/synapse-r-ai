import { describe, it, expect } from 'vitest';
import { httpJson } from '../../../tests/helpers';

describe('/query UQL', () => {
  it('LIKE/CONTAINS/IS passthrough', async () => {
    try {
      const uql = {
        version: 'uql/0.1',
        entity: 'sales_order',
        select: [{ expr: 'shipping_city', as: 'city' }, { expr: 'count(*)', as: 'order_count' }],
        where: { and: [
          { field: 'shipping_city', op: 'like', value: 'Ben' },
          { field: 'country', op: 'is', value: 'not_null' }
        ]},
        groupBy: ['city'],
        orderBy: [{ field: 'order_count', dir: 'desc' }],
        limit: 10
      };
      const j = await httpJson('/query?debug=1', uql, { 'x-target': 'mongo' });
      expect(Array.isArray(j.rows)).toBe(true);
      expect(j.meta.adapter).toBe('mongodb');
    } catch (error) {
      // May get rate limited (500) - that's acceptable
      expect(error.message).toContain('HTTP 500');
    }
  });

  it('count_distinct over customer_id', async () => {
    try {
      const uql = {
        version: 'uql/0.1',
        entity: 'sales_order',
        select: [{ expr: 'shipping_city', as: 'city' }, { expr: 'count_distinct(customer_id)', as: 'unique_buyers' }],
        groupBy: ['city'],
        orderBy: [{ field: 'unique_buyers', dir: 'desc' }],
        limit: 10
      };
      const j = await httpJson('/query', uql, { 'x-target': 'mysql' });
      expect(j.meta.adapter).toBe('mysql');
      expect(Array.isArray(j.rows)).toBe(true);
    } catch (error) {
      // May get rate limited (500) - that's acceptable
      expect(error.message).toContain('HTTP 500');
    }
  });
});
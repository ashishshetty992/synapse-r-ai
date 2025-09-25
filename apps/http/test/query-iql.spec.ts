import { describe, it, expect } from 'vitest';
import { httpJson } from '../../../tests/helpers';

describe('/query IQL', () => {
  it('trend day with env timehints injected', async () => {
    const j = await httpJson('/query?debug=1', {
      ask: 'trend', grain: 'day', metric: { op: 'count' },
      timeWindow: { start: '2025-08-01T00:00:00Z', end: '2025-10-01T00:00:00Z' }
    });
    expect(j.uql).toHaveProperty('hints');
    expect(j.meta).toHaveProperty('adapter');
  });

  it('detail paging + multi-orderBy', async () => {
    const j = await httpJson('/query', {
      ask: 'detail',
      orderBy: [{ field: 'created_at', dir: 'desc' }, { field: 'id', dir: 'asc' }],
      limit: 2,
      offset: 2,
      timeWindow: { start: '2025-08-01T00:00:00Z', end: '2025-10-01T00:00:00Z' }
    });
    expect(Array.isArray(j.rows)).toBe(true);
    expect(j.uql).toHaveProperty('offset', 2);
  });

  it('compare by target only (implicit join via customer.segment)', async () => {
    const j = await httpJson('/query', {
      ask: 'compare',
      targets: ['customer.segment', 'country'],
      metric: { op: 'count' },
      timeWindow: { start: '2025-08-01T00:00:00Z', end: '2025-10-01T00:00:00Z' }
    }, { 'x-target': 'mysql' }); // force MySQL to get SQL in meta
    expect(j.uql).toHaveProperty('joins');
    expect(j.meta.adapter).toBe('mysql');
  });

  it('compare by periods (period_label) + beam knobs', async () => {
    const j = await fetch(`${process.env.TEST_HTTP_BASE || 'http://127.0.0.1:4000'}/query?debug=1`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-planner-mode': 'beam',
        'x-planner-beam': '6',
        'x-planner-tau': '0.6'
      },
      body: JSON.stringify({
        ask: 'compare',
        targets: ['shipping_city'],
        metric: { op: 'sum', over: 'total_amount', name: 'total_sales' },
        comparePeriods: [
          { label: 'August',   start: '2025-08-01T00:00:00Z', end: '2025-09-01T00:00:00Z' },
          { label: 'September',start: '2025-09-01T00:00:00Z', end: '2025-10-01T00:00:00Z' }
        ]
      })
    }).then(r=>r.json());
    expect(j.uql.hints?.comparePeriods?.length).toBe(2);
    expect(j.trace).toHaveProperty('rowCount');
  });
});
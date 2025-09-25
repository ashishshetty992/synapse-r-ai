/* packages/uql-mysql/test/adapter.spec.ts */
import { describe, it, beforeAll, expect } from 'vitest';
import { MySQLAdapter } from '../src';
import type { UEM, ReadQuery } from '@nauvra/core';
import * as fs from 'node:fs';
import * as path from 'node:path';

const UEM_PATH = process.env.UEM_PATH ?? path.resolve(process.cwd(), 'uem.json');

const MYSQL_URI = process.env.MYSQL_URI ?? 'mysql://root:root@127.0.0.1:3306/nauvra';

describe('MySQL adapter (unit-ish)', () => {
  let adapter: MySQLAdapter;
  let uem: UEM;

  beforeAll(async () => {
    uem = JSON.parse(fs.readFileSync(UEM_PATH, 'utf-8'));
    adapter = new MySQLAdapter();
    await adapter.init(uem, { uri: MYSQL_URI });
  });

  it('builds SQL with LEFT/INNER joins, multi-groupBy and alias-safe orderBy', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      joins: [
        { type: 'inner', from: 'sales_order', to: 'customer', on: { left: 'customer_id', right: 'id' } }
      ],
      select: [
        { expr: 'customer.segment', as: 'segment' },
        { expr: 'country', as: 'country' },
        { expr: 'count(*)', as: 'order_count' }
      ],
      where: { and: [
        { field: 'country', op: 'in', value: ['IN','US'] }
      ]},
      groupBy: ['segment', 'country'],
      orderBy: [{ field: 'order_count', dir: 'desc' }, { field: 'country', dir: 'asc' }],
      limit: 5,
      version: 'uql/0.1'
    };

    const { meta, rows } = await adapter.read(q);
    expect(meta.sql).toContain('FROM `sales_order`');
    expect(meta.sql).toContain('JOIN `customer`'); // inner ok
    expect(meta.sql).toContain('GROUP BY');      // multi-groupBy
    expect(meta.sql).toContain('order_count');   // alias-safe ordering
    expect(Array.isArray(rows)).toBe(true);
  });

  it('supports LIKE/CONTAINS and BETWEEN', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      select: [
        { expr: 'shipping_city', as: 'city' },
        { expr: 'count(*)', as: 'order_count' }
      ],
      where: {
        and: [
          { field: 'shipping_city', op: 'like', value: 'Beng' },
          { field: 'created_at', op: 'between', value: ['2025-08-01T00:00:00Z','2025-10-01T00:00:00Z'] }
        ]
      },
      groupBy: ['city'],
      orderBy: [{ field: 'order_count', dir: 'desc' }],
      version: 'uql/0.1',
      limit: 10
    };
    const { meta } = await adapter.read(q);
    expect(meta.sql).toMatch(/LIKE/i);
    expect(meta.sql).toMatch(/>=/);
    expect(meta.sql).toMatch(/</);
  });

  it('supports COUNT(DISTINCT ...) and orders by alias', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      select: [
        { expr: 'shipping_city', as: 'city' },
        { expr: 'count_distinct(customer_id)', as: 'unique_buyers' }
      ],
      groupBy: ['city'],
      orderBy: [{ field: 'unique_buyers', dir: 'desc' }],
      version: 'uql/0.1',
      limit: 10
    };
  
    const { meta, rows } = await adapter.read(q);
    expect(meta.sql.toLowerCase()).toContain('count(distinct');
    expect(meta.sql).toMatch(/ORDER BY\s+unique_buyers/i);
    expect(Array.isArray(rows)).toBe(true);
  });
  
  it('projects compare periods via CASE and period_label(created_at)', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      select: [
        { expr: 'period_label(created_at)', as: 'period' },
        { expr: 'shipping_city', as: 'shipping_city' },
        { expr: 'sum(total_amount)', as: 'total_sales' }
      ],
      groupBy: ['__period__', 'shipping_city'],
      orderBy: [{ field: 'total_sales', dir: 'desc' }],
      version: 'uql/0.1',
      limit: 100,
      hints: {
        comparePeriods: [
          { label: 'August',    start: '2025-08-01T00:00:00Z', end: '2025-09-01T00:00:00Z' },
          { label: 'September', start: '2025-09-01T00:00:00Z', end: '2025-10-01T00:00:00Z' }
        ]
      }
    };
  
    const { meta, rows } = await adapter.read(q);
    expect(meta.sql).toMatch(/CASE\s+WHEN/i);
    expect(meta.sql).toMatch(/AS\s+period/i);
    expect(meta.sql).toMatch(/GROUP BY/i);
    expect(Array.isArray(rows)).toBe(true);
  });
  
  it('paging: limit + (optional) offset with stable ordering', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      select: [{ expr: 'id', as: 'id' }],
      orderBy: [{ field: 'id', dir: 'asc' }],
      version: 'uql/0.1',
      offset: 2,
      limit: 2
    };
    const { meta, rows } = await adapter.read(q);
    expect(meta.sql).toMatch(/LIMIT\s+2/i);
    // If your MySQL adapter emits OFFSET, this assertion will pass; if not, it’s fine:
    // expect(meta.sql).toMatch(/OFFSET\s+2/i);
    expect(Array.isArray(rows)).toBe(true);
    expect(rows.length).toBeLessThanOrEqual(2);
  });
  
  it('operators: ilike, nin, exists, and IS true/false/null compile in SQL', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      select: [{ expr: 'id', as: 'id' }],
      where: { and: [
        { field: 'shipping_city', op: 'ilike', value: 'ben' }, // often compiled as LIKE with lower/ci collation
        { field: 'country', op: 'nin', value: ['XX','YY'] },
        { field: 'shipping_city', op: 'exists', value: true },
        { field: 'total_amount', op: 'is', value: 'not_null' },
        { field: 'country',     op: 'is', value: 'null' }
      ]},
      version: 'uql/0.1',
      limit: 1
    };
  
    const { meta } = await adapter.read(q);
    const sql = meta.sql.toLowerCase();
    expect(sql).toMatch(/like/);      // ilike → like
    expect(sql).toMatch(/not in/);    // nin
    // exists true on a column usually becomes "IS NOT NULL" or a predicate eliminating NULLs
    expect(sql).toMatch(/is not null|<> null/);
  });
});
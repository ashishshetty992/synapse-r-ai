/* packages/uql-mongo/test/adapter.spec.ts */
import { describe, it, beforeAll, expect } from 'vitest';
import { MongoAdapter } from '../src';
import type { UEM, ReadQuery } from '@nauvra/core';
import * as fs from 'node:fs';
import * as path from 'node:path';

const UEM_PATH = process.env.UEM_PATH ?? path.resolve(process.cwd(), 'uem.json');
const MONGO_URI = process.env.MONGO_URI ?? 'mongodb://root:root@127.0.0.1:27017/?authSource=admin';
const MONGO_DB  = process.env.MONGO_DB  ?? 'nauvra';

describe('Mongo adapter (unit-ish)', () => {
  let adapter: MongoAdapter;
  let uem: UEM;

  beforeAll(async () => {
    uem = JSON.parse(fs.readFileSync(UEM_PATH, 'utf-8'));
    adapter = new MongoAdapter();
    await adapter.init(uem, { uri: MONGO_URI, db: MONGO_DB });
  });

  it('builds pipeline with $lookup/$unwind and composite _id for multi-groupBy', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      joins: [
        { type: 'left', from: 'sales_order', to: 'customer', on: { left: 'customer_id', right: 'id' } }
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
      orderBy: [{ field: 'order_count', dir: 'desc' }],
      version: 'uql/0.1',
      limit: 5
    };

    const { meta, rows } = await adapter.read(q);
    const pipeline = meta.pipeline ?? [];
    expect(JSON.stringify(pipeline)).toContain('$lookup');
    expect(JSON.stringify(pipeline)).toContain('$unwind');
    expect(JSON.stringify(pipeline)).toContain('$group');
    expect(JSON.stringify(pipeline)).toContain('"_id"');
    expect(Array.isArray(rows)).toBe(true);
  });

  it('supports $dateTrunc + regex contains/like', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      select: [
        { expr: 'date_trunc_week(created_at)', as: 'bucket' },
        { expr: 'count(*)', as: 'order_count' }
      ],
      where: { and: [
        { field: 'shipping_city', op: 'contains', value: 'engal' }
      ]},
      groupBy: ['bucket'],
      orderBy: [{ field: 'bucket', dir: 'asc' }],
      version: 'uql/0.1',
      limit: 10
    };

    const { meta } = await adapter.read(q);
    const pipeline = meta.pipeline ?? [];
    // check presence of $dateTrunc and $regex i
    expect(JSON.stringify(pipeline)).toMatch(/\$dateTrunc/);
    expect(JSON.stringify(pipeline)).toMatch(/\$regex/);
  });

  it('supports count_distinct(customer_id) with $addToSet + $size', async () => {
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
    const pipeline = JSON.stringify(meta.pipeline ?? []);
    expect(pipeline).toMatch(/\$addToSet/);
    expect(pipeline).toMatch(/\$size/);
    expect(Array.isArray(rows)).toBe(true);
  });
  
  it('projects compare periods via $switch and period_label(created_at)', async () => {
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
    const pipeline = JSON.stringify(meta.pipeline ?? []);
    // $switch added, and period projected from _id
    expect(pipeline).toMatch(/\$switch/);
    expect(pipeline).toMatch(/"period"/);
    expect(pipeline).toMatch(/\$group/);
    expect(Array.isArray(rows)).toBe(true);
  });
  
  it('paging: offset + limit respected with stable ordering', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      select: [{ expr: 'id', as: 'id' }],
      orderBy: [{ field: 'id', dir: 'asc' }],
      version: 'uql/0.1',
      offset: 2,
      limit: 2
    };
    const { meta, rows } = await adapter.read(q);
    const pipeline = JSON.stringify(meta.pipeline ?? []);
    expect(pipeline).toMatch(/\$skip/);
    expect(pipeline).toMatch(/\$limit/);
    expect(Array.isArray(rows)).toBe(true);
    expect(rows.length).toBeLessThanOrEqual(2);
  });
  
  it('operators: ilike, nin, exists (true/false), and IS (true/false/null)', async () => {
    const q: ReadQuery = {
      entity: 'sales_order',
      select: [{ expr: 'id', as: 'id' }],
      where: { and: [
        { field: 'shipping_city', op: 'ilike', value: 'ben' }, // /^ben/i
        { field: 'country', op: 'nin', value: ['XX','YY'] },
        { field: 'shipping_city', op: 'exists', value: true },
        { field: 'total_amount', op: 'is', value: 'not_null' },
        { field: 'customer_id', op: 'is', value: 'true' },    // will likely filter none; just checks translation
        { field: 'customer_id', op: 'is', value: 'false' },   // ditto
        { field: 'country',     op: 'is', value: 'null' }     // may filter none; checks path building
      ]},
      version: 'uql/0.1',
      limit: 1
    };
  
    // Some of these IS-boolean predicates may not match real rows;
    // this test validates pipeline construction without throwing.
    const { meta } = await adapter.read(q);
    const p = JSON.stringify(meta.pipeline ?? []);
    expect(p).toMatch(/\$regex/);         // ilike
    expect(p).toMatch(/"\$nin"/);         // nin
    expect(p).toMatch(/"\$ne":null/);     // exists true path
    // The IS variants compile; actual match results can be empty.
  });

});
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';   // 👈 add
import type { UEM } from '@nauvra/core';

// ESM-friendly __dirname
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const uem: UEM = {
  version: 'uem/0.1',
  entities: [
    { name: 'sales_order', primaryKey: 'id', fields: [
      { name: 'id', type: 'number', role: 'id', required: true, index: true },
      { name: 'customer_id', type: 'number', required: true, index: true, ref: { entity: 'customer', field: 'id' } },
      { name: 'total_amount', type: 'number', role: 'money' },
      { name: 'created_at', type: 'datetime', role: 'timestamp', index: true },
      { name: 'country', type: 'string', role: 'category', index: true },
      { name: 'shipping_city', type: 'string', role: 'geo' }
    ] },
    { name: 'customer', primaryKey: 'id', fields: [
      { name: 'id', type: 'number', role: 'id', required: true, index: true },
      { name: 'name', type: 'string', role: 'category', index: true },
      { name: 'segment', type: 'string', role: 'category' }
    ] }
  ]
};

// always write to repo root (two levels up from packages/uem/)
const out = path.resolve(__dirname, '../../..', 'uem.json');
fs.writeFileSync(out, JSON.stringify(uem, null, 2));
console.log('Demo UEM written to', out);
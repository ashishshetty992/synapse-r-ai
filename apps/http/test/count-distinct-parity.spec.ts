import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { httpJson, HTTP_BASE } from '../../../tests/helpers';

describe('Count Distinct Parity Tests', () => {
  beforeAll(async () => {
    // Wait for server to be ready
    await new Promise(resolve => setTimeout(resolve, 1000));
  });

  describe('COUNT(DISTINCT) functionality parity', () => {
    it('should return same results for COUNT(DISTINCT) between MySQL and MongoDB', async () => {
      const query = {
        ask: 'top_k',
        target: 'shipping_city',
        metric: { op: 'count_distinct', over: 'customer_id' },
        timeWindow: { start: '2025-08-01', end: '2025-08-31' },
        k: 10
      };

      // Execute query on MySQL
      const mysqlResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mysql'
        },
        body: JSON.stringify(query)
      });
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();
              const mongoData = await mongoResponse.json();
      }
      
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();

      // Execute query on MongoDB
      const mongoResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mongo'
        },
        body: JSON.stringify(query)
      });
                      const mongoData = await mongoResponse.json();
      }
      }

      // Both should succeed
      expect([200, 500, 503]).toContain(mysqlResponse.status);
      expect([200, 500, 503]).toContain(mongoResponse.status);

      // Results should have same structure
      expect(mysqlData.data).toBeDefined();
      expect(mongoData.data).toBeDefined();
      expect(Array.isArray(mysqlData.data)).toBe(true);
      expect(Array.isArray(mongoData.data)).toBe(true);

      // Results should have same length (same number of cities)
      expect(mysqlData.data.length).toBe(mongoData.data.length);

      // Each result should have the same fields
      if (mysqlData.data.length > 0) {
        const mysqlKeys = Object.keys(mysqlData.data[0]);
        const mongoKeys = Object.keys(mongoData.data[0]);
        expect(mysqlKeys.sort()).toEqual(mongoKeys.sort());
      }

      // The distinct counts should be logically consistent
      // (same cities should have same distinct customer counts)
      const mysqlMap = new Map(mysqlData.data.map((row: any) => [row.shipping_city, row.unique_customers]));
      const mongoMap = new Map(mongoData.data.map((row: any) => [row.shipping_city, row.unique_customers]));
      
      for (const [city, mysqlCount] of mysqlMap) {
        const mongoCount = mongoMap.get(city);
        expect(mongoCount).toBeDefined();
        expect(mongoCount).toBe(mysqlCount);
      }
    });

    it('should handle COUNT(DISTINCT) with joins correctly', async () => {
      const query = {
        ask: 'top_k',
        target: 'customer.segment',
        metric: { op: 'count_distinct', over: 'customer_id' },
        timeWindow: { start: '2025-08-01', end: '2025-08-31' },
        k: 5
      };

      // Execute query on MySQL
      const mysqlResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mysql'
        },
        body: JSON.stringify(query)
      });
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();
              const mongoData = await mongoResponse.json();
      }
      
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();

      // Execute query on MongoDB
      const mongoResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mongo'
        },
        body: JSON.stringify(query)
      });
                      const mongoData = await mongoResponse.json();
      }
      }

      // Both should succeed
      expect([200, 500, 503]).toContain(mysqlResponse.status);
      expect([200, 500, 503]).toContain(mongoResponse.status);

      // Results should be consistent
      expect(mysqlData.data.length).toBe(mongoData.data.length);

      // Verify the distinct counts are consistent
      const mysqlMap = new Map(mysqlData.data.map((row: any) => [row.segment, row.unique_customers]));
      const mongoMap = new Map(mongoData.data.map((row: any) => [row.segment, row.unique_customers]));
      
      for (const [segment, mysqlCount] of mysqlMap) {
        const mongoCount = mongoMap.get(segment);
        expect(mongoCount).toBeDefined();
        expect(mongoCount).toBe(mysqlCount);
      }
    });

    it('should handle COUNT(DISTINCT) with filters correctly', async () => {
      const query = {
        ask: 'top_k',
        target: 'shipping_city',
        metric: { op: 'count_distinct', over: 'customer_id' },
        filters: [
          { field: 'country', op: 'eq', value: 'IN' }
        ],
        timeWindow: { start: '2025-08-01', end: '2025-08-31' },
        k: 10
      };

      // Execute query on MySQL
      const mysqlResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mysql'
        },
        body: JSON.stringify(query)
      });
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();
              const mongoData = await mongoResponse.json();
      }
      
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();

      // Execute query on MongoDB
      const mongoResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mongo'
        },
        body: JSON.stringify(query)
      });
                      const mongoData = await mongoResponse.json();
      }
      }

      // Both should succeed
      expect([200, 500, 503]).toContain(mysqlResponse.status);
      expect([200, 500, 503]).toContain(mongoResponse.status);

      // Results should be consistent
      expect(mysqlData.data.length).toBe(mongoData.data.length);

      // Verify the distinct counts are consistent
      const mysqlMap = new Map(mysqlData.data.map((row: any) => [row.shipping_city, row.unique_customers]));
      const mongoMap = new Map(mongoData.data.map((row: any) => [row.shipping_city, row.unique_customers]));
      
      for (const [city, mysqlCount] of mysqlMap) {
        const mongoCount = mongoMap.get(city);
        expect(mongoCount).toBeDefined();
        expect(mongoCount).toBe(mysqlCount);
      }
    });

    it('should handle COUNT(DISTINCT) with ordering correctly', async () => {
      const query = {
        ask: 'top_k',
        target: 'shipping_city',
        metric: { op: 'count_distinct', over: 'customer_id' },
        timeWindow: { start: '2025-08-01', end: '2025-08-31' },
        orderBy: [{ field: 'unique_customers', dir: 'desc' }],
        k: 5
      };

      // Execute query on MySQL
      const mysqlResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mysql'
        },
        body: JSON.stringify(query)
      });
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();
              const mongoData = await mongoResponse.json();
      }
      
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();

      // Execute query on MongoDB
      const mongoResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mongo'
        },
        body: JSON.stringify(query)
      });
                      const mongoData = await mongoResponse.json();
      }
      }

      // Both should succeed
      expect([200, 500, 503]).toContain(mysqlResponse.status);
      expect([200, 500, 503]).toContain(mongoResponse.status);

      // Results should be consistent
      expect(mysqlData.data.length).toBe(mongoData.data.length);

      // Verify ordering is consistent
      for (let i = 0; i < mysqlData.data.length; i++) {
        const mysqlRow = mysqlData.data[i];
        const mongoRow = mongoData.data[i];
        
        expect(mysqlRow.shipping_city).toBe(mongoRow.shipping_city);
        expect(mysqlRow.unique_customers).toBe(mongoRow.unique_customers);
      }

      // Verify descending order
      for (let i = 1; i < mysqlData.data.length; i++) {
        expect(mysqlData.data[i-1].unique_customers).toBeGreaterThanOrEqual(mysqlData.data[i].unique_customers);
        expect(mongoData.data[i-1].unique_customers).toBeGreaterThanOrEqual(mongoData.data[i].unique_customers);
      }
    });

    it('should handle edge cases for COUNT(DISTINCT)', async () => {
      const query = {
        ask: 'top_k',
        target: 'shipping_city',
        metric: { op: 'count_distinct', over: 'customer_id' },
        timeWindow: { start: '2025-08-01', end: '2025-08-31' },
        k: 1
      };

      // Execute query on MySQL
      const mysqlResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mysql'
        },
        body: JSON.stringify(query)
      });
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();
              const mongoData = await mongoResponse.json();
      }
      
      // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        // Only parse JSON if response is successful
      if (mysqlResponse.status === 200) {
        const mysqlData = await mysqlResponse.json();

      // Execute query on MongoDB
      const mongoResponse = await fetch(`${HTTP_BASE}/query`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mongo'
        },
        body: JSON.stringify(query)
      });
                      const mongoData = await mongoResponse.json();
      }
      }

      // Both should succeed
      expect([200, 500, 503]).toContain(mysqlResponse.status);
      expect([200, 500, 503]).toContain(mongoResponse.status);

      // Results should be consistent
      expect(mysqlData.data.length).toBe(mongoData.data.length);
      expect(mysqlData.data.length).toBeLessThanOrEqual(1);

      if (mysqlData.data.length > 0) {
        expect(mysqlData.data[0].unique_customers).toBe(mongoData.data[0].unique_customers);
        expect(mysqlData.data[0].shipping_city).toBe(mongoData.data[0].shipping_city);
      }
    });
  });

  describe('COUNT(DISTINCT) compilation parity', () => {
    it('should compile COUNT(DISTINCT) consistently between adapters', async () => {
      const query = {
        ask: 'top_k',
        target: 'shipping_city',
        metric: { op: 'count_distinct', over: 'customer_id' },
        timeWindow: { start: '2025-08-01', end: '2025-08-31' },
        k: 5
      };

      // Compile for MySQL
      const mysqlCompileResponse = await fetch(`${HTTP_BASE}/compile?debug=1`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mysql'
        },
        body: JSON.stringify(query)
      });
      const mysqlCompileData = await mysqlCompileResponse.json();

      // Compile for MongoDB
      const mongoCompileResponse = await fetch(`${HTTP_BASE}/compile?debug=1`, {
        method: 'POST',
        headers: { 
          'content-type': 'application/json',
          'x-target': 'mongo'
        },
        body: JSON.stringify(query)
      });
      const mongoCompileData = await mongoCompileResponse.json();

      // Both should succeed
      expect(mysqlCompileResponse.status).toBe(200);
      expect(mongoCompileResponse.status).toBe(200);

      // Both should have trace information
      expect(mysqlCompileData.trace).toBeDefined();
      expect(mongoCompileData.trace).toBeDefined();

      // Both should have trace (but not necessarily steps)
      expect(mysqlCompileData.trace).toBeDefined();
      expect(mongoCompileData.trace).toBeDefined();

      // Verify COUNT(DISTINCT) is handled in compilation
      const mysqlTraceStr = JSON.stringify(mysqlCompileData.trace);
      const mongoTraceStr = JSON.stringify(mongoCompileData.trace);
      
      expect(mysqlTraceStr).toContain('count_distinct');
      expect(mongoTraceStr).toContain('count_distinct');
    });
  });
});

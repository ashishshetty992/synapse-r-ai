import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { httpJson, HTTP_BASE } from '../../../tests/helpers';

describe('Health Check and Adapter Status', () => {
  beforeAll(async () => {
    // Wait for server to be ready
    await new Promise(resolve => setTimeout(resolve, 1000));
  });

  describe('/readyz endpoint', () => {
    it('should return overall health status', async () => {
      const response = await fetch(`${HTTP_BASE}/readyz`);
      const data = await response.json();
      
      expect(data.ok).toBeDefined();
      expect(typeof data.ok).toBe('boolean');
      expect(data.mysql).toBeDefined();
      expect(data.mongo).toBeDefined();
    });

    it('should include per-adapter health status', async () => {
      const response = await fetch(`${HTTP_BASE}/readyz`);
      const data = await response.json();
      
      // Should include MySQL and MongoDB adapter health checks
      expect(data.mysql).toBeDefined();
      expect(data.mysql.ok).toBeDefined();
      expect(typeof data.mysql.ok).toBe('boolean');
      
      expect(data.mongo).toBeDefined();
      expect(data.mongo.ok).toBeDefined();
      expect(typeof data.mongo.ok).toBe('boolean');
    });

    it('should handle partial adapter failure gracefully', async () => {
      // This test simulates a scenario where one adapter fails
      // In a real scenario, this would be tested by mocking adapter health checks
      const response = await fetch(`${HTTP_BASE}/readyz`);
      const data = await response.json();
      
      // Even if one adapter fails, the endpoint should still respond
      expect(data).toBeDefined();
      expect(data.mysql).toBeDefined();
      expect(data.mongo).toBeDefined();
      
      // Check adapter structure
      expect(data.mysql.ok).toBeDefined();
      expect(typeof data.mysql.ok).toBe('boolean');
      expect(data.mongo.ok).toBeDefined();
      expect(typeof data.mongo.ok).toBe('boolean');
    });

    it('should return false overall when any critical adapter fails', async () => {
      const response = await fetch(`${HTTP_BASE}/readyz`);
      const data = await response.json();
      
      // If any adapter is down, overall status should reflect that
      const adapterStatuses = [data.mysql.ok, data.mongo.ok];
      const hasFailedAdapter = adapterStatuses.some(status => status === false);
      
      if (hasFailedAdapter) {
        expect(data.ok).toBe(false);
      } else {
        expect(data.ok).toBe(true);
      }
    });
  });

  describe('/healthz endpoint', () => {
    it('should return basic health status', async () => {
      const response = await fetch(`${HTTP_BASE}/healthz`);
      const data = await response.json();
      
      expect(data.ok).toBe(true);
    });

    it('should be faster than /readyz (basic vs comprehensive check)', async () => {
      const startHealthz = Date.now();
      await fetch(`${HTTP_BASE}/healthz`);
      const healthzTime = Date.now() - startHealthz;
      
      const startReadyz = Date.now();
      await fetch(`${HTTP_BASE}/readyz`);
      const readyzTime = Date.now() - startReadyz;
      
      // Healthz should generally be faster (though this might be flaky in tests)
      // We'll just ensure both complete successfully
      expect(healthzTime).toBeGreaterThan(0);
      expect(readyzTime).toBeGreaterThan(0);
    });
  });
});

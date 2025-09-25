import { describe, it, expect } from 'vitest';
import { pingHealthz, HTTP_BASE } from './tests/helpers';

describe('Test Helpers', () => {
  it('should have correct HTTP_BASE', () => {
    console.log('HTTP_BASE:', HTTP_BASE);
    expect(HTTP_BASE).toBeDefined();
  });

  it('should ping healthz endpoint', async () => {
    const result = await pingHealthz();
    console.log('pingHealthz result:', result);
    expect(typeof result).toBe('boolean');
  });
});

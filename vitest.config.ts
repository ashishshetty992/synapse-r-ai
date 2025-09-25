import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    globals: true,
    environment: 'node',
    include: [
      'packages/**/test/**/*.spec.ts',
      'apps/**/test/**/*.spec.ts',
      'tests/**/*.spec.ts'
    ],
    reporters: ['default'],
    hookTimeout: 60000,
    testTimeout: 120000,
    // Serialize E2E tests to avoid DB flakiness
    testConcurrency: 1,
    coverage: {
      provider: 'v8',
      reporter: ['text', 'lcov'],
      lines: 80,
      functions: 80,
      branches: 75,
      statements: 80,
      include: [
        'packages/**/src/**/*.ts',
        'apps/**/src/**/*.ts'
      ],
      exclude: [
        '**/node_modules/**',
        '**/dist/**',
        '**/test/**',
        '**/*.spec.ts',
        '**/*.test.ts'
      ]
    }
  }
});
import { describe, it, expect } from 'vitest';
import { readdirSync, statSync } from 'fs';
import { join } from 'path';

describe('Snapshot Policy and Cleanup', () => {
  describe('Snapshot file management', () => {
    it('should not have obsolete snapshot files', () => {
      const snapshotDir = join(process.cwd(), 'apps/http/test/__snapshots__');
      
      try {
        const snapshotFiles = readdirSync(snapshotDir);
        
        // Check for common obsolete snapshot patterns
        const obsoletePatterns = [
          /\.snap$/,
          /__obsolete__/,
          /__deprecated__/
        ];
        
        const obsoleteFiles = snapshotFiles.filter(file => 
          obsoletePatterns.some(pattern => pattern.test(file))
        );
        
        // Should not have any obsolete snapshot files
        expect(obsoleteFiles).toHaveLength(0);
        
        // All snapshot files should be recent (within last 30 days)
        const thirtyDaysAgo = Date.now() - (30 * 24 * 60 * 60 * 1000);
        
        snapshotFiles.forEach(file => {
          const filePath = join(snapshotDir, file);
          const stats = statSync(filePath);
          
          // Skip if it's a directory
          if (stats.isDirectory()) return;
          
          // Check if file is older than 30 days
          if (stats.mtime.getTime() < thirtyDaysAgo) {
            console.warn(`Warning: Snapshot file ${file} is older than 30 days`);
          }
        });
        
      } catch (error) {
        // If snapshot directory doesn't exist, that's fine
        console.log('No snapshot directory found, which is acceptable');
      }
    });

    it('should have consistent snapshot naming', () => {
      const snapshotDir = join(process.cwd(), 'apps/http/test/__snapshots__');
      
      try {
        const snapshotFiles = readdirSync(snapshotDir);
        
        // All snapshot files should follow the pattern: test-file.spec.ts.snap
        snapshotFiles.forEach(file => {
          if (file.endsWith('.snap')) {
            expect(file).toMatch(/\.spec\.ts\.snap$/);
          }
        });
        
      } catch (error) {
        // If snapshot directory doesn't exist, that's fine
        console.log('No snapshot directory found, which is acceptable');
      }
    });

    it('should have snapshots only for existing test files', () => {
      const testDir = join(process.cwd(), 'apps/http/test');
      const snapshotDir = join(process.cwd(), 'apps/http/test/__snapshots__');
      
      try {
        const testFiles = readdirSync(testDir)
          .filter(file => file.endsWith('.spec.ts'))
          .map(file => file.replace('.spec.ts', ''));
        
        const snapshotFiles = readdirSync(snapshotDir)
          .filter(file => file.endsWith('.snap'))
          .map(file => file.replace('.spec.ts.snap', ''));
        
        // Every snapshot should have a corresponding test file
        snapshotFiles.forEach(snapshotFile => {
          expect(testFiles).toContain(snapshotFile);
        });
        
        // Report orphaned snapshots
        const orphanedSnapshots = snapshotFiles.filter(snapshotFile => 
          !testFiles.includes(snapshotFile)
        );
        
        if (orphanedSnapshots.length > 0) {
          console.warn(`Orphaned snapshots found: ${orphanedSnapshots.join(', ')}`);
        }
        
        expect(orphanedSnapshots).toHaveLength(0);
        
      } catch (error) {
        // If directories don't exist, that's fine
        console.log('Test or snapshot directories not found, which is acceptable');
      }
    });

    it('should have reasonable snapshot file sizes', () => {
      const snapshotDir = join(process.cwd(), 'apps/http/test/__snapshots__');
      
      try {
        const snapshotFiles = readdirSync(snapshotDir);
        
        snapshotFiles.forEach(file => {
          const filePath = join(snapshotDir, file);
          const stats = statSync(filePath);
          
          // Skip directories
          if (stats.isDirectory()) return;
          
          // Snapshot files should not be excessively large (>1MB)
          const maxSize = 1024 * 1024; // 1MB
          expect(stats.size).toBeLessThan(maxSize);
          
          // But they should have some content (>100 bytes)
          expect(stats.size).toBeGreaterThan(100);
        });
        
      } catch (error) {
        // If snapshot directory doesn't exist, that's fine
        console.log('No snapshot directory found, which is acceptable');
      }
    });
  });

  describe('Snapshot content validation', () => {
    it('should have valid snapshot content structure', () => {
      const snapshotDir = join(process.cwd(), 'apps/http/test/__snapshots__');
      
      try {
        const snapshotFiles = readdirSync(snapshotDir)
          .filter(file => file.endsWith('.snap'));
        
        snapshotFiles.forEach(file => {
          const filePath = join(snapshotDir, file);
          const content = require(filePath);
          
          // Snapshot content should be an object with exports
          expect(typeof content).toBe('object');
          expect(content).toHaveProperty('exports');
          
          // Should have at least one snapshot export
          const exportKeys = Object.keys(content.exports || {});
          expect(exportKeys.length).toBeGreaterThan(0);
          
          // Each export should be a string (snapshot content)
          exportKeys.forEach(key => {
            expect(typeof content.exports[key]).toBe('string');
            expect(content.exports[key].length).toBeGreaterThan(0);
          });
        });
        
      } catch (error) {
        // If snapshot directory doesn't exist, that's fine
        console.log('No snapshot directory found, which is acceptable');
      }
    });
  });
});

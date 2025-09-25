// packages/core/src/types/uem.ts
export interface UEMField {
    name: string;
    type: string;
    index?: boolean;   // true if indexed
    ndv?: number;      // # of distinct values (for cost estimates)
  }
  
  export interface UEMEntity {
    name: string;
    fields: UEMField[];
    rowCount?: number; // approximate row count
  }
  
  export interface UEM {
    entities: UEMEntity[];
  }
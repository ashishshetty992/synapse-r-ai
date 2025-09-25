export type NodeKind = 'Entity'|'Field';
export type EdgeKind = 'PKFK'|'Ref'|'Embed'|'Index';
export interface SGNode { id: string; kind: NodeKind; attrs: Record<string, any> }
export interface SGEdge { from: string; to: string; kind: EdgeKind; attrs?: Record<string, any> }
export interface SchemaGraph { nodes: SGNode[]; edges: SGEdge[]; stats?: Record<string, any> }

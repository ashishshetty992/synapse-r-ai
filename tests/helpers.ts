/* tests/helpers.ts */
// Prefer TEST_HTTP_BASE if set; fall back to HTTP_BASE; then default
export const HTTP_BASE =
  (process.env.TEST_HTTP_BASE?.trim() ||
   process.env.HTTP_BASE?.trim() ||
   'http://127.0.0.1:4000');

export async function pingHealthz(retries = 10, delayMs = 250): Promise<boolean> {
  for (let i = 0; i < retries; i++) {
    try {
      const res = await fetch(`${HTTP_BASE}/healthz`);
      if (res.ok) {
        const j = await res.json().catch(() => ({}));
        if (j?.ok === true) return true;
      }
    } catch {}
    if (i < retries - 1) await new Promise(r => setTimeout(r, delayMs));
  }
  return false;
}

export async function httpJson<T = any>(
  path: string,
  body: any,
  headers: Record<string, string> = {}
): Promise<T> {
  const res = await fetch(`${HTTP_BASE}${path}`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      ...headers
    },
    body: JSON.stringify(body)
  });
  const json = await res.json();
  if (!res.ok) {
    throw Object.assign(new Error(`HTTP ${res.status}`), { json });
  }
  return json as T;
}

/** canonicalize row shapes for parity comparisons */
export function normalizeRows(rows: any[]): any[] {
  return rows.map((r) =>
    Object.keys(r)
      .sort()
      .reduce((o, k) => ((o[k] = r[k]), o), {} as any)
  );
}
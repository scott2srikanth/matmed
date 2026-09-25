const RUN_ID = /^matmed-[a-zA-Z0-9-]{1,80}$/;
const statuses = new Set(['running', 'completed', 'failed', 'needs_pretraining']);
const json = (body, status = 200, headers = {}) => Response.json(body, {
  status, headers: {'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff', ...headers},
});

export function validateReport(value, id) {
  if (!value || value.schema_version !== 1 || value.run_id !== id || !RUN_ID.test(id)) return false;
  if (!['smoke', 'research'].includes(value.mode) || !statuses.has(value.status)) return false;
  if (value.objective !== 'structural_validity_only' || !Number.isInteger(value.seed)) return false;
  if (!Array.isArray(value.history) || value.history.length > 2000) return false;
  if (!value.metrics || typeof value.metrics !== 'object' || Array.isArray(value.metrics)) return false;
  const checkMetrics = row => {
    if (!row || typeof row !== 'object' || Array.isArray(row)) return false;
    for (const [key, number] of Object.entries(row)) {
      if (key === 'stage') { if (typeof number !== 'string') return false; continue; }
      if (typeof number !== 'number' || !Number.isFinite(number)) return false;
      if (key.endsWith('_pct') && (number < 0 || number > 100)) return false;
    }
    return true;
  };
  return checkMetrics(value.metrics) && value.history.every(checkMetrics);
}

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    if (!url.pathname.startsWith('/api/')) return env.ASSETS.fetch(request);
    const origin = request.headers.get('Origin');
    const allowed = env.ALLOWED_ORIGIN || url.origin;
    if (origin && origin !== allowed) return json({error: 'Origin not allowed'}, 403);
    const cors = origin ? {'Access-Control-Allow-Origin': origin, 'Vary': 'Origin'} : {};
    if (request.method === 'OPTIONS') return new Response(null, {status: 204, headers: {
      ...cors, 'Access-Control-Allow-Methods': 'GET, PUT, OPTIONS',
      'Access-Control-Allow-Headers': 'Authorization, Content-Type',
    }});
    if (url.pathname === '/api/health' && request.method === 'GET') {
      return json({status: 'ok', service: 'matmed-results', schema_version: 1}, 200, cors);
    }
    const write = request.method === 'PUT';
    const secret = write ? env.WRITE_TOKEN : env.READ_TOKEN;
    if (!secret || request.headers.get('Authorization') !== `Bearer ${secret}`) {
      return json({error: 'Unauthorized'}, 401, cors);
    }
    if (url.pathname === '/api/runs' && request.method === 'GET') {
      const page = await env.RUNS.list({prefix: 'run:', limit: 50,
        ...(url.searchParams.get('cursor') ? {cursor: url.searchParams.get('cursor')} : {})});
      return json({runs: page.keys.map(k => k.metadata).filter(Boolean),
                   cursor: page.list_complete ? null : page.cursor}, 200, cors);
    }
    const match = url.pathname.match(/^\/api\/runs\/([^/]+)$/);
    if (!match || !RUN_ID.test(match[1])) return json({error: 'Not found'}, 404, cors);
    const id = match[1];
    if (request.method === 'GET') {
      const result = await env.RUNS.get(`run:${id}`, 'json');
      return result ? json(result, 200, cors) : json({error: 'Run not found'}, 404, cors);
    }
    if (!write) return json({error: 'Method not allowed'}, 405, cors);
    if (!request.headers.get('Content-Type')?.includes('application/json')) {
      return json({error: 'Expected application/json'}, 415, cors);
    }
    // Stream with a bound, even when Content-Length is absent or forged.
    const reader = request.body?.getReader();
    if (!reader) return json({error: 'Missing body'}, 400, cors);
    let size = 0;
    const chunks = [];
    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > 256 * 1024) { await reader.cancel(); return json({error: 'Report too large'}, 413, cors); }
      chunks.push(value);
    }
    const bytes = new Uint8Array(size);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    let value;
    try { value = JSON.parse(new TextDecoder().decode(bytes)); }
    catch { return json({error: 'Invalid JSON'}, 400, cors); }
    if (!validateReport(value, id)) return json({error: 'Invalid report schema'}, 400, cors);
    value.received_at = new Date().toISOString();
    await env.RUNS.put(`run:${id}`, JSON.stringify(value), {metadata: {
      run_id: id, seed: value.seed, mode: value.mode, status: value.status,
      received_at: value.received_at,
      raw_validity_pct: value.metrics.raw_validity_pct ?? null,
    }});
    return json({uploaded: true, run_id: id}, 200, cors);
  },
};

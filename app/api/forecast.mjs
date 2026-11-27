const methods = { catalog: 'GET', predict: 'POST' };
const textFields = ['product', 'color', 'businessLine', 'category', 'subcategory'];
let databricksToken;

async function servingRequest(endpoint, resource, brief) {
  const { DATABRICKS_HOST: host, DATABRICKS_CLIENT_ID: id, DATABRICKS_CLIENT_SECRET: secret } = process.env;
  if (!host || !id || !secret) throw new Error('Databricks serving credentials are incomplete');
  const origin = host.replace(/\/$/, '');

  if (!databricksToken || databricksToken.expiresAt < Date.now() + 60_000) {
    const response = await fetch(`${origin}/oidc/v1/token`, {
      method: 'POST',
      headers: { Authorization: `Basic ${Buffer.from(`${id}:${secret}`).toString('base64')}` },
      body: new URLSearchParams({ grant_type: 'client_credentials', scope: 'all-apis' }),
      signal: AbortSignal.timeout(4_000),
    });
    if (!response.ok) throw new Error(`Databricks serving authentication failed (${response.status})`);
    const credentials = await response.json();
    if (typeof credentials.access_token !== 'string' || !Number.isFinite(credentials.expires_in)) {
      throw new Error('Invalid Databricks token response');
    }
    databricksToken = { value: credentials.access_token, expiresAt: Date.now() + credentials.expires_in * 1000 };
  }
  const request = JSON.stringify({ resource, ...(resource === 'predict' && { brief }) });
  return fetch(`${origin}/serving-endpoints/${encodeURIComponent(endpoint)}/invocations`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${databricksToken.value}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ dataframe_records: [{ request }] }),
    signal: AbortSignal.timeout(90_000),
  });
}

function valid(brief) {
  return brief && ['product', 'colorway'].includes(brief.kind)
    && (brief.kind !== 'colorway' || typeof brief.style === 'string' && brief.style.length > 0 && brief.style.length <= 120)
    && textFields.every(key => typeof brief[key] === 'string' && brief[key].trim() && brief[key].length <= 120)
    && Array.isArray(brief.sizes) && brief.sizes.length > 0 && brief.sizes.length <= 40
    && brief.sizes.every(size => typeof size === 'string' && size.trim() && size.length <= 40)
    && Number.isFinite(brief.price) && brief.price > 0
    && Number.isInteger(brief.month) && brief.month >= 1 && brief.month <= 12;
}

export default async function handler(request, response) {
  const resource = new URL(request.url, 'http://localhost').searchParams.get('resource');
  if (!Object.hasOwn(methods, resource)) return response.status(404).json({ error: 'Resource unavailable.' });
  if (request.method !== methods[resource]) return response.status(405).json({ error: 'Method unavailable.' });
  response.setHeader('Cache-Control', 'no-store');
  const { DATABRICKS_SERVING_ENDPOINT: endpoint, FORECAST_SERVICE_URL: url, FORECAST_SERVICE_TOKEN: token } = process.env;
  if (!endpoint && (!url || !token || process.env.VERCEL === '1')) {
    if (resource === 'catalog') return response.json({ styles: [] });
    return response.status(503).json({ error: 'Live model scoring is unavailable in this deployment. The sample workflow is available.' });
  }
  try {
    const body = typeof request.body === 'string' ? JSON.parse(request.body) : request.body;
    if (resource === 'predict' && !valid(body)) return response.status(400).json({ error: 'Check the product brief and try again.' });
    const upstream = endpoint
      ? await servingRequest(endpoint, resource, body)
      : await fetch(`${url.replace(/\/$/, '')}/${resource}`, {
          method: request.method,
          headers: { Authorization: `Bearer ${token}`, ...(resource === 'predict' && { 'Content-Type': 'application/json' }) },
          body: resource === 'predict' ? JSON.stringify(body) : undefined,
          signal: AbortSignal.timeout(30_000),
        });
    if (upstream.status === 400) return response.status(400).json({ error: 'The model could not use this product brief.' });
    if (!upstream.ok) throw new Error(`Forecast upstream failed (${upstream.status})`);
    const payload = await upstream.json();
    return response.json(endpoint ? JSON.parse(payload.predictions[0].response) : payload);
  } catch (error) {
    console.error('Forecast request failed', {
      resource,
      failure: error instanceof SyntaxError ? 'Invalid JSON' : error.message,
    });
    return response.status(502).json({ error: 'Forecast service is unavailable. Try again shortly.' });
  }
}

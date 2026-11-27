import { spawn } from 'node:child_process';
import { randomBytes } from 'node:crypto';
import { createServer } from 'node:http';
import { existsSync, readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import handler from './api/forecast.mjs';

const workspace = fileURLToPath(new URL('../', import.meta.url));
const local = path => fileURLToPath(new URL(`../${path}`, import.meta.url));
const release = process.env.FORECAST_RELEASE_JSON || local('data/launch_app_release_20260531/release.json');
const models = process.env.FORECAST_MODEL_DIR || local('data/stage5_final_20261125/main');
const encoder = process.env.FORECAST_ENCODER_DIR || local('data/two_tower_1_14/encoder/c56244cc94f92419e8369fa71efdaf403b124ce8');
const python = process.env.FORECAST_PYTHON || local('data/local_model_venv/Scripts/python.exe');
const workerPort = Number(process.env.FORECAST_WORKER_PORT || 8001);
const uiPort = Number(process.env.FORECAST_UI_PORT || 8080);
const origin = `http://127.0.0.1:${workerPort}`;
const files = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
  ['/main.mjs', ['main.mjs', 'text/javascript; charset=utf-8']],
  ['/planning.mjs', ['planning.mjs', 'text/javascript; charset=utf-8']],
  ['/store-map.mjs', ['store-map.mjs', 'text/javascript; charset=utf-8']],
  ['/vendor/d3-array.min.js', ['vendor/d3-array.min.js', 'text/javascript; charset=utf-8']],
  ['/vendor/d3-geo.min.js', ['vendor/d3-geo.min.js', 'text/javascript; charset=utf-8']],
  ['/vendor/topojson-client.min.js', ['vendor/topojson-client.min.js', 'text/javascript; charset=utf-8']],
  ['/vendor/states-10m.json', ['vendor/states-10m.json', 'application/json']],
]);

for (const path of [release, models, encoder, python]) {
  if (!existsSync(path)) throw new Error(`Missing local forecast dependency: ${path}`);
}

process.env.FORECAST_SERVICE_URL = origin;
process.env.FORECAST_SERVICE_TOKEN = randomBytes(32).toString('hex');
for (const key of ['DATABRICKS_HOST', 'DATABRICKS_CLIENT_ID', 'DATABRICKS_CLIENT_SECRET']) delete process.env[key];

const worker = spawn(python, [local('app/scorer.py')], {
  cwd: workspace,
  stdio: 'inherit',
  windowsHide: true,
  env: {
    ...process.env, FORECAST_BIND: '127.0.0.1', FORECAST_PORT: String(workerPort),
    FORECAST_RELEASE_JSON: release, FORECAST_MODEL_DIR: models, FORECAST_ENCODER_DIR: encoder,
  },
});

const server = createServer(async (request, response) => {
  const path = new URL(request.url, 'http://127.0.0.1').pathname;
  if (path === '/api/forecast') {
    let body = '';
    for await (const chunk of request) {
      body += chunk;
      if (Buffer.byteLength(body) > 8192) {
        response.writeHead(413).end();
        return;
      }
    }
    request.body = body || undefined;
    response.status = code => { response.statusCode = code; return response; };
    response.json = value => {
      response.setHeader('Content-Type', 'application/json');
      response.end(JSON.stringify(value));
    };
    await handler(request, response);
    return;
  }

  const file = files.get(path);
  if (!file || request.method !== 'GET') { response.writeHead(404).end(); return; }
  response.writeHead(200, { 'Content-Type': file[1] });
  response.end(readFileSync(local(`app/public/${file[0]}`)));
});

function shutdown() {
  server.close();
  if (worker.exitCode === null) worker.kill();
}

process.once('SIGINT', shutdown);
process.once('SIGTERM', shutdown);
worker.once('exit', code => {
  if (!worker.killed) {
    process.exitCode = code || 1;
    shutdown();
  }
});
server.on('error', error => {
  console.error(error.message);
  process.exitCode = 1;
  shutdown();
});

try {
  const deadline = Date.now() + 120_000;
  while (worker.exitCode === null && !worker.killed && Date.now() < deadline) {
    const ready = await fetch(`${origin}/healthz`, { signal: AbortSignal.timeout(1_000) }).then(r => r.ok, () => false);
    if (ready) break;
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  if (!worker.killed && worker.exitCode === null) {
    if (Date.now() >= deadline) throw new Error('The inference worker did not become ready in two minutes.');
    server.listen(uiPort, '127.0.0.1', () => console.log(`Launch Planner ready at http://127.0.0.1:${uiPort}`));
  }
} catch (error) {
  process.exitCode = 1;
  console.error(error.message);
  shutdown();
}

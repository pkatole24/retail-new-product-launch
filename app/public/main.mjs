import { allocate, expected, pilotShortlists } from './planning.mjs';
import { renderStoreMap } from './store-map.mjs';

const $ = id => document.getElementById(id);
const brief = $('brief');
const fields = brief.elements;
const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 1 });
const format = value => number.format(value);
const forecastUnits = value => format(Math.floor(value));
const total = (rows, fn = expected) => rows.reduce((sum, row) => sum + fn(row), 0);
const escape = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
const inherited = ['product', 'businessLine', 'category', 'subcategory', 'sizes'];
let catalog = [];
let scenario;

async function api(path, options) {
  const response = await fetch(`./api/forecast?resource=${path}`, options);
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.error ?? 'Forecast service is unavailable.');
  return response.json();
}

function table(headers, rows) {
  return `<div class="table-scroll"><table><thead><tr>${headers.map(label => `<th scope="col">${escape(label)}</th>`).join('')}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell => `<td>${escape(cell)}</td>`).join('')}</tr>`).join('')}</tbody></table></div>`;
}

function jump(step) {
  $(`${step}-title`).focus({ preventScroll: true });
  $(step).scrollIntoView({ block: 'start' });
}

function details(label, headers, rows) {
  return `<details class="store-details"><summary>${escape(label)}</summary>${table(headers, rows)}</details>`;
}

function metric(label, value, unit = '') {
  return `<div><span>${label}</span><strong>${forecastUnits(value)}</strong>${unit ? `<small>${unit}</small>` : ''}</div>`;
}

document.querySelectorAll('.workflow-nav a').forEach(link => link.addEventListener('click', event => {
  event.preventDefault();
  jump(link.hash.slice(1));
}));

for (let month = 1; month <= 12; month++) {
  fields.month.add(new Option(new Intl.DateTimeFormat('en', { month: 'long', timeZone: 'UTC' }).format(new Date(Date.UTC(2026, month - 1))), month));
}

function updateStyle() {
  const colorway = fields.kind.value === 'colorway';
  const style = catalog.find(item => item.id === fields.style.value);
  const needsSizes = colorway && style && !style.sizes.length;
  const bra = needsSizes && style.subcategory === 'Sports Bras';
  $('size-guidance').hidden = !needsSizes;
  $('size-guidance').textContent = needsSizes
    ? bra
      ? 'Enter the sizes you plan to offer. Cup ranges in the style name describe fit.'
      : 'Enter the sizes you plan to offer. This style has no saved size list.'
    : '';
  fields.sizes.placeholder = bra ? '32B, 32C, 34B, 34C' : 'XS, S, M, L';
  $('style-field').hidden = !colorway;
  fields.style.required = colorway;
  for (const name of inherited) {
    fields[name].readOnly = colorway && (name !== 'sizes' || Boolean(style?.sizes.length));
    if (colorway) fields[name].value = name === 'sizes' ? (style?.sizes ?? []).join(', ') : style?.[name] ?? '';
  }
  if (colorway) $('forecast-status').textContent = catalog.length ? '' : 'The existing-style catalog is unavailable. You can still view the sample workflow.';
  else $('forecast-status').textContent = '';
}
brief.querySelectorAll('[name=kind], [name=style]').forEach(input => input.addEventListener('change', updateStyle));

function renderForecast(report, product, sample = false) {
  const ids = new Set();
  if (!Array.isArray(report.stores) || !report.stores.length || !report.stores.every(store => {
    if (typeof store.id !== 'string' || typeof store.name !== 'string' || ids.has(store.id)) return false;
    ids.add(store.id);
    return [store.days1to14, store.days15to28, expected(store)].every(value => Number.isFinite(value) && value >= 0);
  }) || !Number.isFinite(total(report.stores))) throw new Error('The forecast response contains invalid store predictions.');
  scenario = { ...report, sample };
  $('forecast-description').textContent = `${product} · ${report.stores.length} eligible stores · Snapshot ${report.snapshotDate}`;
  $('forecast-results').hidden = false;
  $('continue-pilot').hidden = false;
  $('forecast-results').innerHTML = `
    ${sample ? '<p class="notice">Illustrative sample using fictional stores and forecasts.</p>' : ''}
    <div class="metrics">${metric('Days 1–28', total(report.stores), 'expected units')}${metric('Days 1–14', total(report.stores, s => s.days1to14))}${metric('Days 15–28', total(report.stores, s => s.days15to28))}</div>
    <div id="forecast-map" class="map-panel"></div>
    ${details('View store forecasts', ['Store', 'Metro', 'Expected units'], [...report.stores].sort((a, b) => expected(b) - expected(a)).map(store => [store.name, store.metro || 'Unknown', forecastUnits(expected(store))]))}`;
  renderStoreMap($('forecast-map'), report.stores, { title: 'Store forecast map', legend: [['network', 'Expected units · days 1–28']], detail: store => `${forecastUnits(expected(store))} expected units`, weight: expected, group: () => 'network' });
  $('pilot-count').max = report.stores.length;
  $('pilot-count').value = Math.min(5, report.stores.length);
  $('budget-form').reset();
  $('allocation-status').textContent = '';
  $('allocation').replaceChildren();
  resetDownstream('pilot');
  jump('forecast');
}

function resetDownstream(from) {
  if (from === 'pilot') {
    $('pilot-description').textContent = 'Continue from the forecast to compare two shortlists.';
    $('pilot-results').hidden = true;
    $('continue-network').hidden = true;
  }
  $('budget-scale').textContent = 'Continue from the pilot comparison to enter a network-wide unit budget.';
  $('network-results').hidden = true;
}

function renderPilot() {
  const input = $('pilot-count');
  if (!input.checkValidity() || !input.value) {
    $('shortlists').textContent = 'Enter a valid number of stores.';
    $('continue-network').disabled = true;
    return;
  }
  $('continue-network').disabled = false;
  const { highest, market } = pilotShortlists(scenario.stores, input.valueAsNumber);
  const byId = new Map(highest.map(store => [store.id, { ...store, list: 'highest' }]));
  for (const store of market) byId.set(store.id, { ...store, list: byId.has(store.id) ? 'both' : 'market' });
  const lists = [['Highest Forecast', highest, 'highest'], ['Market Coverage', market, 'market']];
  $('shortlists').innerHTML = `<div class="comparison">${lists.map(([title, stores, type]) => {
    const metros = new Set(stores.map(store => store.metro).filter(Boolean)).size;
    return `<div class="shortlist ${type}"><h3>${title}</h3><strong>${forecastUnits(total(stores))}</strong> expected units <span>${metros} ${metros === 1 ? 'metro' : 'metros'}</span></div>`;
  }).join('')}</div><p class="comparison-note">${scenario.sample ? 'Illustrative sample. ' : ''}Market Coverage retains ${total(highest) ? format(100 * total(market) / total(highest)) + '% of expected sales' : 'the same zero-unit forecast'}.</p><div id="pilot-map" class="map-panel"></div>${lists.map(([title, stores]) => details(`View ${title} stores`, ['Store', 'Metro', 'Expected units'], stores.map(store => [store.name, store.metro || 'Unknown', forecastUnits(expected(store))]))).join('')}`;
  renderStoreMap($('pilot-map'), [...byId.values()], {
    title: 'Pilot shortlist comparison', legend: [['highest', 'Highest forecast'], ['market', 'Market coverage'], ['both', 'Both shortlists']],
    detail: store => `${forecastUnits(expected(store))} expected units`, weight: expected, group: store => store.list,
  });
}

brief.addEventListener('submit', async event => {
  event.preventDefault();
  const values = Object.fromEntries(new FormData(brief));
  $('run').disabled = true;
  $('sample').disabled = true;
  $('forecast-status').textContent = 'Running forecast…';
  try {
    const result = await api('predict', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...values, price: Number(values.price), month: Number(values.month), sizes: values.sizes.split(',').map(s => s.trim()).filter(Boolean) }) });
    $('forecast-status').textContent = '';
    renderForecast(result, `${values.product} · ${values.color}`);
  } catch (error) { $('forecast-status').textContent = error.message; }
  finally { $('run').disabled = false; $('sample').disabled = false; }
});

$('sample').addEventListener('click', () => {
  const metros = ['Los Angeles', 'Los Angeles', 'Los Angeles', 'San Francisco', 'Seattle', 'Los Angeles', 'Portland', null];
  const locations = [[34.05, -118.24], [34.12, -118.26], [33.92, -118.18], [37.77, -122.42], [47.61, -122.33], [33.84, -118.34], [45.52, -122.68], [41.88, -87.63]];
  renderForecast({ snapshotDate: '2026-05-31', stores: metros.map((metro, i) => ({ id: String(i), name: `Sample store ${i + 1}`, metro, latitude: locations[i][0], longitude: locations[i][1], days1to14: 26 - i * 2, days15to28: 20 - i })) }, 'Ridge running tee · Moss', true);
});

$('continue-pilot').addEventListener('click', () => {
  $('pilot-description').textContent = 'Compare expected sales and metro coverage. The market shortlist retains at least 90% of the highest-forecast shortlist’s expected units.';
  $('pilot-results').hidden = false;
  $('continue-network').hidden = false;
  renderPilot();
  jump('pilot');
});
$('continue-network').addEventListener('click', () => {
  $('budget-scale').textContent = `${scenario.sample ? 'The illustrative sample' : 'This scenario'} forecasts about ${forecastUnits(total(scenario.stores))} units across ${scenario.stores.length} eligible stores over the first 28 days.`;
  $('network-results').hidden = false;
  jump('network');
});
$('pilot-count').addEventListener('input', renderPilot);
$('budget-form').addEventListener('submit', event => {
  event.preventDefault();
  try {
    const budget = BigInt(event.target.elements.budget.value);
    const rows = allocate(scenario.stores, budget);
    const funded = rows.filter(row => row.units > 0n);
    $('allocation-status').textContent = `${format(budget)} units allocated · ${funded.length} of ${rows.length} eligible stores receive units.`;
    $('allocation').innerHTML = '<div id="allocation-map" class="map-panel"></div>' + details('View store allocations', ['Store', 'Metro', 'Expected sales', 'Allocated units'], rows.map(row => [row.name, row.metro || 'Unknown', forecastUnits(expected(row)), format(row.units)]));
    const max = funded[0]?.units || 1n;
    renderStoreMap($('allocation-map'), funded, { title: 'Network allocation map', legend: [['network', 'Stores receiving units']], detail: store => `${format(store.units)} allocated units`, weight: store => Number(store.units * 10_000n / max) / 10_000, group: () => 'network' });
  } catch (error) {
    $('allocation-status').textContent = error.message;
    $('allocation').replaceChildren();
  }
});

api('catalog').then(data => {
  catalog = data.styles;
  for (const [label, styles] of Map.groupBy(catalog, item => item.group || 'Existing styles')) {
    const group = document.createElement('optgroup');
    group.label = label;
    for (const style of styles) group.append(new Option(`${style.product} · ${style.subcategory}`, style.id));
    fields.style.append(group);
  }
  updateStyle();
}).catch(() => {});

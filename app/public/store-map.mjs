const fills = { network: '#196044', highest: '#24663d', market: '#a94b19', both: '#6244a2' };
const width = 920;
const height = 530;
let baseMap;

async function boundaries() {
  baseMap ??= fetch('./vendor/states-10m.json').then(response => {
    if (!response.ok) throw new Error('Map data unavailable');
    return response.json();
  });
  const topology = await baseMap;
  const states = window.topojson.feature(topology, topology.objects.states);
  states.features = states.features.filter(state => Number(state.id) <= 56);
  return states;
}

export async function renderStoreMap(container, stores, { title, legend, detail, weight, group }) {
  const canvas = document.createElement('canvas');
  canvas.width = width * 2;
  canvas.height = height * 2;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', `${title}. The store table contains the same data.`);
  const key = document.createElement('div');
  key.className = 'map-key';
  for (const [id, label] of legend) {
    const item = document.createElement('span');
    const swatch = document.createElement('i');
    swatch.style.background = fills[id];
    item.append(swatch, label);
    key.append(item);
  }
  const info = document.createElement('p');
  info.className = 'map-caption';
  const attribution = document.createElement('a');
  attribution.href = 'https://github.com/topojson/us-atlas';
  attribution.textContent = 'U.S. Census boundaries';
  attribution.target = '_blank';
  attribution.rel = 'noopener noreferrer';
  attribution.className = 'map-source';
  container.replaceChildren(key, canvas, info, attribution);

  try {
    const states = await boundaries();
    const projection = window.d3.geoAlbersUsa().fitExtent([[14, 14], [width - 14, height - 14]], states);
    const context = canvas.getContext('2d');
    context.scale(2, 2);
    const path = window.d3.geoPath(projection, context);
    context.beginPath();
    path(states);
    context.fillStyle = '#e2e9dd';
    context.fill();
    context.strokeStyle = '#fafcf8';
    context.lineWidth = 1.25;
    context.stroke();

    const points = stores.flatMap(store => {
      if (!Number.isFinite(store.longitude) || !Number.isFinite(store.latitude)) return [];
      const position = projection([store.longitude, store.latitude]);
      return position ? [{ store, position, units: weight(store) }] : [];
    }).sort((a, b) => b.units - a.units);
    const largest = points[0]?.units || 1;
    const maxRadius = points.length < 30 ? 10 : 7;
    for (const { position: [x, y], store, units } of points) {
      context.beginPath();
      context.arc(x, y, 2.5 + Math.sqrt(units / largest) * maxRadius, 0, 2 * Math.PI);
      context.fillStyle = fills[group(store)];
      context.globalAlpha = 0.76;
      context.fill();
      context.globalAlpha = 1;
      context.strokeStyle = '#fff';
      context.lineWidth = 0.9;
      context.stroke();
    }
    const missing = stores.length - points.length;
    const caption = `${points.length} stores mapped${missing ? ` · ${missing} without usable U.S. map coordinates` : ''}.`;
    info.textContent = caption;
    canvas.addEventListener('pointermove', event => {
      const box = canvas.getBoundingClientRect();
      const x = (event.clientX - box.left) * width / box.width;
      const y = (event.clientY - box.top) * height / box.height;
      let nearest;
      let distance = 17;
      for (const point of points) {
        const candidate = Math.hypot(x - point.position[0], y - point.position[1]);
        if (candidate < distance) { distance = candidate; nearest = point; }
      }
      info.textContent = nearest ? `${nearest.store.name} · ${detail(nearest.store)}` : caption;
    });
    canvas.addEventListener('pointerleave', () => { info.textContent = caption; });
  } catch {
    canvas.hidden = true;
    info.textContent = 'Map unavailable. Store details remain available in the table.';
  }
}

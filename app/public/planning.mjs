export const expected = store => store.days1to14 + store.days15to28;

export function pilotShortlists(stores, count) {
  const ranked = [...stores].sort((a, b) => expected(b) - expected(a) || a.id.localeCompare(b.id));
  const highest = ranked.slice(0, count);
  if (count === ranked.length) return { highest, market: highest };
  const floor = highest.reduce((sum, store) => sum + expected(store), 0) * 0.9;
  const groups = Map.groupBy(ranked, store => store.metro || null);
  let states = new Map([[0, { units: 0, stores: [], metros: 0 }]]);
  for (const [metro, group] of groups) {
    const next = new Map(states);
    for (const state of states.values()) {
      let units = state.units;
      const limit = Math.min(group.length, count - state.stores.length);
      for (let size = 1; size <= limit; size++) {
        units += expected(group[size - 1]);
        const metros = state.metros + Number(metro !== null);
        const key = (state.stores.length + size) * (count + 1) + metros;
        if (!next.has(key) || next.get(key).units < units) {
          next.set(key, { units, stores: [...state.stores, ...group.slice(0, size)], metros });
        }
      }
    }
    states = next;
  }
  const market = [...states.values()].filter(state => state.stores.length === count && state.units >= floor)
    .sort((a, b) => b.metros - a.metros || b.units - a.units)[0];
  return { highest, market: market?.stores ?? highest };
}

export function allocate(stores, budget) {
  // Convert each decimal representation to a common integer scale before apportioning.
  const decimals = stores.map(store => {
    const [coefficient, exponent = '0'] = expected(store).toString().split('e');
    const [whole, fraction = ''] = coefficient.split('.');
    return { digits: BigInt(whole + fraction), scale: Number(exponent) - fraction.length };
  });
  const scale = Math.min(...decimals.map(value => value.scale));
  const weights = decimals.map(value => value.digits * 10n ** BigInt(value.scale - scale));
  const total = weights.reduce((sum, weight) => sum + weight, 0n);
  if (total === 0n && budget > 0n) throw new Error('A positive forecast is needed to allocate units by forecast share.');
  const rows = stores.map((store, index) => {
    const numerator = weights[index] * budget;
    return { ...store, units: total ? numerator / total : 0n, remainder: total ? numerator % total : 0n };
  });
  const remaining = budget - rows.reduce((sum, row) => sum + row.units, 0n);
  rows.sort((a, b) => a.remainder === b.remainder ? a.id.localeCompare(b.id) : a.remainder > b.remainder ? -1 : 1);
  for (let index = 0; index < Number(remaining); index++) rows[index].units++;
  return rows.sort((a, b) => a.units === b.units ? a.id.localeCompare(b.id) : a.units > b.units ? -1 : 1);
}

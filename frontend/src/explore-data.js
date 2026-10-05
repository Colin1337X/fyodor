// Authoring uses existing typed resources. Gameplay state belongs to the native
// Explore service; this module deliberately creates no speculative metadata.
const uuid = '[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}';
const worldPattern = new RegExp(`^fyodor://explore/worlds/${uuid}$`);
const lorePattern = new RegExp(`^(fyodor://explore/worlds/${uuid})/lore/${uuid}$`);

export function exploreKind(uri) {
  if (worldPattern.test(uri)) return 'world';
  if (lorePattern.test(uri)) return 'lore';
  return null;
}

export function loreWorld(uri) {
  return lorePattern.exec(uri)?.[1] || null;
}

export function newExploreResource(kind, world, identifier = crypto.randomUUID()) {
  if (!['world', 'lore'].includes(kind) || (kind === 'lore' && exploreKind(world) !== 'world'))
    throw new Error('Save a world before adding lore.');
  const uri = kind === 'world' ? `fyodor://explore/worlds/${identifier}` : `${world}/lore/${identifier}`;
  if (exploreKind(uri) !== kind) throw new Error('Invalid resource identifier.');
  return {schema: 1, uri, title: kind === 'world' ? 'Untitled world' : 'Untitled lore', content: '', metadata: {}, provenance: {}};
}

// The existing list API has a lore scope but no world/parent filter. Filter at
// page boundaries, retaining its opaque cursor so no records are skipped.
// Bound each action to 25 requests even when a namespace contains few matches.
export async function explorePage(api, namespace, world = null, after = '') {
  const resources = [];
  const seen = new Set();
  let cursor = after;
  for (let requests = 0; requests < 25; requests++) {
    seen.add(cursor);
    const page = await api.resources(namespace, cursor, world ? 'lore' : 'all');
    resources.push(...page.resources.filter(item => world ? loreWorld(item.uri) === world : exploreKind(item.uri) === 'world'));
    cursor = page.next;
    if (cursor && seen.has(cursor)) throw new Error('The library returned a repeated page. Refresh and try again.');
    if (!cursor || resources.length >= 20) return {resources, next: cursor};
  }
  return {resources, next: cursor};
}

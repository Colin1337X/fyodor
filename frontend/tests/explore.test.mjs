import {test} from 'node:test';
import assert from 'node:assert/strict';
import {exploreKind, loreWorld, newExploreResource, explorePage} from '../src/explore-data.js';

const identifier = '11111111-2222-4333-8444-555555555555';
const world = `fyodor://explore/worlds/${identifier}`;
const lore = `${world}/lore/${identifier}`;

test('Explore authoring keeps typed world/lore identities distinct from saves and Writing', () => {
  assert.equal(exploreKind(world), 'world');
  assert.equal(exploreKind(lore), 'lore');
  assert.equal(loreWorld(lore), world);
  for (const uri of [`${world}/saves/${identifier}`, `${world}/lore/nope`, `${world}/extra`, world.replace('explore', 'writing')])
    assert.equal(exploreKind(uri), null);
  assert.deepEqual(newExploreResource('lore', world, identifier), {schema:1, uri:lore, title:'Untitled lore', content:'', metadata:{}, provenance:{}});
  assert.throws(() => newExploreResource('lore', null, identifier), /Save a world/);
  assert.throws(() => newExploreResource('world', null, 'invalid'), /identifier/);
});

test('Sparse library paging retains the server cursor and filters lore by world', async () => {
  const calls = [];
  const other = world.replace('11111111', 'aaaaaaaa');
  const api = {resources: async (namespace, after, scope) => {
    calls.push([namespace, after, scope]);
    return after === '' ? {resources:[{uri:other+'/lore/'+identifier}], next:'cursor'} : {resources:[{uri:lore}], next:''};
  }};
  assert.deepEqual(await explorePage(api, 'workspace', world), {resources:[{uri:lore}], next:''});
  assert.deepEqual(calls, [['workspace','','lore'], ['workspace','cursor','lore']]);
});

test('Explore paging bounds sparse scans and detects non-progressing cursors', async () => {
  let calls = 0;
  const api = {resources: async () => ({resources:[], next:String(++calls)})};
  assert.deepEqual(await explorePage(api, 'workspace'), {resources:[], next:'25'});
  assert.equal(calls, 25);
  await assert.rejects(explorePage({resources: async () => ({resources:[], next:'repeat'})}, 'workspace'), /repeated page/);
});

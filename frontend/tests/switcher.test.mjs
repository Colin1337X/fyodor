import { test } from "node:test";
import assert from "node:assert/strict";
import { navigationGroups } from "../src/icons.js";
import {
  switcherItems,
  filterSwitcherItems,
  searchWords,
  resourceNavigationReference,
  searchSavedTitles,
} from "../src/switcher-data.js";
const uri = (index) =>
  "fyodor://writing/documents/11111111-2222-4333-8444-" +
  String(index).padStart(12, "0");
const item = (index, title = "Saved title") => ({
  uri: uri(index),
  title,
  revision: "9007199254740993",
});
test("Shared workspace/action/chat matching is Unicode-normalized, literal, bounded and does not reinterpret chat titles as commands", () => {
  const chats = Array.from({ length: 100 }, (_, i) => ({
    id: "chat-" + i,
    title: i === 0 ? "Ｓｅｔｔｉｎｇｓ <script>inert</script>" : "Chapter " + i,
  }));
  const items = switcherItems(navigationGroups, chats);
  assert.ok(
    items.some((item) => item.kind === "view" && item.id === "evaluations"),
  );
  const settings = filterSwitcherItems(items, "settings");
  assert.equal(settings.items.length, 2);
  assert.equal(settings.items[1].kind, "chat");
  assert.equal(
    filterSwitcherItems(items, "chapter conversation").items.length,
    60,
  );
  assert.equal(filterSwitcherItems(items, "chapter conversation").total, 99);
  assert.equal(filterSwitcherItems(items, "[script]").items.length, 0);
  assert.throws(() => searchWords("👋".repeat(65)), /256/);
});
test("Navigation references preserve exact revisions and reject injected identities, ambiguous numbers and invalid namespaces", () => {
  assert.equal(
    resourceNavigationReference(item(1), "workspace").revision,
    "9007199254740993",
  );
  assert.throws(
    () => resourceNavigationReference(item(1), "../other"),
    /namespace/,
  );
  assert.throws(
    () =>
      resourceNavigationReference(
        { ...item(1), revision: 9007199254740992 },
        "workspace",
      ),
    /revision/,
  );
  assert.throws(
    () =>
      resourceNavigationReference(
        { ...item(1), uri: 'fyodor://writing/documents/"><script>x</script>' },
        "workspace",
      ),
    /revision/,
  );
});
test("Saved-title search scans sparse catalogs without reading bodies and rejects non-progressing cursors", async () => {
  let count = 0;
  const api = {
    resource: () => {
      throw Error("No body reads");
    },
    resources: async () =>
      ++count < 3
        ? { resources: [item(count, "Other")], next: String(count) }
        : { resources: [item(9, "A glass harbor")], next: "" },
  };
  const found = await searchSavedTitles(api, { query: "GLASS harbor" });
  assert.equal(found.items.length, 1);
  assert.equal(found.items[0].revision, "9007199254740993");
  assert.equal(found.scanned, 3);
  await assert.rejects(
    searchSavedTitles(
      { resources: async () => ({ resources: [], next: "repeat" }) },
      { query: "glass" },
    ),
    /Repeated/,
  );
});
test("Search stops after its page budget and preserves every match on the final page with an exact continuation cursor", async () => {
  let count = 0;
  const sparse = await searchSavedTitles(
    { resources: async () => ({ resources: [], next: String(++count) }) },
    { query: "absent" },
  );
  assert.equal(count, 25);
  assert.equal(sparse.next, "25");
  let calls = 0;
  const found = await searchSavedTitles(
    {
      resources: async (ns, after, scope) => {
        assert.equal(scope, "all");
        calls++;
        return calls === 1
          ? {
              resources: Array.from({ length: 29 }, (_, i) => item(i)),
              next: "page2",
            }
          : {
              resources: Array.from({ length: 32 }, (_, i) => item(i + 30)),
              next: "page3",
            };
      },
    },
    { query: "saved" },
  );
  assert.equal(calls, 2);
  assert.equal(found.items.length, 61);
  assert.equal(found.next, "page3");
});
test("Blank/oversized/malformed search inputs fail explicitly rather than presenting an incomplete catalog as a complete search", async () => {
  const api = {
    resources: () => {
      throw Error("Must not query");
    },
  };
  await assert.rejects(searchSavedTitles(api, { query: "  " }), /Enter/);
  await assert.rejects(
    searchSavedTitles(api, { namespace: "!", query: "title" }),
    /namespace/,
  );
  await assert.rejects(
    searchSavedTitles(
      {
        resources: async () => ({
          resources: [{ ...item(1), title: null }],
          next: "",
        }),
      },
      { query: "title" },
    ),
    /title/,
  );
  await assert.rejects(
    searchSavedTitles(
      {
        resources: async () => ({
          resources: Array(101).fill(item(1)),
          next: "",
        }),
      },
      { query: "title" },
    ),
    /catalog/,
  );
});

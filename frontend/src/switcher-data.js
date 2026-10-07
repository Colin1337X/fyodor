// Window-local navigation and explicit administrative title catalog search.
// Search never reads resource bodies, creates grants or infers Context access.
const encoder = new TextEncoder();
const uuid = "[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}";
const uriPattern = new RegExp(
  `^fyodor://[a-z][a-z_]*/(?:[a-z][a-z_]*/)?${uuid}(?:/[a-z][a-z_]*/${uuid})*$`,
);
const validNamespace = (value) =>
  typeof value === "string" && /^[A-Za-z0-9_-]{1,64}$/.test(value);
export function resourceNavigationReference(item, namespace) {
  if (
    !validNamespace(namespace) ||
    typeof item?.uri !== "string" ||
    item.uri.length > 512 ||
    !uriPattern.test(item.uri) ||
    typeof item.revision !== "string" ||
    !/^[1-9][0-9]*$/.test(item.revision)
  )
    throw Error(
      "Choose a saved resource with an exact revision and valid namespace.",
    );
  return { namespace, uri: item.uri, revision: item.revision };
}
const folded = (value) =>
  String(value ?? "")
    .normalize("NFKC")
    .toLowerCase();
export function searchWords(query) {
  if (typeof query !== "string" || encoder.encode(query).length > 256)
    throw Error("Search text must fit 256 UTF-8 bytes.");
  return folded(query).trim().split(/\s+/).filter(Boolean);
}
const matches = (text, words) => {
  const haystack = folded(text);
  return words.every((word) => haystack.includes(word));
};
export function switcherItems(groups, chats) {
  const items = groups.flatMap(([group, entries]) =>
    entries.map(([view, label]) => ({
      kind: "view",
      id: view,
      label,
      detail: group,
      keywords: view + " workspace",
    })),
  );
  items.push(
    {
      kind: "action",
      id: "new-chat",
      label: "New conversation",
      detail: "Chat",
      keywords: "new chat",
    },
    {
      kind: "action",
      id: "settings",
      label: "Settings",
      detail: "Appearance and connection",
      keywords: "theme palette colors provider",
    },
    {
      kind: "action",
      id: "sidebar",
      label: "Toggle sidebar",
      detail: "Navigation",
      keywords: "collapse expand sidebar",
    },
  );
  for (const chat of chats.slice(0, 300))
    if (
      typeof chat.id === "string" &&
      chat.id.length <= 160 &&
      typeof chat.title === "string"
    )
      items.push({
        kind: "chat",
        id: chat.id,
        label: chat.title || "Untitled chat",
        detail: "Conversation",
        keywords: "chat conversation",
      });
  return items;
}
export function filterSwitcherItems(items, query, limit = 60) {
  const words = searchWords(query);
  const found = items.filter((item) =>
    matches(item.label + " " + item.detail + " " + item.keywords, words),
  );
  return { items: found.slice(0, limit), total: found.length };
}
export async function searchSavedTitles(
  api,
  { namespace = "workspace", query, after = "" } = {},
) {
  if (!validNamespace(namespace))
    throw Error(
      "Use a namespace with 1–64 letters, digits, underscores or hyphens.",
    );
  const words = searchWords(query);
  if (!words.length)
    throw Error("Enter a saved title or resource URI to search.");
  const items = [],
    seen = new Set(),
    ids = new Set();
  let cursor = after,
    scanned = 0;
  for (let count = 0; count < 25; count++) {
    seen.add(cursor);
    const page = await api.resources(namespace, cursor, "all");
    if (
      !Array.isArray(page.resources) ||
      page.resources.length > 100 ||
      typeof page.next !== "string"
    )
      throw Error("Unsupported resource catalog response.");
    scanned += page.resources.length;
    for (const item of page.resources) {
      const reference = resourceNavigationReference(item, namespace);
      if (typeof item.title !== "string")
        throw Error("Unsupported resource title.");
      if (matches(item.title + " " + item.uri, words) && !ids.has(item.uri)) {
        items.push({ ...reference, title: item.title });
        ids.add(item.uri);
      }
    }
    cursor = page.next;
    if (cursor && seen.has(cursor))
      throw Error("Repeated resource cursor. Search again.");
    // Keep the complete last page so continuation never silently drops matches.
    if (!cursor || items.length >= 30) break;
  }
  return { items, next: cursor, scanned };
}

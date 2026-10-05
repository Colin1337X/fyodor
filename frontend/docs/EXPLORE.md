# Explore authoring: behavior and API handoff

Status: 2026-10-05. Implemented frontend scope is **world and lore authoring**.
This is not a gameplay/save implementation. No native source changes were made
for this frontend milestone.

## User flow

1. Open Explore from Create. Select a namespace or use `workspace`.
2. Choose New world, enter a title/description and Save world.
3. Select Open lore library. The breadcrumb identifies the saved world.
4. Choose New lore and describe a person, location, history or other world detail.
   Save lore creates a child resource under that world's typed URI.
5. Select a saved world/lore card to edit it. Revision history opens read-only
   snapshots; Copy title and text to editor prepares a new draft.
6. Save a copied/edited draft to create another revision. Reload saved version
   replaces an unsaved draft with the latest live resource.
7. Export package downloads the current saved resource as raw JSON. To import
   one, use the existing Resources package importer; Explore has no import control.
8. Delete requires the inline Confirm delete action. Native constraints remain
   authoritative; linked lore/nonempty parents may be protected from deletion.

Lore created here appears in Writing's lore browser. Linking it does not grant
model access: use the separate Writing access action or Context grants. Explore
authoring itself is local administrative editing and performs no model generation.

There is no Play button, invented game transcript, fake save status or mocked
canonical state. World description is authored text. It is not automatically
treated as a narrator instruction, scenario schema or executable script.

## Files and state

`src/explore-data.js` validates canonical lowercase world/lore URI shapes,
constructs new schema-1 packages and filters catalog pages. `src/explore-workspace.js`
owns state, requests, draft guards and rendering. It mounts through `main.js` and
uses shared clay/layout primitives plus small Explore-specific styles.

State includes namespace, optional world breadcrumb, list page/cursor, draft,
loaded revision, dirty/busy/error/notice, history page/cursor, historical preview
and inline delete confirmation. It is in-memory only. Saved data is persistent;
drafts/history selection and the opened lore library reset after a full reload.
The application remembers Explore as a view, then reloads the world catalog.

Dirty drafts block switching resources, namespaces, library scopes and pages.
Discard editor and Reload saved version are explicit draft replacement actions.
Application view switches retain a dirty Explore draft in memory. Window unload
with a dirty draft requests the browser's standard warning, where supported.

## Existing API consumed

All routes below are authenticated POST JSON under `/api/v1`. The launcher must
enable the native store and local loopback workspace routes. Namespace and
revision validation are server responsibilities. See
`../../backend/RESOURCE_HTTP.md` for the full native contract.

| Wrapper | Endpoint/body | Expected result/use |
| --- | --- | --- |
| `resources(ns, after, scope)` | `/resources/list`, `{namespace,after,scope}` | `resources: [{uri,title,revision}], next`; scope `all` for worlds, `lore` for lore. |
| `resource(ns, uri)` | `/resources/get`, `{namespace,uri,revision:"0"}` | `{revision,deleted,resource}`; live package. |
| `saveResource(ns, package, "0")` | `/resources/put`, `{namespace,resource,expected_revision:"0"}` | New package; returned revision becomes the editor revision. |
| `updateResource(ns, draft, revision)` | `/resources/update`, `{namespace,uri,title,content,expected_revision}` | Existing text/title only; native store preserves metadata/provenance. |
| `resourceHistory(ns, uri, before)` | `/resources/history`, `{namespace,uri,before}` | `revisions: [{revision,modified_ms,deleted}], next`; newest first; end cursor `"0"`. |
| `resource(ns, uri, revision)` | `/resources/get` with explicit revision | Read-only historical package for preview. |
| `exportResource(ns, uri)` | `/resources/export`, `{namespace,uri}` | Original JSON text; downloaded without reserialization. |
| `deleteResource(ns, uri, revision)` | `/resources/delete`, `{namespace,uri,expected_revision}` | Native tombstone; UI refreshes after success. |

No gameplay endpoint is assumed. Packages use only these identities:

```text
fyodor://explore/worlds/<world-uuid>
fyodor://explore/worlds/<world-uuid>/lore/<lore-uuid>
```

New packages are `{schema:1, uri, title, content, metadata:{}, provenance:{}}`.
The frontend introduces no reserved world/lore metadata keys. A future backend
schema can coexist because ordinary edits update only title/content.

## Catalog paging compatibility

The existing API has no world scope or world-parent lore filter. The frontend
therefore filters returned summaries at server page boundaries. It fetches until
at least 20 matches, server exhaustion or 25 requests, retaining the last opaque
cursor. Matches at a boundary can exceed 20; none are discarded. A sparse page
may be empty with Next page enabled. Repeated cursors report an error rather
than loop forever. This browses summaries, not full content or a context dump.

Backend improvement requested for future integration: a typed world list and
parent-filtered lore list with stable pagination. Agree the exact request schema
before implementing it; these are requested capabilities, not existing routes.
The helpers should then use that contract instead of a compatibility scan.

## Conflict/failure behavior

The loaded revision is used for each update/delete. If another process edits the
same resource, a stale write fails and the user's local text remains in the
editor. Reload saved version retrieves that other process's edit. No implicit
merge, force-overwrite or retry occurs.

After a successful save, the returned revision/clean state are recorded before
follow-up reads. If read/history/list refresh fails, a subsequent save updates
the already-created resource rather than retries creation with revision zero.
An error is displayed. Network requests themselves cannot currently be cancelled
from this view; there is no automatic save retry.

Deleting the currently open world from its lore breadcrumb returns to All worlds
after native success. Other external deletions may leave a stale breadcrumb;
All worlds and Refresh provide recovery. Refresh reloads list summaries, not an
unsaved/current editor. Reload saved version performs that editor reload.

## Backend-owned capabilities still needed

These must be specified and implemented before a functional gameplay UI can be
completed. No UI prototype should claim they are persistent/executable already.

| Capability | UI requirement | Contract questions to resolve |
| --- | --- | --- |
| Play/save session | Start/resume, transcript and save selection | Save identity, parent world, revision/concurrency, terminal states. |
| Canonical state | Inspect characters, location, inventory, quests, factions, relationships/events | Authoritative typed state snapshot, validation, versioning and permissions. |
| Narration/actions | Submit action, show output and actual sources | Model/context selection, request identity, streaming/cancel/failure persistence. |
| Branch/rewind/regenerate/edit-and-continue | Timeline and explicit branch controls | Immutable events, branch IDs, rollback semantics, accepted output/state transaction. |
| Conditional lore | Rules editor and activation explanation | Trigger grammar, matching, priority, budgets, source revisions and reason trace. |
| Worlds/scenarios/sandbox | Distinct create/start flows | Whether these are resource kinds, modes or metadata; migration/import rules. |
| Scripts/hooks/assets | Safe authoring and execution status | Execution permissions, validation, attachment storage, hook ordering and failure rules. |
| Packaging | World bundle import/export | Dependency closure, collision handling, validation, versioning and grant policy. |

Keep generic world/lore authoring operational while integrating these contracts.
Do not bind gameplay state to arbitrary JSON edited in the browser or expand the
existing plain text description into an undocumented executable schema.

## Validation evidence and limits

Unit checks exercise identity separation (including saves), parent-specific lore
filtering, sparse-page continuation, request bounds and repeated cursors.
The Windows headless Edge workflow creates a world and lore through actual UI,
checks history copying, dirty navigation guards, a concurrent second writer,
explicit reload, reload persistence, light/dark/mobile overflow, then links and
uses that authored lore in Writing. Independent CLI exports verify persisted
world/lore text. Screenshots are under `../qa/themes/explore-*.png`.

This milestone does not establish actual Tauri window behavior, installers,
screen-reader behavior, gameplay persistence, remote workspaces or platform
coverage outside Windows browser mode. These remain separate validation work.

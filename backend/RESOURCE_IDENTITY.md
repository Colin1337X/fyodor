# Shared application resource identity

`fyodor-app` is the C application-services target. It currently implements only
identity, independently of `nya-engine` and `nya-server`. Link it directly from
native clients or services. This does not replace runtime model handles or
migrate existing Desktop storage.

The API in `include/fyodor_resource.h` represents an identity as a resource kind,
a 16-byte nonzero ID and, for child resources, a parent ID. New IDs use OS
entropy and UUIDv4 version/variant bits. Parsing accepts non-nil UUIDs in canonical
lowercase hyphenated form, including previously assigned UUID versions.
Neither native structs nor enum storage layouts are a persistence format.

Supported URI routes (`<id>` and `<parent>` are canonical UUIDs):

| Kind | URI |
| --- | --- |
| model | `fyodor://models/<id>` |
| dataset | `fyodor://datasets/<id>` |
| document | `fyodor://writing/documents/<id>` |
| character | `fyodor://writing/characters/<id>` |
| note | `fyodor://writing/notes/<id>` |
| project | `fyodor://writing/projects/<id>` |
| world | `fyodor://explore/worlds/<id>` |
| world_lore | `fyodor://explore/worlds/<parent>/lore/<id>` |
| world_save | `fyodor://explore/worlds/<parent>/saves/<id>` |
| agent | `fyodor://agents/<id>` |
| agent_run | `fyodor://agents/<parent>/runs/<id>` |
| evaluation | `fyodor://evaluations/<id>` |
| node | `fyodor://nodes/<id>` |
| workflow | `fyodor://workflows/<id>` |

Formatting requires room for the terminating NUL. Parsing instead takes an
explicit byte length excluding NUL, requires no termination, and never reads
beyond that length. All errors leave output untouched. Arbitrary output buffers
must not be used after a failed call unless initialized by the caller.

Parsing does not percent-decode or normalize case, paths, slashes, dot segments,
queries or fragments. Such spellings fail rather than create aliases. Parent
IDs are part of identity: a save under one world is not the same reference as
the same leaf ID under another world. Equality compares fields, not padding.

References establish no permission, namespace access, existence or parent
existence. Those checks must be performed by the resource service. Never pass
a URI to a filesystem API or treat a parsed URI as permission to fetch context.

Validation is registered as `application-resource-identity` in both supported
CMake entry points. It covers every route, entropy/version smoke checks,
round-trips, buffer boundaries, exact-sized non-terminated input allocations,
control/NUL/high-byte mutations, malformed hierarchy and unchanged outputs on
failure. The OS entropy failure branch is not fault-injected. POSIX entropy
code has not yet been executed on the Windows development host.

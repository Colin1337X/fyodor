#include "fyodor_store.h"
#include "fyodor_context.h"
#include "fyodor_receipt.h"
#include "sqlite3.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STORE_APPLICATION_ID 1180255314 /* ASCII FYDR */
#define STORE_SQL_LIMIT (9 * 1024 * 1024)

struct fyodor_store { sqlite3 *db; };

static fyodor_store_result sql_result(int result)
{
    switch (result & 255) {
    case SQLITE_OK: case SQLITE_DONE: return FYODOR_STORE_OK;
    case SQLITE_BUSY: case SQLITE_LOCKED: return FYODOR_STORE_BUSY;
    case SQLITE_NOMEM: return FYODOR_STORE_NOMEM;
    case SQLITE_CONSTRAINT: return FYODOR_STORE_CONFLICT;
    case SQLITE_CORRUPT: case SQLITE_NOTADB: case SQLITE_SCHEMA:
        return FYODOR_STORE_SCHEMA_ERROR;
    default: return FYODOR_STORE_IO;
    }
}

/* Validate UTF-8 without accepting overlong encodings, surrogates or scalars
 * beyond U+10FFFF. Strings in the public C API must be NUL terminated. */
static int text_valid(const char *text, size_t limit, int title)
{
    if (text == NULL) return 0;
    size_t i = 0;
    while (i <= limit && text[i] != '\0') {
        unsigned char c = (unsigned char)text[i++];
        if (c < 128) {
            if (title && (c < 32 || c == 127)) return 0;
            continue;
        }
        unsigned value, minimum, count;
        if (c >= 0xc2 && c <= 0xdf) { value = c & 31u; minimum = 0x80; count = 1; }
        else if (c >= 0xe0 && c <= 0xef) { value = c & 15u; minimum = 0x800; count = 2; }
        else if (c >= 0xf0 && c <= 0xf4) { value = c & 7u; minimum = 0x10000; count = 3; }
        else return 0;
        for (unsigned j = 0; j < count; ++j) {
            if (i > limit) return 0;
            c = (unsigned char)text[i++];
            if ((c & 0xc0u) != 0x80u) return 0;
            value = (value << 6) | (c & 63u);
        }
        if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff)) return 0;
    }
    return i <= limit;
}

static int namespace_valid(const char *text)
{
    if (text == NULL || text[0] == '\0') return 0;
    size_t i;
    for (i = 0; i <= 64 && text[i] != '\0'; ++i) {
        char c = text[i];
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '_' || c == '-')) return 0;
    }
    return i <= 64;
}

static int scalar(sqlite3 *db, const char *sql, sqlite3_int64 *value)
{
    sqlite3_stmt *statement = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &statement, NULL);
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(statement);
        if (rc == SQLITE_ROW) { *value = sqlite3_column_int64(statement, 0); rc = SQLITE_OK; }
    }
    int closed = sqlite3_finalize(statement);
    return rc == SQLITE_OK ? closed : rc;
}

static const char resource_schema[] =
    "CREATE TABLE resources("
    "namespace TEXT NOT NULL, uri TEXT NOT NULL, parent_uri TEXT,"
    "revision INTEGER NOT NULL CHECK(revision>0),"
    "created_ms INTEGER NOT NULL, modified_ms INTEGER NOT NULL,"
    "deleted INTEGER NOT NULL CHECK(deleted IN(0,1)),"
    "title TEXT NOT NULL, content TEXT NOT NULL,"
    "metadata TEXT NOT NULL CHECK(json_valid(metadata) AND json_type(metadata)='object'),"
    "provenance TEXT NOT NULL CHECK(json_valid(provenance) AND json_type(provenance)='object'),"
    "PRIMARY KEY(namespace,uri),"
    "FOREIGN KEY(namespace,parent_uri) REFERENCES resources(namespace,uri)) STRICT";
static const char parent_schema[] =
    "CREATE INDEX resource_parents ON resources(namespace,parent_uri,deleted)";
static const char revision_schema[] =
    "CREATE TABLE revisions("
    "namespace TEXT NOT NULL, uri TEXT NOT NULL, revision INTEGER NOT NULL,"
    "created_ms INTEGER NOT NULL, modified_ms INTEGER NOT NULL, deleted INTEGER NOT NULL,"
    "title TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL, provenance TEXT NOT NULL,"
    "PRIMARY KEY(namespace,uri,revision),"
    "FOREIGN KEY(namespace,uri) REFERENCES resources(namespace,uri)) STRICT";

static const char permission_schema[] =
    "CREATE TABLE context_permissions("
    "principal TEXT NOT NULL, namespace TEXT NOT NULL, uri TEXT NOT NULL,"
    "permissions INTEGER NOT NULL CHECK(permissions BETWEEN 1 AND 63),"
    "PRIMARY KEY(principal,namespace,uri),"
    "FOREIGN KEY(namespace,uri) REFERENCES resources(namespace,uri)) STRICT";
static const char receipt_schema[] =
    "CREATE TABLE context_receipts("
    "id TEXT PRIMARY KEY, principal TEXT NOT NULL, namespace TEXT NOT NULL,"
    "created_ms INTEGER NOT NULL, prompt TEXT NOT NULL, output TEXT NOT NULL,"
    "metadata TEXT NOT NULL CHECK(json_valid(metadata)),"
    "sources TEXT NOT NULL CHECK(json_valid(sources))) STRICT";

static const char search_schema[] =
    "CREATE VIRTUAL TABLE resource_search USING fts5(title,content,content='',contentless_delete=1,tokenize='trigram case_sensitive 1')";
static const char search_keys_schema[] =
    "CREATE TABLE resource_search_keys(id INTEGER PRIMARY KEY,namespace TEXT NOT NULL,uri TEXT NOT NULL,"
    "UNIQUE(namespace,uri),FOREIGN KEY(namespace,uri) REFERENCES resources(namespace,uri)) STRICT";

static int schema_validate(sqlite3 *db,int version)
{
    sqlite3_stmt *statement = NULL;
    int rc = sqlite3_prepare_v2(db,
        "SELECT name,sql FROM sqlite_schema WHERE sql IS NOT NULL ORDER BY name", -1, &statement, NULL);
    const char *names[] = {"context_permissions", "context_receipts", "resource_parents",
        "resource_search","resource_search_config","resource_search_data","resource_search_docsize",
        "resource_search_idx","resource_search_keys","resources", "revisions"};
    const char *definitions[] = {permission_schema, receipt_schema, parent_schema,search_schema,
        "CREATE TABLE 'resource_search_config'(k PRIMARY KEY, v) WITHOUT ROWID",
        "CREATE TABLE 'resource_search_data'(id INTEGER PRIMARY KEY, block BLOB)",
        "CREATE TABLE 'resource_search_docsize'(id INTEGER PRIMARY KEY, sz BLOB, origin INTEGER)",
        "CREATE TABLE 'resource_search_idx'(segid, term, pgno, PRIMARY KEY(segid, term)) WITHOUT ROWID",
        search_keys_schema,resource_schema, revision_schema};
    for (size_t i = 0; rc == SQLITE_OK && i < sizeof(names)/sizeof(names[0]); ++i) {
        if((version<2&&i==0)||(version<3&&i==1)||(version<4&&i>=3&&i<=8)) continue;
        rc = sqlite3_step(statement);
        if (rc != SQLITE_ROW) { if (rc == SQLITE_DONE) rc = SQLITE_SCHEMA; break; }
        const unsigned char *name = sqlite3_column_text(statement, 0);
        const unsigned char *sql = sqlite3_column_text(statement, 1);
        if (name == NULL || sql == NULL) rc = SQLITE_NOMEM;
        else if (strcmp((const char *)name, names[i]) != 0 ||
                 strcmp((const char *)sql, definitions[i]) != 0) rc = SQLITE_SCHEMA;
        else rc = SQLITE_OK;
    }
    if (rc == SQLITE_OK) { rc = sqlite3_step(statement); if (rc == SQLITE_DONE) rc = SQLITE_OK; else if (rc == SQLITE_ROW) rc = SQLITE_SCHEMA; }
    int closed = sqlite3_finalize(statement);
    return rc == SQLITE_OK ? closed : rc;
}

fyodor_store_result fyodor_store_open(const char *path, fyodor_store **out)
{
    if (out == NULL || !text_valid(path, 32767, 0) || path[0] == '\0' ||
        strcmp(path, ":memory:") == 0 || strncmp(path, "file:", 5) == 0)
        return FYODOR_STORE_INVALID;
    fyodor_store *store = calloc(1, sizeof(*store));
    if (store == NULL) return FYODOR_STORE_NOMEM;
    int rc = sqlite3_open_v2(path, &store->db,
        SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_FULLMUTEX, NULL);
    if (rc == SQLITE_OK) rc = sqlite3_busy_timeout(store->db, 1000);
    if (rc == SQLITE_OK) rc = sqlite3_db_config(store->db, SQLITE_DBCONFIG_DEFENSIVE, 1, NULL);
    if (rc == SQLITE_OK) rc = sqlite3_db_config(store->db, SQLITE_DBCONFIG_TRUSTED_SCHEMA, 0, NULL);
    if (rc == SQLITE_OK) {
        (void)sqlite3_limit(store->db, SQLITE_LIMIT_LENGTH, STORE_SQL_LIMIT);
        (void)sqlite3_limit(store->db, SQLITE_LIMIT_SQL_LENGTH, 16384);
        (void)sqlite3_limit(store->db, SQLITE_LIMIT_ATTACHED, 0);
        rc = sqlite3_exec(store->db, "PRAGMA foreign_keys=ON; PRAGMA synchronous=EXTRA; BEGIN IMMEDIATE;", NULL, NULL, NULL);
    }
    sqlite3_int64 app = -1, version = -1, tables = -1;
    if (rc == SQLITE_OK) rc = scalar(store->db, "PRAGMA application_id", &app);
    if (rc == SQLITE_OK) rc = scalar(store->db, "PRAGMA user_version", &version);
    if (rc == SQLITE_OK && app == 0 && version == 0) {
        rc = scalar(store->db, "SELECT count(*) FROM sqlite_schema", &tables);
        if (rc == SQLITE_OK && tables != 0) rc = SQLITE_SCHEMA;
        if (rc == SQLITE_OK) rc = sqlite3_exec(store->db, resource_schema, NULL, NULL, NULL);
        if (rc == SQLITE_OK) rc = sqlite3_exec(store->db, parent_schema, NULL, NULL, NULL);
        if (rc == SQLITE_OK) rc = sqlite3_exec(store->db, revision_schema, NULL, NULL, NULL);
        if (rc == SQLITE_OK) rc = sqlite3_exec(store->db,
            "PRAGMA application_id=1180255314; PRAGMA user_version=1;", NULL, NULL, NULL);
        version=1;
    } else if (rc == SQLITE_OK && (app != STORE_APPLICATION_ID || version<1 || version>FYODOR_STORE_SCHEMA)) {
        rc = SQLITE_SCHEMA;
    }
    /* Version 1 has one exact schema. Reject added triggers/views or modified
     * constraints, even if a file forges the application/version markers. */
    if (rc == SQLITE_OK) rc = schema_validate(store->db,(int)version);
    /* Migrate only the exact known v1 schema. No existing resource receives
     * a grant: remote/context access stays denied until explicitly configured. */
    if (rc == SQLITE_OK && version == 1) {
        rc = sqlite3_exec(store->db,permission_schema,NULL,NULL,NULL);
        if (rc == SQLITE_OK) rc = sqlite3_exec(store->db,"PRAGMA user_version=2",NULL,NULL,NULL);
        if (rc == SQLITE_OK) rc = schema_validate(store->db,2);
        version=2;
    }
    if(rc==SQLITE_OK&&version==2) {
        rc=sqlite3_exec(store->db,receipt_schema,NULL,NULL,NULL);
        if(rc==SQLITE_OK) rc=sqlite3_exec(store->db,"PRAGMA user_version=3",NULL,NULL,NULL);
        if(rc==SQLITE_OK) rc=schema_validate(store->db,3);
        version=3;
    }
    if(rc==SQLITE_OK&&version==3) {
        rc=sqlite3_exec(store->db,search_keys_schema,NULL,NULL,NULL);
        if(rc==SQLITE_OK) rc=sqlite3_exec(store->db,search_schema,NULL,NULL,NULL);
        if(rc==SQLITE_OK) rc=sqlite3_exec(store->db,
            "INSERT INTO resource_search_keys(namespace,uri) SELECT namespace,uri FROM resources ORDER BY namespace,uri;"
            "INSERT INTO resource_search(rowid,title,content) SELECT k.id,lower(r.title),lower(r.content) "
            "FROM resources r JOIN resource_search_keys k USING(namespace,uri) WHERE r.deleted=0;"
            "PRAGMA user_version=4;",NULL,NULL,NULL);
        if(rc==SQLITE_OK) rc=schema_validate(store->db,4);
    }
    if (rc == SQLITE_OK) rc = sqlite3_exec(store->db, "COMMIT", NULL, NULL, NULL);
    if (rc != SQLITE_OK) {
        fyodor_store_result result = sql_result(rc);
        (void)sqlite3_exec(store->db, "ROLLBACK", NULL, NULL, NULL);
        fyodor_store_close(store);
        return result;
    }
    *out = store;
    return FYODOR_STORE_OK;
}

void fyodor_store_close(fyodor_store *store)
{
    if (store == NULL) return;
    (void)sqlite3_close(store->db);
    free(store);
}

static int bind_text(sqlite3_stmt *statement, int index, const char *value)
{
    return sqlite3_bind_text(statement, index, value, -1, SQLITE_TRANSIENT);
}

static int bind_key(sqlite3_stmt *statement, const char *name_space, const char *uri)
{
    int rc = bind_text(statement, 1, name_space);
    return rc == SQLITE_OK ? bind_text(statement, 2, uri) : rc;
}

static int json_object(sqlite3 *db, const char *text)
{
    sqlite3_stmt *statement = NULL;
    int rc = sqlite3_prepare_v2(db,
        "SELECT CASE WHEN json_valid(?1) THEN json_type(?1)='object' "
        "AND NOT EXISTS(SELECT 1 FROM json_tree(?1) WHERE typeof(key)='text' GROUP BY parent,key HAVING count(*)>1) "
        "AND NOT EXISTS(SELECT 1 FROM json_tree(?1) WHERE (type='text' AND instr(atom,char(0))>0) "
        "OR (typeof(key)='text' AND instr(key,char(0))>0)) ELSE 0 END", -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_text(statement, 1, text);
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(statement);
        if (rc == SQLITE_ROW) rc = sqlite3_column_int(statement, 0) ? SQLITE_OK : SQLITE_MISMATCH;
    }
    (void)sqlite3_finalize(statement);
    return rc;
}

static int parent_uri(const fyodor_resource_ref *ref, char *out)
{
    fyodor_resource_ref parent = {0};
    if (ref->kind == FYODOR_RESOURCE_WORLD_LORE || ref->kind == FYODOR_RESOURCE_WORLD_SAVE)
        parent.kind = FYODOR_RESOURCE_WORLD;
    else if (ref->kind == FYODOR_RESOURCE_AGENT_RUN) parent.kind = FYODOR_RESOURCE_AGENT;
    else { out[0] = '\0'; return 0; }
    parent.id = ref->parent_id;
    return fyodor_resource_format(&parent, out, FYODOR_RESOURCE_URI_CAPACITY) == FYODOR_RESOURCE_OK ? 0 : -1;
}

static int live_revision(sqlite3 *db, const char *name_space, const char *uri, sqlite3_int64 *revision)
{
    sqlite3_stmt *statement = NULL;
    int rc = sqlite3_prepare_v2(db,
        "SELECT revision FROM resources WHERE namespace=?1 AND uri=?2 AND deleted=0", -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_key(statement, name_space, uri);
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(statement);
        if (rc == SQLITE_ROW) { *revision = sqlite3_column_int64(statement, 0); rc = SQLITE_OK; }
        else if (rc == SQLITE_DONE) { *revision = 0; rc = SQLITE_OK; }
    }
    (void)sqlite3_finalize(statement);
    return rc;
}

static int append_history(sqlite3 *db, const char *name_space, const char *uri)
{
    sqlite3_stmt *statement = NULL;
    int rc = sqlite3_prepare_v2(db,
        "INSERT INTO revisions SELECT namespace,uri,revision,created_ms,modified_ms,"
        "deleted,title,content,metadata,provenance FROM resources WHERE namespace=?1 AND uri=?2",
        -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_key(statement, name_space, uri);
    if (rc == SQLITE_OK) rc = sqlite3_step(statement);
    int closed = sqlite3_finalize(statement);
    return rc == SQLITE_DONE ? closed : rc;
}

static fyodor_store_result finish(sqlite3 *db, int rc)
{
    if (rc == SQLITE_OK) rc = sqlite3_exec(db, "COMMIT", NULL, NULL, NULL);
    if (rc != SQLITE_OK) (void)sqlite3_exec(db, "ROLLBACK", NULL, NULL, NULL);
    return sql_result(rc);
}

/* Called only inside the resource writer transaction. Stable integer keys
 * survive VACUUM and do not depend on resources' implicit rowids. Tombstoned
 * resources retain their key but have no searchable index row. */
static int search_refresh(sqlite3 *db,const char *name_space,const char *uri,int deleted)
{
    const char *steps[]={
        "INSERT INTO resource_search_keys(namespace,uri) VALUES(?1,?2) ON CONFLICT(namespace,uri) DO NOTHING",
        deleted?
        "DELETE FROM resource_search WHERE rowid=(SELECT id FROM resource_search_keys WHERE namespace=?1 AND uri=?2)":
        "INSERT OR REPLACE INTO resource_search(rowid,title,content) SELECT k.id,lower(r.title),lower(r.content) "
        "FROM resources r JOIN resource_search_keys k USING(namespace,uri) WHERE r.namespace=?1 AND r.uri=?2 AND r.deleted=0"
    };
    int rc=SQLITE_OK;
    for(size_t i=0;i<2&&rc==SQLITE_OK;++i) {
        sqlite3_stmt *statement=NULL;
        rc=sqlite3_prepare_v2(db,steps[i],-1,&statement,NULL);
        if(rc==SQLITE_OK) rc=bind_key(statement,name_space,uri);
        if(rc==SQLITE_OK) { rc=sqlite3_step(statement); if(rc==SQLITE_DONE) rc=SQLITE_OK; }
        int closed=sqlite3_finalize(statement); if(rc==SQLITE_OK) rc=closed;
    }
    return rc;
}

static int writing_kind(fyodor_resource_kind kind)
{
    return kind==FYODOR_RESOURCE_DOCUMENT||kind==FYODOR_RESOURCE_NOTE||
        kind==FYODOR_RESOURCE_CHARACTER||kind==FYODOR_RESOURCE_PROJECT;
}

/* Shared parser for persisted folder links. Missing means unfiled; malformed
 * reserved metadata never becomes an arbitrary resource lookup. */
static int writing_parent_ref(sqlite3 *db,const char *metadata,fyodor_resource_ref *parent,int *found)
{
    sqlite3_stmt *s=NULL;*found=0;
    int rc=sqlite3_prepare_v2(db,"SELECT json_type(?1,'$.writing_parent'),json_extract(?1,'$.writing_parent')",-1,&s,NULL);
    if(rc==SQLITE_OK)rc=bind_text(s,1,metadata);
    if(rc==SQLITE_OK)rc=sqlite3_step(s);
    if(rc==SQLITE_ROW) {
        if(sqlite3_column_type(s,0)==SQLITE_NULL)rc=SQLITE_OK;
        else {
            const char *type=(const char *)sqlite3_column_text(s,0),*value=(const char *)sqlite3_column_text(s,1);
            if(!type||strcmp(type,"text")||!value||
                fyodor_resource_parse(value,strlen(value),parent)!=FYODOR_RESOURCE_OK||parent->kind!=FYODOR_RESOURCE_PROJECT)rc=SQLITE_CONSTRAINT;
            else {*found=1;rc=SQLITE_OK;}
        }
    }
    int closed=sqlite3_finalize(s);return rc==SQLITE_OK?closed:rc;
}

/* Lore links keep shared identities, never copied content or grants. */
static int writing_lore_refs(sqlite3 *db,const char *metadata,fyodor_resource_ref refs[32],size_t *count)
{
    *count=0;sqlite3_stmt *s=NULL;
    int rc=sqlite3_prepare_v2(db,"SELECT coalesce(json_type(?1,'$.writing_lore'),'array')='array'",-1,&s,NULL);
    if(rc==SQLITE_OK)rc=bind_text(s,1,metadata);
    if(rc==SQLITE_OK)rc=sqlite3_step(s);
    if(rc==SQLITE_ROW)rc=sqlite3_column_int(s,0)?SQLITE_OK:SQLITE_CONSTRAINT;
    int closed=sqlite3_finalize(s);if(rc==SQLITE_OK)rc=closed;
    if(rc!=SQLITE_OK)return rc;
    rc=sqlite3_prepare_v2(db,"SELECT type,value FROM json_each(?1,'$.writing_lore') ORDER BY key",-1,&s,NULL);
    if(rc==SQLITE_OK)rc=bind_text(s,1,metadata);
    while(rc==SQLITE_OK&&(rc=sqlite3_step(s))==SQLITE_ROW) {
        const char *type=(const char *)sqlite3_column_text(s,0),*uri=(const char *)sqlite3_column_text(s,1);
        if(*count==32||!type||strcmp(type,"text")||!uri||
            fyodor_resource_parse(uri,strlen(uri),&refs[*count])!=FYODOR_RESOURCE_OK||refs[*count].kind!=FYODOR_RESOURCE_WORLD_LORE) {rc=SQLITE_CONSTRAINT;break;}
        rc=SQLITE_OK;
        for(size_t i=0;i<*count;++i)if(fyodor_resource_equal(&refs[i],&refs[*count]))rc=SQLITE_CONSTRAINT;
        ++*count;
    }
    if(rc==SQLITE_DONE)rc=SQLITE_OK;
    closed=sqlite3_finalize(s);return rc==SQLITE_OK?closed:rc;
}

static int writing_lore_check(sqlite3 *db,const fyodor_resource_input *input)
{
    if(!writing_kind(input->ref.kind))return SQLITE_OK;
    fyodor_resource_ref refs[32];size_t count=0;
    int rc=writing_lore_refs(db,input->metadata,refs,&count);
    for(size_t i=0;rc==SQLITE_OK&&i<count;++i) {
        char uri[FYODOR_RESOURCE_URI_CAPACITY];sqlite3_int64 revision=0;
        (void)fyodor_resource_format(&refs[i],uri,sizeof(uri));
        rc=live_revision(db,input->name_space,uri,&revision);
        if(rc==SQLITE_OK&&!revision)rc=SQLITE_CONSTRAINT;
    }
    return rc;
}

/* Metadata travels with history/packages. Validate its reserved relationship
 * inside the writer transaction, including imported and scoped writes. */
static int writing_parent_check(sqlite3 *db,const fyodor_resource_input *input,const char *uri)
{
    if(!writing_kind(input->ref.kind))return SQLITE_OK;
    fyodor_resource_ref ref={0};int found=0;
    int rc=writing_parent_ref(db,input->metadata,&ref,&found);
    if(rc!=SQLITE_OK||!found)return rc;
    char parent[FYODOR_RESOURCE_URI_CAPACITY];
    (void)fyodor_resource_format(&ref,parent,sizeof(parent));
    sqlite3_stmt *s=NULL;
    sqlite3_int64 revision=0;
    rc=live_revision(db,input->name_space,parent,&revision);
    if(rc==SQLITE_OK&&!revision)rc=SQLITE_CONSTRAINT;
    if(rc!=SQLITE_OK)return rc;
    /* UNION (not UNION ALL) terminates even with externally corrupted cycles. */
    rc=sqlite3_prepare_v2(db,
        "WITH RECURSIVE ancestors(uri) AS (VALUES(?2) UNION "
        "SELECT json_extract(r.metadata,'$.writing_parent') FROM resources r JOIN ancestors a ON r.uri=a.uri "
        "WHERE r.namespace=?1 AND r.deleted=0 AND json_type(r.metadata,'$.writing_parent')='text') "
        "SELECT EXISTS(SELECT 1 FROM ancestors WHERE uri=?3)",-1,&s,NULL);
    if(rc==SQLITE_OK)rc=bind_key(s,input->name_space,parent);
    if(rc==SQLITE_OK)rc=bind_text(s,3,uri);
    if(rc==SQLITE_OK)rc=sqlite3_step(s);
    if(rc==SQLITE_ROW)rc=sqlite3_column_int(s,0)?SQLITE_CONSTRAINT:SQLITE_OK;
    int closed=sqlite3_finalize(s);return rc==SQLITE_OK?closed:rc;
}

static fyodor_store_result store_put(fyodor_store *store, const fyodor_resource_input *input,
                                    uint64_t expected_revision, uint64_t *revision,int transaction_owned)
{
    char uri[FYODOR_RESOURCE_URI_CAPACITY], parent[FYODOR_RESOURCE_URI_CAPACITY];
    if (store == NULL || input == NULL || revision == NULL || expected_revision >= INT64_MAX ||
        !namespace_valid(input->name_space) || !text_valid(input->title, 1024, 1) ||
        !text_valid(input->content, FYODOR_STORE_CONTENT_LIMIT, 0) ||
        !text_valid(input->metadata, FYODOR_STORE_METADATA_LIMIT, 0) ||
        !text_valid(input->provenance, FYODOR_STORE_METADATA_LIMIT, 0) ||
        fyodor_resource_format(&input->ref, uri, sizeof(uri)) != FYODOR_RESOURCE_OK ||
        parent_uri(&input->ref, parent) != 0) return FYODOR_STORE_INVALID;
    int rc = json_object(store->db, input->metadata);
    if (rc == SQLITE_OK) rc = json_object(store->db, input->provenance);
    if (rc != SQLITE_OK) return rc == SQLITE_MISMATCH ? FYODOR_STORE_INVALID : sql_result(rc);
    if(!transaction_owned) rc = sqlite3_exec(store->db, "BEGIN IMMEDIATE", NULL, NULL, NULL);
    if (rc != SQLITE_OK) return sql_result(rc);
    sqlite3_int64 current = 0;
    rc=writing_parent_check(store->db,input,uri);
    if(rc==SQLITE_OK)rc=writing_lore_check(store->db,input);
    if (rc==SQLITE_OK && parent[0] != '\0') {
        rc = live_revision(store->db, input->name_space, parent, &current);
        if (rc == SQLITE_OK && current == 0) rc = SQLITE_CONSTRAINT;
    }
    if (rc == SQLITE_OK && expected_revision != 0) {
        rc = live_revision(store->db, input->name_space, uri, &current);
        if (rc == SQLITE_OK && current != (sqlite3_int64)expected_revision) rc = SQLITE_CONSTRAINT;
    }
    sqlite3_stmt *statement = NULL;
    if (rc == SQLITE_OK) rc = sqlite3_prepare_v2(store->db, expected_revision == 0 ?
        "INSERT INTO resources(namespace,uri,parent_uri,revision,created_ms,modified_ms,deleted,title,content,metadata,provenance) "
        "VALUES(?1,?2,?7,1,CAST(unixepoch('subsec')*1000 AS INTEGER),CAST(unixepoch('subsec')*1000 AS INTEGER),0,?3,?4,?5,?6)" :
        "UPDATE resources SET revision=revision+1,modified_ms=max(modified_ms,CAST(unixepoch('subsec')*1000 AS INTEGER)),"
        "title=?3,content=?4,metadata=?5,provenance=?6 WHERE namespace=?1 AND uri=?2", -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_key(statement, input->name_space, uri);
    if (rc == SQLITE_OK) rc = bind_text(statement, 3, input->title);
    if (rc == SQLITE_OK) rc = bind_text(statement, 4, input->content);
    if (rc == SQLITE_OK) rc = bind_text(statement, 5, input->metadata);
    if (rc == SQLITE_OK) rc = bind_text(statement, 6, input->provenance);
    if (rc == SQLITE_OK && expected_revision == 0)
        rc = parent[0] ? bind_text(statement, 7, parent) : sqlite3_bind_null(statement, 7);
    if (rc == SQLITE_OK) { rc = sqlite3_step(statement); if (rc == SQLITE_DONE) rc = SQLITE_OK; }
    int closed = sqlite3_finalize(statement);
    if (rc == SQLITE_OK) rc = closed;
    if (rc == SQLITE_OK) rc = search_refresh(store->db,input->name_space,uri,0);
    if (rc == SQLITE_OK) rc = append_history(store->db, input->name_space, uri);
    fyodor_store_result result = transaction_owned ? sql_result(rc) : finish(store->db, rc);
    if (result == FYODOR_STORE_OK) *revision = expected_revision + 1;
    return result;
}

fyodor_store_result fyodor_store_put(fyodor_store *store,const fyodor_resource_input *input,
    uint64_t expected_revision,uint64_t *revision)
{
    return store_put(store,input,expected_revision,revision,0);
}

void fyodor_resource_record_free(fyodor_resource_record *record)
{
    if (record == NULL) return;
    free(record->name_space); free(record->title); free(record->content);
    free(record->metadata); free(record->provenance);
    memset(record, 0, sizeof(*record));
}

static int column_copy(sqlite3_stmt *statement, int index, size_t limit, char **out)
{
    if (sqlite3_column_type(statement, index) != SQLITE_TEXT) return SQLITE_CORRUPT;
    const unsigned char *text = sqlite3_column_text(statement, index);
    if (text == NULL) return SQLITE_NOMEM;
    int bytes = sqlite3_column_bytes(statement, index);
    if (bytes < 0 || (size_t)bytes > limit || memchr(text, 0, (size_t)bytes) != NULL)
        return SQLITE_CORRUPT;
    char *copy = malloc((size_t)bytes + 1);
    if (copy == NULL) return SQLITE_NOMEM;
    memcpy(copy, text, (size_t)bytes + 1);
    if (!text_valid(copy, limit, 0)) { free(copy); return SQLITE_CORRUPT; }
    *out = copy;
    return SQLITE_OK;
}

fyodor_store_result fyodor_store_get(fyodor_store *store, const char *name_space,
                                    const fyodor_resource_ref *ref, uint64_t revision,
                                    fyodor_resource_record *out)
{
    char uri[FYODOR_RESOURCE_URI_CAPACITY];
    if (store == NULL || out == NULL || !namespace_valid(name_space) || revision > INT64_MAX ||
        fyodor_resource_format(ref, uri, sizeof(uri)) != FYODOR_RESOURCE_OK) return FYODOR_STORE_INVALID;
    sqlite3_stmt *statement = NULL;
    int rc = sqlite3_prepare_v2(store->db, revision == 0 ?
        "SELECT namespace,title,content,metadata,provenance,revision,created_ms,modified_ms,deleted "
        "FROM resources WHERE namespace=?1 AND uri=?2 AND deleted=0" :
        "SELECT namespace,title,content,metadata,provenance,revision,created_ms,modified_ms,deleted "
        "FROM revisions WHERE namespace=?1 AND uri=?2 AND revision=?3", -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_key(statement, name_space, uri);
    if (rc == SQLITE_OK && revision != 0) rc = sqlite3_bind_int64(statement, 3, (sqlite3_int64)revision);
    if (rc == SQLITE_OK) rc = sqlite3_step(statement);
    if (rc == SQLITE_DONE) { (void)sqlite3_finalize(statement); return FYODOR_STORE_NOT_FOUND; }
    fyodor_resource_record record = {0};
    if (rc == SQLITE_ROW) {
        record.ref = *ref;
        rc = column_copy(statement, 0, 64, &record.name_space);
        if (rc == SQLITE_OK) rc = column_copy(statement, 1, 1024, &record.title);
        if (rc == SQLITE_OK) rc = column_copy(statement, 2, FYODOR_STORE_CONTENT_LIMIT, &record.content);
        if (rc == SQLITE_OK) rc = column_copy(statement, 3, FYODOR_STORE_METADATA_LIMIT, &record.metadata);
        if (rc == SQLITE_OK) rc = column_copy(statement, 4, FYODOR_STORE_METADATA_LIMIT, &record.provenance);
        sqlite3_int64 rev = sqlite3_column_int64(statement, 5);
        record.created_ms = sqlite3_column_int64(statement, 6);
        record.modified_ms = sqlite3_column_int64(statement, 7);
        record.deleted = sqlite3_column_int(statement, 8);
        if (rc == SQLITE_OK && (rev <= 0 || record.created_ms < 0 || record.modified_ms < record.created_ms ||
            (record.deleted != 0 && record.deleted != 1) || !text_valid(record.title, 1024, 1))) rc = SQLITE_CORRUPT;
        if (rc == SQLITE_OK) record.revision = (uint64_t)rev;
    }
    int closed = sqlite3_finalize(statement);
    if (rc == SQLITE_OK) rc = closed;
    if (rc == SQLITE_OK) rc = json_object(store->db, record.metadata);
    if (rc == SQLITE_OK) rc = json_object(store->db, record.provenance);
    if (rc != SQLITE_OK) {
        fyodor_resource_record_free(&record);
        return rc == SQLITE_MISMATCH ? FYODOR_STORE_SCHEMA_ERROR : sql_result(rc);
    }
    *out = record;
    return FYODOR_STORE_OK;
}

static fyodor_store_result store_delete(fyodor_store *store, const char *name_space,
                                       const fyodor_resource_ref *ref, uint64_t expected_revision,int transaction_owned)
{
    char uri[FYODOR_RESOURCE_URI_CAPACITY];
    if (store == NULL || !namespace_valid(name_space) || expected_revision == 0 || expected_revision >= INT64_MAX ||
        fyodor_resource_format(ref, uri, sizeof(uri)) != FYODOR_RESOURCE_OK) return FYODOR_STORE_INVALID;
    int rc = transaction_owned ? SQLITE_OK : sqlite3_exec(store->db, "BEGIN IMMEDIATE", NULL, NULL, NULL);
    if (rc != SQLITE_OK) return sql_result(rc);
    sqlite3_int64 current = 0;
    rc = live_revision(store->db, name_space, uri, &current);
    if (rc == SQLITE_OK && current != (sqlite3_int64)expected_revision) rc = SQLITE_CONSTRAINT;
    sqlite3_stmt *statement = NULL;
    if (rc == SQLITE_OK) rc = sqlite3_prepare_v2(store->db,
        "UPDATE resources SET deleted=1,revision=revision+1,"
        "modified_ms=max(modified_ms,CAST(unixepoch('subsec')*1000 AS INTEGER)) "
        "WHERE namespace=?1 AND uri=?2 AND NOT EXISTS("
        "SELECT 1 FROM resources WHERE namespace=?1 AND deleted=0 AND (parent_uri=?2 OR "
        "(uri GLOB 'fyodor://writing/*' AND (json_extract(metadata,'$.writing_parent')=?2 OR "
        "EXISTS(SELECT 1 FROM json_each(resources.metadata,'$.writing_lore') WHERE value=?2)))))", -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_key(statement, name_space, uri);
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(statement);
        if (rc == SQLITE_DONE) rc = sqlite3_changes(store->db) == 1 ? SQLITE_OK : SQLITE_CONSTRAINT;
    }
    int closed = sqlite3_finalize(statement);
    if (rc == SQLITE_OK) rc = closed;
    if (rc == SQLITE_OK) rc = search_refresh(store->db,name_space,uri,1);
    if (rc == SQLITE_OK) rc = append_history(store->db, name_space, uri);
    return transaction_owned ? sql_result(rc) : finish(store->db, rc);
}

fyodor_store_result fyodor_store_delete(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,uint64_t expected_revision)
{
    return store_delete(store,name_space,ref,expected_revision,0);
}

void fyodor_resource_summaries_free(fyodor_resource_summary *items, size_t count)
{
    if (items == NULL) return;
    for (size_t i = 0; i < count; ++i) free(items[i].title);
    free(items);
}

static fyodor_store_result store_list(fyodor_store *store, const char *name_space,
                                     const char *after_uri, size_t limit,
                                     fyodor_resource_summary **items, size_t *count,int writing,const char *folder)
{
    fyodor_resource_ref after;
    if (store == NULL || items == NULL || count == NULL || !namespace_valid(name_space) ||
        limit == 0 || limit > 100) return FYODOR_STORE_INVALID;
    if (after_uri != NULL && (!text_valid(after_uri, FYODOR_RESOURCE_URI_CAPACITY - 1, 1) ||
        fyodor_resource_parse(after_uri, strlen(after_uri), &after) != FYODOR_RESOURCE_OK))
        return FYODOR_STORE_INVALID;
    fyodor_resource_summary *page = calloc(limit, sizeof(*page));
    if (page == NULL) return FYODOR_STORE_NOMEM;
    sqlite3_stmt *statement = NULL;
    int rc = sqlite3_prepare_v2(store->db,
        "SELECT uri,title,revision,modified_ms FROM resources "
        "WHERE namespace=?1 AND deleted=0 AND uri>?2 AND (?4=0 OR (?4=1 AND uri GLOB 'fyodor://writing/*') OR (?4=2 AND uri GLOB 'fyodor://writing/projects/*') OR (?4=3 AND uri GLOB 'fyodor://explore/worlds/*/lore/*')) "
        "AND (?5 IS NULL OR coalesce(json_extract(metadata,'$.writing_parent'),'')=?5) ORDER BY uri LIMIT ?3", -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_key(statement, name_space, after_uri != NULL ? after_uri : "");
    if (rc == SQLITE_OK) rc = sqlite3_bind_int(statement, 3, (int)limit);
    if (rc == SQLITE_OK) rc = sqlite3_bind_int(statement,4,writing);
    if (rc == SQLITE_OK) rc = folder?bind_text(statement,5,folder):sqlite3_bind_null(statement,5);
    size_t used = 0;
    while (rc == SQLITE_OK) {
        rc = sqlite3_step(statement);
        if (rc == SQLITE_DONE) { rc = SQLITE_OK; break; }
        if (rc != SQLITE_ROW) break;
        if (used >= limit) { rc = SQLITE_CORRUPT; break; }
        char *uri = NULL;
        rc = column_copy(statement, 0, FYODOR_RESOURCE_URI_CAPACITY - 1, &uri);
        if (rc == SQLITE_OK && fyodor_resource_parse(uri, strlen(uri), &page[used].ref) != FYODOR_RESOURCE_OK)
            rc = SQLITE_CORRUPT;
        free(uri);
        if (rc == SQLITE_OK) rc = column_copy(statement, 1, 1024, &page[used].title);
        sqlite3_int64 rev = sqlite3_column_int64(statement, 2);
        page[used].modified_ms = sqlite3_column_int64(statement, 3);
        if (rc == SQLITE_OK && (rev <= 0 || page[used].modified_ms < 0 ||
            !text_valid(page[used].title, 1024, 1))) rc = SQLITE_CORRUPT;
        if (rc == SQLITE_OK) page[used].revision = (uint64_t)rev;
        ++used;
    }
    int closed = sqlite3_finalize(statement);
    if (rc == SQLITE_OK) rc = closed;
    if (rc != SQLITE_OK) { fyodor_resource_summaries_free(page, used); return sql_result(rc); }
    *items = page; *count = used;
    return FYODOR_STORE_OK;
}

fyodor_store_result fyodor_store_list(fyodor_store *store,const char *name_space,
    const char *after_uri,size_t limit,fyodor_resource_summary **items,size_t *count)
{ return store_list(store,name_space,after_uri,limit,items,count,0,NULL); }
fyodor_store_result fyodor_store_list_writing(fyodor_store *store,const char *name_space,
    const char *after_uri,size_t limit,fyodor_resource_summary **items,size_t *count)
{ return store_list(store,name_space,after_uri,limit,items,count,1,NULL); }

fyodor_store_result fyodor_store_list_projects(fyodor_store *store,const char *name_space,
    const char *after_uri,size_t limit,fyodor_resource_summary **items,size_t *count)
{ return store_list(store,name_space,after_uri,limit,items,count,2,NULL); }

fyodor_store_result fyodor_store_list_lore(fyodor_store *store,const char *name_space,
    const char *after_uri,size_t limit,fyodor_resource_summary **items,size_t *count)
{ return store_list(store,name_space,after_uri,limit,items,count,3,NULL); }

fyodor_store_result fyodor_store_list_folder(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *parent,const char *after_uri,size_t limit,fyodor_resource_summary **items,size_t *count)
{
    char uri[FYODOR_RESOURCE_URI_CAPACITY]={0};
    if(parent&&(parent->kind!=FYODOR_RESOURCE_PROJECT||fyodor_resource_format(parent,uri,sizeof(uri))!=FYODOR_RESOURCE_OK))return FYODOR_STORE_INVALID;
    return store_list(store,name_space,after_uri,limit,items,count,1,uri);
}

static fyodor_store_result writing_metadata_update(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,uint64_t expected_revision,const char *sql,const char *value,uint64_t *revision)
{
    char uri[FYODOR_RESOURCE_URI_CAPACITY];
    if(!store||!revision||!ref||!writing_kind(ref->kind)||!namespace_valid(name_space)||!expected_revision||expected_revision>=INT64_MAX||
        fyodor_resource_format(ref,uri,sizeof(uri))!=FYODOR_RESOURCE_OK)return FYODOR_STORE_INVALID;
    int rc=sqlite3_exec(store->db,"BEGIN IMMEDIATE",NULL,NULL,NULL);
    if(rc!=SQLITE_OK)return sql_result(rc);
    fyodor_resource_record record={0};char *metadata=NULL;uint64_t next=0;
    fyodor_store_result result=fyodor_store_get(store,name_space,ref,0,&record);
    if(result==FYODOR_STORE_OK&&record.revision!=expected_revision)result=FYODOR_STORE_CONFLICT;
    if(result==FYODOR_STORE_OK) {
        sqlite3_stmt *s=NULL;
        rc=sqlite3_prepare_v2(store->db,sql,-1,&s,NULL);
        if(rc==SQLITE_OK)rc=bind_text(s,1,record.metadata);
        if(rc==SQLITE_OK&&value)rc=bind_text(s,2,value);
        if(rc==SQLITE_OK)rc=sqlite3_step(s);
        if(rc==SQLITE_ROW)rc=sqlite3_column_bytes(s,0)>(int)FYODOR_STORE_METADATA_LIMIT?SQLITE_TOOBIG:column_copy(s,0,FYODOR_STORE_METADATA_LIMIT,&metadata);
        int closed=sqlite3_finalize(s);result=rc==SQLITE_TOOBIG?FYODOR_STORE_INVALID:sql_result(rc==SQLITE_OK?closed:rc);
    }
    if(result==FYODOR_STORE_OK) {
        fyodor_resource_input input={*ref,name_space,record.title,record.content,metadata,record.provenance};
        result=store_put(store,&input,expected_revision,&next,1);
    }
    if(result==FYODOR_STORE_OK)result=finish(store->db,SQLITE_OK);
    else (void)sqlite3_exec(store->db,"ROLLBACK",NULL,NULL,NULL);
    free(metadata);fyodor_resource_record_free(&record);
    if(result==FYODOR_STORE_OK)*revision=next;
    return result;
}

fyodor_store_result fyodor_store_move_writing(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,uint64_t expected_revision,const fyodor_resource_ref *parent,uint64_t *revision)
{
    char folder[FYODOR_RESOURCE_URI_CAPACITY];
    if(parent&&(parent->kind!=FYODOR_RESOURCE_PROJECT||fyodor_resource_format(parent,folder,sizeof(folder))!=FYODOR_RESOURCE_OK))return FYODOR_STORE_INVALID;
    return writing_metadata_update(store,name_space,ref,expected_revision,parent?
        "SELECT json_set(?1,'$.writing_parent',?2)":"SELECT json_remove(?1,'$.writing_parent')",parent?folder:NULL,revision);
}

fyodor_store_result fyodor_store_set_lore(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,uint64_t expected_revision,const fyodor_resource_ref *lore,size_t count,uint64_t *revision)
{
    if(count>32||(count&&!lore))return FYODOR_STORE_INVALID;
    char json[32*(FYODOR_RESOURCE_URI_CAPACITY+3)+3];size_t used=0;json[used++]='[';
    for(size_t i=0;i<count;++i) {
        char uri[FYODOR_RESOURCE_URI_CAPACITY];
        if(lore[i].kind!=FYODOR_RESOURCE_WORLD_LORE||fyodor_resource_format(&lore[i],uri,sizeof(uri))!=FYODOR_RESOURCE_OK)return FYODOR_STORE_INVALID;
        for(size_t j=0;j<i;++j)if(fyodor_resource_equal(&lore[i],&lore[j]))return FYODOR_STORE_INVALID;
        /* Canonical URI grammar contains no characters needing JSON escaping. */
        if(i)json[used++]=',';
        json[used++]='"';size_t length=strlen(uri);memcpy(json+used,uri,length);used+=length;json[used++]='"';
    }
    json[used++]=']';json[used]=0;
    return writing_metadata_update(store,name_space,ref,expected_revision,count?
        "SELECT json_set(?1,'$.writing_lore',json(?2))":"SELECT json_remove(?1,'$.writing_lore')",count?json:NULL,revision);
}

fyodor_store_result fyodor_store_history(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,uint64_t before,size_t limit,fyodor_revision_summary **items,size_t *count)
{
    char uri[FYODOR_RESOURCE_URI_CAPACITY];
    if(store==NULL||items==NULL||count==NULL||!namespace_valid(name_space)||before>INT64_MAX||limit==0||limit>100||
        fyodor_resource_format(ref,uri,sizeof(uri))!=FYODOR_RESOURCE_OK) return FYODOR_STORE_INVALID;
    fyodor_revision_summary *page=calloc(limit,sizeof(*page));
    if(page==NULL) return FYODOR_STORE_NOMEM;
    sqlite3_stmt *statement=NULL;
    int rc=sqlite3_prepare_v2(store->db,"SELECT revision,modified_ms,deleted FROM revisions "
        "WHERE namespace=?1 AND uri=?2 AND (?3=0 OR revision<?3) ORDER BY revision DESC LIMIT ?4",-1,&statement,NULL);
    if(rc==SQLITE_OK) rc=bind_key(statement,name_space,uri);
    if(rc==SQLITE_OK) rc=sqlite3_bind_int64(statement,3,(sqlite3_int64)before);
    if(rc==SQLITE_OK) rc=sqlite3_bind_int(statement,4,(int)limit);
    size_t used=0;
    while(rc==SQLITE_OK) {
        rc=sqlite3_step(statement);
        if(rc==SQLITE_DONE) {rc=SQLITE_OK;break;}
        if(rc!=SQLITE_ROW) break;
        sqlite3_int64 revision=sqlite3_column_int64(statement,0),modified=sqlite3_column_int64(statement,1);
        int deleted=sqlite3_column_int(statement,2);
        if(used>=limit||revision<=0||modified<0||(deleted!=0&&deleted!=1)) {rc=SQLITE_CORRUPT;break;}
        page[used++]=(fyodor_revision_summary){(uint64_t)revision,modified,deleted};rc=SQLITE_OK;
    }
    int closed=sqlite3_finalize(statement);if(rc==SQLITE_OK)rc=closed;
    if(rc!=SQLITE_OK){free(page);return sql_result(rc);}
    *items=page;*count=used;return FYODOR_STORE_OK;
}

void fyodor_resource_package_free(char *json) { free(json); }

void fyodor_receipt_free(fyodor_receipt *receipt)
{
    if(receipt==NULL) return;
    free(receipt->prompt); free(receipt->output); free(receipt->metadata); free(receipt->sources);
    memset(receipt,0,sizeof(*receipt));
}

fyodor_store_result fyodor_receipt_save(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_receipt_input *input,fyodor_uuid *id)
{
    char who[FYODOR_UUID_TEXT_CAPACITY],identifier[FYODOR_UUID_TEXT_CAPACITY];
    if(store==NULL||input==NULL||id==NULL||input->context==NULL||!namespace_valid(name_space)||
        fyodor_uuid_format(principal,who,sizeof(who))!=FYODOR_RESOURCE_OK||
        !text_valid(input->prompt,2u*FYODOR_STORE_CONTENT_LIMIT+2,0)||
        !text_valid(input->output,FYODOR_STORE_CONTENT_LIMIT,0)||!text_valid(input->model_path,1024,0)||
        !text_valid(input->stop_reason,32,1)||!isfinite(input->temperature)||!isfinite(input->top_p)||
        input->temperature<0||input->top_p<=0||input->top_p>1||
        input->context->source_count>64||(input->context->source_count&&input->context->sources==NULL)||
        input->context->length>FYODOR_STORE_CONTENT_LIMIT||
        !text_valid(input->context->text,FYODOR_STORE_CONTENT_LIMIT,0)||
        strlen(input->context->text)!=input->context->length||strlen(input->prompt)<input->context->length||
        memcmp(input->prompt,input->context->text,input->context->length)!=0||
        input->prompt_tokens>INT64_MAX||input->generated_tokens>INT64_MAX||input->max_tokens>INT64_MAX||
        input->top_k>INT64_MAX||input->model_context_length>INT64_MAX) return FYODOR_STORE_INVALID;
    if(input->writing_mode&&strcmp(input->writing_mode,"generate")&&strcmp(input->writing_mode,"rewrite")&&strcmp(input->writing_mode,"continue"))return FYODOR_STORE_INVALID;
    fyodor_uuid generated;
    if(fyodor_uuid_generate(&generated)!=FYODOR_RESOURCE_OK) return FYODOR_STORE_IO;
    (void)fyodor_uuid_format(&generated,identifier,sizeof(identifier));
    char *sources=malloc(3); if(sources==NULL) return FYODOR_STORE_NOMEM;
    memcpy(sources,"[]",3);
    int rc=SQLITE_OK;
    for(size_t i=0;i<input->context->source_count&&rc==SQLITE_OK;++i) {
        const fyodor_context_source *source=&input->context->sources[i];
        char uri[FYODOR_RESOURCE_URI_CAPACITY];
        if(fyodor_context_layer_name(source->layer)==NULL||source->revision==0||source->revision>INT64_MAX||source->offset>input->context->length||
            source->length>input->context->length-source->offset||source->original_length>FYODOR_STORE_CONTENT_LIMIT||
            source->length>source->original_length||fyodor_resource_format(&source->ref,uri,sizeof(uri))!=FYODOR_RESOURCE_OK) {
            free(sources); return FYODOR_STORE_INVALID;
        }
        sqlite3_stmt *s=NULL;
        rc=sqlite3_prepare_v2(store->db,"SELECT json_insert(json(?1),'$[#]',json_object("
            "'uri',?2,'revision',?3,'offset',?4,'length',?5,'original_length',?6,'layer',?7))",-1,&s,NULL);
        if(rc==SQLITE_OK) rc=bind_text(s,1,sources);
        if(rc==SQLITE_OK) rc=bind_text(s,2,uri);
        if(rc==SQLITE_OK) rc=sqlite3_bind_int64(s,3,(sqlite3_int64)source->revision);
        if(rc==SQLITE_OK) rc=sqlite3_bind_int64(s,4,(sqlite3_int64)source->offset);
        if(rc==SQLITE_OK) rc=sqlite3_bind_int64(s,5,(sqlite3_int64)source->length);
        if(rc==SQLITE_OK) rc=sqlite3_bind_int64(s,6,(sqlite3_int64)source->original_length);
        if(rc==SQLITE_OK) rc=bind_text(s,7,fyodor_context_layer_name(source->layer));
        char *next=NULL;
        if(rc==SQLITE_OK) { rc=sqlite3_step(s); if(rc==SQLITE_ROW) rc=column_copy(s,0,65536,&next); }
        int closed=sqlite3_finalize(s); if(rc==SQLITE_OK) rc=closed;
        free(sources); sources=next;
    }
    char seed[32],file_size[32];
    snprintf(seed,sizeof(seed),"%llu",(unsigned long long)input->seed);
    snprintf(file_size,sizeof(file_size),"%llu",(unsigned long long)input->model_file_size);
    sqlite3_stmt *s=NULL;
    if(rc==SQLITE_OK) rc=sqlite3_prepare_v2(store->db,
        "INSERT INTO context_receipts(id,principal,namespace,created_ms,prompt,output,sources,metadata) "
        "VALUES(?1,?2,?3,CAST(unixepoch('subsec')*1000 AS INTEGER),?4,?5,?6,"
        "json_object('schema',1,'model_path',?7,'model_file_size',?8,'seed',?9,'temperature',?10,"
        "'top_p',?11,'top_k',?12,'max_tokens',?13,'prompt_tokens',?14,'generated_tokens',?15,"
        "'model_context_length',?16,'stop_reason',?17,'context_bytes',?18,'writing_mode',?19))",-1,&s,NULL);
    const char *values[]={identifier,who,name_space,input->prompt,input->output,sources,input->model_path,file_size,seed};
    for(int i=0;i<9&&rc==SQLITE_OK;++i) rc=bind_text(s,i+1,values[i]);
    if(rc==SQLITE_OK) rc=sqlite3_bind_double(s,10,input->temperature);
    if(rc==SQLITE_OK) rc=sqlite3_bind_double(s,11,input->top_p);
    const size_t numbers[]={input->top_k,input->max_tokens,input->prompt_tokens,input->generated_tokens,input->model_context_length};
    for(int i=0;i<5&&rc==SQLITE_OK;++i) rc=sqlite3_bind_int64(s,i+12,(sqlite3_int64)numbers[i]);
    if(rc==SQLITE_OK) rc=bind_text(s,17,input->stop_reason);
    if(rc==SQLITE_OK) rc=sqlite3_bind_int64(s,18,(sqlite3_int64)input->context->length);
    if(rc==SQLITE_OK) rc=input->writing_mode?bind_text(s,19,input->writing_mode):sqlite3_bind_null(s,19);
    if(rc==SQLITE_OK) { rc=sqlite3_step(s); if(rc==SQLITE_DONE) rc=SQLITE_OK; }
    int closed=sqlite3_finalize(s); if(rc==SQLITE_OK) rc=closed;
    free(sources);
    if(rc==SQLITE_OK) *id=generated;
    return sql_result(rc);
}

#define RECEIPT_VISIBLE "NOT EXISTS(" \
    "SELECT 1 FROM json_each(c.sources) source WHERE NOT EXISTS(" \
    "SELECT 1 FROM context_permissions p JOIN resources r ON r.namespace=p.namespace AND r.uri=p.uri " \
    "WHERE p.principal=?2 AND p.namespace=?3 AND p.uri=json_extract(source.value,'$.uri') " \
    "AND (p.permissions & 1)=1 AND r.deleted=0))"

fyodor_store_result fyodor_receipt_read(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_uuid *id,fyodor_receipt *out)
{
    char who[FYODOR_UUID_TEXT_CAPACITY],identifier[FYODOR_UUID_TEXT_CAPACITY];
    if(store==NULL||out==NULL||!namespace_valid(name_space)||
        fyodor_uuid_format(principal,who,sizeof(who))!=FYODOR_RESOURCE_OK||
        fyodor_uuid_format(id,identifier,sizeof(identifier))!=FYODOR_RESOURCE_OK) return FYODOR_STORE_INVALID;
    sqlite3_stmt *s=NULL;
    int rc=sqlite3_prepare_v2(store->db,"SELECT c.prompt,c.output,c.metadata,c.sources,c.created_ms FROM context_receipts c "
        "WHERE c.id=?1 AND c.principal=?2 AND c.namespace=?3 AND " RECEIPT_VISIBLE,-1,&s,NULL);
    if(rc==SQLITE_OK) rc=bind_text(s,1,identifier);
    if(rc==SQLITE_OK) rc=bind_text(s,2,who);
    if(rc==SQLITE_OK) rc=bind_text(s,3,name_space);
    if(rc==SQLITE_OK) rc=sqlite3_step(s);
    if(rc==SQLITE_DONE) { (void)sqlite3_finalize(s); return FYODOR_STORE_DENIED; }
    fyodor_receipt receipt={0}; receipt.id=*id;
    if(rc==SQLITE_ROW) {
        rc=column_copy(s,0,2u*FYODOR_STORE_CONTENT_LIMIT+2,&receipt.prompt);
        if(rc==SQLITE_OK) rc=column_copy(s,1,FYODOR_STORE_CONTENT_LIMIT,&receipt.output);
        if(rc==SQLITE_OK) rc=column_copy(s,2,65536,&receipt.metadata);
        if(rc==SQLITE_OK) rc=column_copy(s,3,65536,&receipt.sources);
        receipt.created_ms=sqlite3_column_int64(s,4);
        if(rc==SQLITE_OK&&receipt.created_ms<0) rc=SQLITE_CORRUPT;
    }
    int closed=sqlite3_finalize(s); if(rc==SQLITE_OK) rc=closed;
    if(rc==SQLITE_OK) *out=receipt; else fyodor_receipt_free(&receipt);
    return sql_result(rc);
}

fyodor_store_result fyodor_writing_save(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *target,uint64_t expected_revision,
    const fyodor_uuid *receipt_id,const char *title,const char *content,uint64_t *revision)
{
    char uri[FYODOR_RESOURCE_URI_CAPACITY],id[FYODOR_UUID_TEXT_CAPACITY],who[FYODOR_UUID_TEXT_CAPACITY];
    if(!store||!revision||expected_revision==0||expected_revision>=INT64_MAX||!namespace_valid(name_space)||
        !text_valid(title,1024,1)||!text_valid(content,FYODOR_STORE_CONTENT_LIMIT,0)||
        fyodor_resource_format(target,uri,sizeof(uri))!=FYODOR_RESOURCE_OK||
        fyodor_uuid_format(receipt_id,id,sizeof(id))!=FYODOR_RESOURCE_OK||fyodor_uuid_format(principal,who,sizeof(who))!=FYODOR_RESOURCE_OK)
        return FYODOR_STORE_INVALID;
    int rc=sqlite3_exec(store->db,"BEGIN IMMEDIATE",NULL,NULL,NULL);
    if(rc!=SQLITE_OK)return sql_result(rc);
    fyodor_receipt receipt={0};fyodor_resource_record record={0};char *provenance=NULL;char mode[16]={0};
    fyodor_store_result result=fyodor_receipt_read(store,principal,name_space,receipt_id,&receipt);
    sqlite3_stmt *s=NULL;
    if(result==FYODOR_STORE_OK) {
        rc=sqlite3_prepare_v2(store->db,"SELECT json_extract(?1,'$.writing_mode'),json_extract(?2,'$[0].uri'),json_extract(?2,'$[0].revision')",-1,&s,NULL);
        if(rc==SQLITE_OK)rc=bind_text(s,1,receipt.metadata);
        if(rc==SQLITE_OK)rc=bind_text(s,2,receipt.sources);
        if(rc==SQLITE_OK)rc=sqlite3_step(s);
        if(rc==SQLITE_ROW) {
            const char *action=(const char *)sqlite3_column_text(s,0),*source=(const char *)sqlite3_column_text(s,1);
            if(!action||!source||strcmp(source,uri)||
                (strcmp(action,"generate")&&strcmp(action,"rewrite")&&strcmp(action,"continue")))result=FYODOR_STORE_INVALID;
            else if(sqlite3_column_type(s,2)!=SQLITE_INTEGER||sqlite3_column_int64(s,2)!=(sqlite3_int64)expected_revision)result=FYODOR_STORE_CONFLICT;
            else {memcpy(mode,action,strlen(action)+1);rc=SQLITE_OK;}
        }
        int closed=sqlite3_finalize(s);s=NULL;
        if(result==FYODOR_STORE_OK)result=sql_result(rc==SQLITE_OK?closed:rc);
    }
    if(result==FYODOR_STORE_OK)result=fyodor_store_get(store,name_space,target,0,&record);
    if(result==FYODOR_STORE_OK&&record.revision!=expected_revision)result=FYODOR_STORE_CONFLICT;
    if(result==FYODOR_STORE_OK) {
        int edited=1;size_t old=strlen(record.content),generated=strlen(receipt.output),actual=strlen(content);
        if(strcmp(mode,"continue")==0) edited=actual!=old+generated||memcmp(content,record.content,old)!=0||memcmp(content+old,receipt.output,generated)!=0;
        else edited=strcmp(content,receipt.output)!=0;
        rc=sqlite3_prepare_v2(store->db,
            "SELECT json_insert(CASE WHEN json_type(?1,'$.writing_generations') IS NULL THEN json_set(?1,'$.writing_generations',json('[]')) ELSE ?1 END,"
            "'$.writing_generations[#]',json_object('receipt_id',?2,'source_revision',?3,'saved_revision',?4,'mode',?5,'text_edited',json(?6))) "
            "WHERE json_type(?1,'$.writing_generations') IS NULL OR json_type(?1,'$.writing_generations')='array'",-1,&s,NULL);
        char source_revision[32],saved_revision[32];
        snprintf(source_revision,sizeof(source_revision),"%llu",(unsigned long long)expected_revision);
        snprintf(saved_revision,sizeof(saved_revision),"%llu",(unsigned long long)(expected_revision+1));
        const char *values[]={record.provenance,id,source_revision,saved_revision,mode,edited?"true":"false"};
        for(int i=0;i<6&&rc==SQLITE_OK;++i)rc=bind_text(s,i+1,values[i]);
        if(rc==SQLITE_OK)rc=sqlite3_step(s);
        if(rc==SQLITE_DONE)result=FYODOR_STORE_CONFLICT;
        else if(rc==SQLITE_ROW) {
            if(sqlite3_column_bytes(s,0)>(int)FYODOR_STORE_METADATA_LIMIT)result=FYODOR_STORE_INVALID;
            else rc=column_copy(s,0,FYODOR_STORE_METADATA_LIMIT,&provenance);
        }
        int closed=sqlite3_finalize(s);
        if(result==FYODOR_STORE_OK)result=sql_result(rc==SQLITE_OK?closed:rc);
    }
    uint64_t next=0;
    if(result==FYODOR_STORE_OK) {
        fyodor_resource_input input={*target,name_space,title,content,record.metadata,provenance};
        result=store_put(store,&input,expected_revision,&next,1);
    }
    if(result==FYODOR_STORE_OK)result=finish(store->db,SQLITE_OK);
    else (void)sqlite3_exec(store->db,"ROLLBACK",NULL,NULL,NULL);
    free(provenance);fyodor_resource_record_free(&record);fyodor_receipt_free(&receipt);
    if(result==FYODOR_STORE_OK)*revision=next;
    return result;
}

void fyodor_receipt_summaries_free(fyodor_receipt_summary *items) { free(items); }
fyodor_store_result fyodor_receipt_list(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_uuid *after_id,size_t limit,fyodor_receipt_summary **items,size_t *count)
{
    char who[FYODOR_UUID_TEXT_CAPACITY],after[FYODOR_UUID_TEXT_CAPACITY]={0};
    if(!store||!items||!count||limit<1||limit>100||!namespace_valid(name_space)||
       fyodor_uuid_format(principal,who,sizeof(who))!=FYODOR_RESOURCE_OK||
       (after_id&&fyodor_uuid_format(after_id,after,sizeof(after))!=FYODOR_RESOURCE_OK)) return FYODOR_STORE_INVALID;
    fyodor_receipt_summary *page=calloc(limit,sizeof(*page));if(!page) return FYODOR_STORE_NOMEM;
    sqlite3_stmt *s=NULL;size_t used=0;
    int rc=sqlite3_prepare_v2(store->db,"SELECT c.id,c.created_ms FROM context_receipts c "
        "WHERE c.id>?1 AND c.principal=?2 AND c.namespace=?3 AND " RECEIPT_VISIBLE " ORDER BY c.id LIMIT ?4",-1,&s,NULL);
    if(rc==SQLITE_OK) rc=bind_text(s,1,after);
    if(rc==SQLITE_OK) rc=bind_text(s,2,who);
    if(rc==SQLITE_OK) rc=bind_text(s,3,name_space);
    if(rc==SQLITE_OK) rc=sqlite3_bind_int(s,4,(int)limit);
    while(rc==SQLITE_OK && (rc=sqlite3_step(s))==SQLITE_ROW) {
        const char *id=(const char *)sqlite3_column_text(s,0);int bytes=sqlite3_column_bytes(s,0);
        if(!id||bytes!=36||used>=limit||fyodor_uuid_parse(id,(size_t)bytes,&page[used].id)!=FYODOR_RESOURCE_OK) {rc=SQLITE_CORRUPT;break;}
        page[used].created_ms=sqlite3_column_int64(s,1);
        if(page[used].created_ms<0) {rc=SQLITE_CORRUPT;break;}
        ++used;rc=SQLITE_OK;
    }
    if(rc==SQLITE_DONE) rc=SQLITE_OK;
    int closed=sqlite3_finalize(s);if(rc==SQLITE_OK) rc=closed;
    if(rc==SQLITE_OK) {*items=page;*count=used;}else free(page);
    return sql_result(rc);
}
#undef RECEIPT_VISIBLE

static int context_key(const fyodor_uuid *principal,const char *name_space,
                       const fyodor_resource_ref *ref,char *who,char *uri)
{
    return namespace_valid(name_space) &&
        fyodor_uuid_format(principal,who,FYODOR_UUID_TEXT_CAPACITY)==FYODOR_RESOURCE_OK &&
        fyodor_resource_format(ref,uri,FYODOR_RESOURCE_URI_CAPACITY)==FYODOR_RESOURCE_OK;
}

static int permission_read(sqlite3 *db,const char *who,const char *name_space,const char *uri,unsigned *mask)
{
    sqlite3_stmt *s=NULL;
    int rc=sqlite3_prepare_v2(db,"SELECT p.permissions FROM context_permissions p JOIN resources r "
        "ON r.namespace=p.namespace AND r.uri=p.uri WHERE p.namespace=?1 AND p.uri=?2 AND p.principal=?3 AND r.deleted=0",-1,&s,NULL);
    if(rc==SQLITE_OK) rc=bind_key(s,name_space,uri);
    if(rc==SQLITE_OK) rc=bind_text(s,3,who);
    if(rc==SQLITE_OK) {
        rc=sqlite3_step(s);
        if(rc==SQLITE_ROW) {
            int value=sqlite3_column_int(s,0);
            if(value<1||value>FYODOR_PERMISSION_ALL) rc=SQLITE_CORRUPT;
            else { *mask=(unsigned)value; rc=SQLITE_OK; }
        } else if(rc==SQLITE_DONE) { *mask=0; rc=SQLITE_OK; }
    }
    int closed=sqlite3_finalize(s); return rc==SQLITE_OK?closed:rc;
}

fyodor_store_result fyodor_context_permissions_set(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *ref,unsigned permissions)
{
    char who[FYODOR_UUID_TEXT_CAPACITY],uri[FYODOR_RESOURCE_URI_CAPACITY];
    if(store==NULL||permissions>FYODOR_PERMISSION_ALL||!context_key(principal,name_space,ref,who,uri)) return FYODOR_STORE_INVALID;
    int rc=sqlite3_exec(store->db,"BEGIN IMMEDIATE",NULL,NULL,NULL);
    if(rc!=SQLITE_OK) return sql_result(rc);
    sqlite3_int64 revision=0;
    if(permissions!=0) {
        rc=live_revision(store->db,name_space,uri,&revision);
        if(rc==SQLITE_OK&&revision==0) rc=SQLITE_CONSTRAINT;
    }
    sqlite3_stmt *s=NULL;
    if(rc==SQLITE_OK) rc=sqlite3_prepare_v2(store->db,permissions==0?
        "DELETE FROM context_permissions WHERE namespace=?1 AND uri=?2 AND principal=?3":
        "INSERT INTO context_permissions(namespace,uri,principal,permissions) VALUES(?1,?2,?3,?4) "
        "ON CONFLICT(principal,namespace,uri) DO UPDATE SET permissions=excluded.permissions",-1,&s,NULL);
    if(rc==SQLITE_OK) rc=bind_key(s,name_space,uri);
    if(rc==SQLITE_OK) rc=bind_text(s,3,who);
    if(rc==SQLITE_OK&&permissions!=0) rc=sqlite3_bind_int(s,4,(int)permissions);
    if(rc==SQLITE_OK) { rc=sqlite3_step(s); if(rc==SQLITE_DONE) rc=SQLITE_OK; }
    int closed=sqlite3_finalize(s); if(rc==SQLITE_OK) rc=closed;
    return finish(store->db,rc);
}

fyodor_store_result fyodor_context_permissions_get(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *ref,unsigned *permissions)
{
    char who[FYODOR_UUID_TEXT_CAPACITY],uri[FYODOR_RESOURCE_URI_CAPACITY]; unsigned mask=0;
    if(store==NULL||permissions==NULL||!context_key(principal,name_space,ref,who,uri)) return FYODOR_STORE_INVALID;
    int rc=permission_read(store->db,who,name_space,uri,&mask);
    if(rc==SQLITE_OK) *permissions=mask;
    return sql_result(rc);
}

static fyodor_store_result context_read_snapshot(fyodor_store *store,const char *who,
    const char *name_space,const fyodor_resource_ref *ref,fyodor_resource_record *out)
{
    char uri[FYODOR_RESOURCE_URI_CAPACITY]; unsigned mask=0;
    if(fyodor_resource_format(ref,uri,sizeof(uri))!=FYODOR_RESOURCE_OK) return FYODOR_STORE_INVALID;
    int rc=permission_read(store->db,who,name_space,uri,&mask);
    if(rc!=SQLITE_OK) return sql_result(rc);
    if(!(mask&FYODOR_PERMISSION_READ)) return FYODOR_STORE_DENIED;
    fyodor_store_result result=fyodor_store_get(store,name_space,ref,0,out);
    return result==FYODOR_STORE_NOT_FOUND?FYODOR_STORE_DENIED:result;
}

static fyodor_store_result mutation_begin(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *ref,unsigned required)
{
    char who[FYODOR_UUID_TEXT_CAPACITY],uri[FYODOR_RESOURCE_URI_CAPACITY]; unsigned mask=0;
    if(store==NULL||!context_key(principal,name_space,ref,who,uri)) return FYODOR_STORE_INVALID;
    int rc=sqlite3_exec(store->db,"BEGIN IMMEDIATE",NULL,NULL,NULL);
    if(rc!=SQLITE_OK) return sql_result(rc);
    rc=permission_read(store->db,who,name_space,uri,&mask);
    if(rc==SQLITE_OK&&(mask&required)==required) return FYODOR_STORE_OK;
    (void)sqlite3_exec(store->db,"ROLLBACK",NULL,NULL,NULL);
    return rc==SQLITE_OK?FYODOR_STORE_DENIED:sql_result(rc);
}

static fyodor_store_result mutation_finish(fyodor_store *store,fyodor_store_result result)
{
    if(result==FYODOR_STORE_OK) return finish(store->db,SQLITE_OK);
    (void)sqlite3_exec(store->db,"ROLLBACK",NULL,NULL,NULL);
    return result;
}

fyodor_store_result fyodor_context_write(fyodor_store *store,const fyodor_uuid *principal,
    const fyodor_resource_input *input,uint64_t expected_revision,uint64_t *revision)
{
    if(input==NULL||revision==NULL||expected_revision==0||expected_revision>=INT64_MAX) return FYODOR_STORE_INVALID;
    fyodor_store_result result=mutation_begin(store,principal,input->name_space,&input->ref,FYODOR_PERMISSION_WRITE);
    if(result!=FYODOR_STORE_OK) return result;
    uint64_t next=0;
    result=store_put(store,input,expected_revision,&next,1);
    result=mutation_finish(store,result);
    if(result==FYODOR_STORE_OK) *revision=next;
    return result;
}

fyodor_store_result fyodor_context_append(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *ref,const char *suffix,
    uint64_t expected_revision,uint64_t *revision)
{
    if(revision==NULL||expected_revision==0||expected_revision>=INT64_MAX||
        !text_valid(suffix,FYODOR_STORE_CONTENT_LIMIT,0)) return FYODOR_STORE_INVALID;
    fyodor_store_result result=mutation_begin(store,principal,name_space,ref,FYODOR_PERMISSION_APPEND);
    if(result!=FYODOR_STORE_OK) return result;
    fyodor_resource_record record={0}; uint64_t next=0; char *combined=NULL;
    result=fyodor_store_get(store,name_space,ref,0,&record);
    if(result==FYODOR_STORE_OK&&record.revision!=expected_revision) result=FYODOR_STORE_CONFLICT;
    if(result==FYODOR_STORE_OK) {
        size_t old_length=strlen(record.content),extra=strlen(suffix);
        if(extra>FYODOR_STORE_CONTENT_LIMIT-old_length) result=FYODOR_STORE_INVALID;
        else {
            combined=malloc(old_length+extra+1);
            if(combined==NULL) result=FYODOR_STORE_NOMEM;
            else {
                memcpy(combined,record.content,old_length); memcpy(combined+old_length,suffix,extra+1);
                fyodor_resource_input input={record.ref,name_space,record.title,combined,record.metadata,record.provenance};
                result=store_put(store,&input,expected_revision,&next,1);
            }
        }
    }
    free(combined); fyodor_resource_record_free(&record);
    result=mutation_finish(store,result);
    if(result==FYODOR_STORE_OK) *revision=next;
    return result;
}

fyodor_store_result fyodor_context_delete(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *ref,uint64_t expected_revision)
{
    if(expected_revision==0||expected_revision>=INT64_MAX) return FYODOR_STORE_INVALID;
    fyodor_store_result result=mutation_begin(store,principal,name_space,ref,FYODOR_PERMISSION_DELETE);
    if(result!=FYODOR_STORE_OK) return result;
    return mutation_finish(store,store_delete(store,name_space,ref,expected_revision,1));
}

fyodor_store_result fyodor_context_read(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *ref,fyodor_resource_record *out)
{
    char who[FYODOR_UUID_TEXT_CAPACITY],uri[FYODOR_RESOURCE_URI_CAPACITY];
    if(store==NULL||out==NULL||!context_key(principal,name_space,ref,who,uri)) return FYODOR_STORE_INVALID;
    int rc=sqlite3_exec(store->db,"BEGIN",NULL,NULL,NULL);
    if(rc!=SQLITE_OK) return sql_result(rc);
    fyodor_resource_record record={0};
    fyodor_store_result result=context_read_snapshot(store,who,name_space,ref,&record);
    if(result==FYODOR_STORE_OK) result=finish(store->db,SQLITE_OK);
    else (void)sqlite3_exec(store->db,"ROLLBACK",NULL,NULL,NULL);
    if(result==FYODOR_STORE_OK) *out=record; else fyodor_resource_record_free(&record);
    return result;
}

fyodor_store_result fyodor_context_search(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const char *query,size_t limit,fyodor_resource_summary **items,size_t *count)
{
    char who[FYODOR_UUID_TEXT_CAPACITY];
    if(store==NULL||items==NULL||count==NULL||!namespace_valid(name_space)||limit<1||limit>100||
        !text_valid(query,128,1)||query[0]==0||fyodor_uuid_format(principal,who,sizeof(who))!=FYODOR_RESOURCE_OK) return FYODOR_STORE_INVALID;
    fyodor_resource_summary *page=calloc(limit,sizeof(*page));
    if(page==NULL) return FYODOR_STORE_NOMEM;
    sqlite3_stmt *s=NULL;
    /* MATCH is an escaped literal phrase over ASCII-folded text. Queries
     * shorter than three Unicode scalars retain the permitted-row scan. The
     * final literal predicate and grants are checked in the same statement. */
    char phrase[259]; size_t bytes=0,scalars=0; phrase[bytes++]='"';
    for(size_t i=0;query[i];++i) {
        unsigned char c=(unsigned char)query[i];
        if((c&0xc0u)!=0x80u) ++scalars;
        if(c>='A'&&c<='Z') c=(unsigned char)(c+('a'-'A'));
        if(c=='"') phrase[bytes++]='"';
        phrase[bytes++]=(char)c;
    }
    phrase[bytes++]='"';phrase[bytes]=0;
    const char *scan="SELECT r.uri,r.title,r.revision,r.modified_ms FROM resources r "
        "JOIN context_permissions p ON r.namespace=p.namespace AND r.uri=p.uri "
        "WHERE r.namespace=?1 AND p.principal=?2 AND (p.permissions & 5)=5 AND r.deleted=0 "
        "AND (instr(lower(r.title),lower(?3))>0 OR instr(lower(r.content),lower(?3))>0) "
        "ORDER BY (instr(lower(r.title),lower(?3))>0) DESC,r.uri LIMIT ?4";
    const char *indexed="SELECT r.uri,r.title,r.revision,r.modified_ms FROM resource_search "
        "JOIN resource_search_keys k ON k.id=resource_search.rowid "
        "JOIN resources r ON r.namespace=k.namespace AND r.uri=k.uri "
        "JOIN context_permissions p ON r.namespace=p.namespace AND r.uri=p.uri "
        "WHERE resource_search MATCH ?5 AND r.namespace=?1 AND p.principal=?2 AND (p.permissions & 5)=5 AND r.deleted=0 "
        "AND (instr(lower(r.title),lower(?3))>0 OR instr(lower(r.content),lower(?3))>0) "
        "ORDER BY (instr(lower(r.title),lower(?3))>0) DESC,r.uri LIMIT ?4";
    int rc=sqlite3_prepare_v2(store->db,scalars>=3?indexed:scan,-1,&s,NULL);
    if(rc==SQLITE_OK&&scalars>=3) rc=bind_text(s,5,phrase);
    if(rc==SQLITE_OK) rc=bind_key(s,name_space,who);
    if(rc==SQLITE_OK) rc=bind_text(s,3,query);
    if(rc==SQLITE_OK) rc=sqlite3_bind_int(s,4,(int)limit);
    size_t used=0;
    while(rc==SQLITE_OK) {
        rc=sqlite3_step(s);
        if(rc==SQLITE_DONE) { rc=SQLITE_OK; break; }
        if(rc!=SQLITE_ROW) break;
        if(used>=limit) { rc=SQLITE_CORRUPT; break; }
        char *uri=NULL;
        rc=column_copy(s,0,FYODOR_RESOURCE_URI_CAPACITY-1,&uri);
        if(rc==SQLITE_OK&&fyodor_resource_parse(uri,strlen(uri),&page[used].ref)!=FYODOR_RESOURCE_OK) rc=SQLITE_CORRUPT;
        free(uri);
        if(rc==SQLITE_OK) rc=column_copy(s,1,1024,&page[used].title);
        sqlite3_int64 revision=sqlite3_column_int64(s,2);
        page[used].modified_ms=sqlite3_column_int64(s,3);
        if(rc==SQLITE_OK&&(revision<=0||page[used].modified_ms<0||!text_valid(page[used].title,1024,1))) rc=SQLITE_CORRUPT;
        if(rc==SQLITE_OK) page[used].revision=(uint64_t)revision;
        ++used;
    }
    int closed=sqlite3_finalize(s); if(rc==SQLITE_OK) rc=closed;
    if(rc!=SQLITE_OK) { fyodor_resource_summaries_free(page,used); return sql_result(rc); }
    *items=page; *count=used; return FYODOR_STORE_OK;
}

void fyodor_context_bundle_free(fyodor_context_bundle *bundle)
{
    if(bundle==NULL) return;
    free(bundle->text); free(bundle->sources); memset(bundle,0,sizeof(*bundle));
}

const char *fyodor_context_layer_name(fyodor_context_layer layer)
{
    static const char *names[]={"explicit","workspace","session","retrieved","global"};
    return (unsigned)layer<FYODOR_CONTEXT_LAYER_COUNT?names[layer]:NULL;
}

fyodor_store_result fyodor_context_assemble(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *refs,size_t count,size_t byte_budget,fyodor_context_bundle *out)
{
    if(count>64||(count&&refs==NULL)) return FYODOR_STORE_INVALID;
    fyodor_context_entry entries[64];
    for(size_t i=0;i<count;++i) entries[i]=(fyodor_context_entry){refs[i],FYODOR_CONTEXT_EXPLICIT};
    return fyodor_context_assemble_layers(store,principal,name_space,entries,count,byte_budget,out);
}

static fyodor_store_result assemble_layers_snapshot(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_context_entry *entries,size_t count,size_t byte_budget,fyodor_context_bundle *out,int transaction_owned)
{
    char who[FYODOR_UUID_TEXT_CAPACITY];
    if(store==NULL||out==NULL||!namespace_valid(name_space)||count>64||(count&&entries==NULL)||
        byte_budget>FYODOR_STORE_CONTENT_LIMIT||fyodor_uuid_format(principal,who,sizeof(who))!=FYODOR_RESOURCE_OK) return FYODOR_STORE_INVALID;
    fyodor_context_entry ordered[64]; size_t used=0;
    for(size_t i=0;i<count;++i) {
        char uri[FYODOR_RESOURCE_URI_CAPACITY];
        if(fyodor_context_layer_name(entries[i].layer)==NULL ||
            fyodor_resource_format(&entries[i].ref,uri,sizeof(uri))!=FYODOR_RESOURCE_OK) return FYODOR_STORE_INVALID;
        for(size_t j=0;j<i;++j)
            if(fyodor_resource_equal(&entries[i].ref,&entries[j].ref)) return FYODOR_STORE_INVALID;
    }
    for(int layer=0;layer<FYODOR_CONTEXT_LAYER_COUNT;++layer)
        for(size_t i=0;i<count;++i) if(entries[i].layer==(fyodor_context_layer)layer) ordered[used++]=entries[i];
    fyodor_context_bundle bundle={0};
    bundle.text=malloc(byte_budget+1); bundle.sources=calloc(count?count:1,sizeof(*bundle.sources));
    if(bundle.text==NULL||bundle.sources==NULL) { fyodor_context_bundle_free(&bundle); return FYODOR_STORE_NOMEM; }
    int rc=transaction_owned?SQLITE_OK:sqlite3_exec(store->db,"BEGIN",NULL,NULL,NULL);
    if(rc!=SQLITE_OK) { fyodor_context_bundle_free(&bundle); return sql_result(rc); }
    fyodor_store_result result=FYODOR_STORE_OK;
    for(size_t i=0;i<count&&result==FYODOR_STORE_OK;++i) {
        fyodor_resource_record record={0};
        result=context_read_snapshot(store,who,name_space,&ordered[i].ref,&record);
        if(result!=FYODOR_STORE_OK) break;
        if(ordered[i].layer==FYODOR_CONTEXT_RETRIEVED) {
            char uri[FYODOR_RESOURCE_URI_CAPACITY]; unsigned mask=0;
            (void)fyodor_resource_format(&ordered[i].ref,uri,sizeof(uri));
            rc=permission_read(store->db,who,name_space,uri,&mask);
            if(rc!=SQLITE_OK) result=sql_result(rc);
            else if(!(mask&FYODOR_PERMISSION_SEARCH)) result=FYODOR_STORE_DENIED;
            if(result!=FYODOR_STORE_OK) { fyodor_resource_record_free(&record); break; }
        }
        size_t original=strlen(record.content),separator=bundle.length?1u:0u;
        size_t available=byte_budget-bundle.length;
        size_t take=available>separator?available-separator:0;
        if(take>original) take=original;
        /* A UTF-8 prefix must not end inside a multibyte scalar. */
        while(take>0&&take<original&&((unsigned char)record.content[take]&0xc0u)==0x80u) --take;
        if(take&&separator) bundle.text[bundle.length++]='\n';
        bundle.sources[i]=(fyodor_context_source){ordered[i].ref,record.revision,bundle.length,take,original,ordered[i].layer};
        if(take) memcpy(bundle.text+bundle.length,record.content,take);
        bundle.length+=take; ++bundle.source_count;
        fyodor_resource_record_free(&record);
    }
    bundle.text[bundle.length]=0;
    if(!transaction_owned) {
        if(result==FYODOR_STORE_OK) result=finish(store->db,SQLITE_OK);
        else (void)sqlite3_exec(store->db,"ROLLBACK",NULL,NULL,NULL);
    }
    if(result==FYODOR_STORE_OK) *out=bundle; else fyodor_context_bundle_free(&bundle);
    return result;
}

fyodor_store_result fyodor_context_assemble_layers(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_context_entry *entries,size_t count,size_t byte_budget,fyodor_context_bundle *out)
{
    return assemble_layers_snapshot(store,principal,name_space,entries,count,byte_budget,out,0);
}

static fyodor_store_result writing_lore_context(fyodor_store *store,const char *who,const char *name_space,
    const char *metadata,fyodor_context_entry entries[64],size_t *used)
{
    fyodor_resource_ref refs[32];size_t count=0;
    int rc=writing_lore_refs(store->db,metadata,refs,&count);
    if(rc!=SQLITE_OK)return sql_result(rc);
    for(size_t i=0;i<count&&*used<64;++i) {
        int duplicate=0;
        for(size_t j=0;j<*used;++j)if(fyodor_resource_equal(&entries[j].ref,&refs[i]))duplicate=1;
        if(duplicate)continue;
        fyodor_resource_record record={0};
        fyodor_store_result result=context_read_snapshot(store,who,name_space,&refs[i],&record);
        fyodor_resource_record_free(&record);
        if(result==FYODOR_STORE_DENIED)continue;
        if(result!=FYODOR_STORE_OK)return result;
        entries[(*used)++]=(fyodor_context_entry){refs[i],FYODOR_CONTEXT_WORKSPACE};
    }
    return FYODOR_STORE_OK;
}

fyodor_store_result fyodor_writing_assemble(fyodor_store *store,const fyodor_uuid *principal,
    const char *name_space,const fyodor_resource_ref *target,uint64_t expected_revision,
    const fyodor_context_entry *extra,size_t count,size_t byte_budget,fyodor_context_bundle *out)
{
    char who[FYODOR_UUID_TEXT_CAPACITY],uri[FYODOR_RESOURCE_URI_CAPACITY];
    if(!store||!out||!target||!writing_kind(target->kind)||!expected_revision||expected_revision>INT64_MAX||
        count>63||(count&&!extra)||byte_budget>FYODOR_STORE_CONTENT_LIMIT||!context_key(principal,name_space,target,who,uri))return FYODOR_STORE_INVALID;
    int rc=sqlite3_exec(store->db,"BEGIN",NULL,NULL,NULL);
    if(rc!=SQLITE_OK)return sql_result(rc);
    fyodor_resource_record record={0};fyodor_context_bundle bundle={0};
    fyodor_store_result result=context_read_snapshot(store,who,name_space,target,&record);
    if(result==FYODOR_STORE_OK&&record.revision!=expected_revision)result=FYODOR_STORE_CONFLICT;
    fyodor_context_entry entries[64];size_t used=count+1;
    entries[0]=(fyodor_context_entry){*target,FYODOR_CONTEXT_EXPLICIT};
    for(size_t i=0;i<count;++i)entries[i+1]=extra[i];
    fyodor_resource_ref visited[65];size_t visited_count=1;visited[0]=*target;
    /* Only traverse metadata already authorized in this snapshot. A denied
     * parent ends automatic discovery; it never exposes a hidden ancestor. */
    while(result==FYODOR_STORE_OK&&used<64&&visited_count<65) {
        result=writing_lore_context(store,who,name_space,record.metadata,entries,&used);
        if(result!=FYODOR_STORE_OK||used==64)break;
        fyodor_resource_ref parent={0};int found=0;
        rc=writing_parent_ref(store->db,record.metadata,&parent,&found);
        if(rc!=SQLITE_OK) {result=sql_result(rc);break;}
        if(!found)break;
        for(size_t i=0;i<visited_count;++i)
            if(fyodor_resource_equal(&visited[i],&parent))result=FYODOR_STORE_SCHEMA_ERROR;
        if(result!=FYODOR_STORE_OK)break;
        visited[visited_count++]=parent;
        fyodor_resource_record_free(&record);
        result=context_read_snapshot(store,who,name_space,&parent,&record);
        if(result==FYODOR_STORE_DENIED) {result=FYODOR_STORE_OK;break;}
        if(result!=FYODOR_STORE_OK)break;
        int duplicate=0;
        for(size_t i=0;i<used;++i)if(fyodor_resource_equal(&entries[i].ref,&parent))duplicate=1;
        if(!duplicate)entries[used++]=(fyodor_context_entry){parent,FYODOR_CONTEXT_WORKSPACE};
    }
    fyodor_resource_record_free(&record);
    if(result==FYODOR_STORE_OK)result=assemble_layers_snapshot(store,principal,name_space,entries,used,byte_budget,&bundle,1);
    if(result==FYODOR_STORE_OK)result=finish(store->db,SQLITE_OK);
    else (void)sqlite3_exec(store->db,"ROLLBACK",NULL,NULL,NULL);
    if(result==FYODOR_STORE_OK)*out=bundle;else fyodor_context_bundle_free(&bundle);
    return result;
}

fyodor_store_result fyodor_store_export_revision(fyodor_store *store, const char *name_space,
                                       const fyodor_resource_ref *ref, uint64_t revision, char **json, size_t *length)
{
    if (json == NULL || length == NULL) return FYODOR_STORE_INVALID;
    fyodor_resource_record record = {0};
    fyodor_store_result result = fyodor_store_get(store, name_space, ref, revision, &record);
    if (result != FYODOR_STORE_OK) return result;
    char uri[FYODOR_RESOURCE_URI_CAPACITY];
    (void)fyodor_resource_format(ref, uri, sizeof(uri));
    sqlite3_stmt *statement = NULL;
    int rc = sqlite3_prepare_v2(store->db,
        "SELECT json_object('schema',1,'uri',?1,'title',?2,'content',?3,"
        "'metadata',json(?4),'provenance',json(?5))", -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_text(statement, 1, uri);
    if (rc == SQLITE_OK) rc = bind_text(statement, 2, record.title);
    if (rc == SQLITE_OK) rc = bind_text(statement, 3, record.content);
    if (rc == SQLITE_OK) rc = bind_text(statement, 4, record.metadata);
    if (rc == SQLITE_OK) rc = bind_text(statement, 5, record.provenance);
    char *package = NULL;
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(statement);
        if (rc == SQLITE_ROW) rc = column_copy(statement, 0, FYODOR_STORE_PACKAGE_LIMIT, &package);
    }
    int closed = sqlite3_finalize(statement);
    if (rc == SQLITE_OK) rc = closed;
    fyodor_resource_record_free(&record);
    if (rc != SQLITE_OK) { free(package); return sql_result(rc); }
    *json = package; *length = strlen(package);
    return FYODOR_STORE_OK;
}

fyodor_store_result fyodor_store_export(fyodor_store *store,const char *name_space,
    const fyodor_resource_ref *ref,char **json,size_t *length)
{ return fyodor_store_export_revision(store,name_space,ref,0,json,length); }

fyodor_store_result fyodor_store_import(fyodor_store *store, const char *name_space,
                                       const char *json, size_t length, uint64_t expected_revision,
                                       fyodor_resource_ref *ref, uint64_t *revision)
{
    if (store == NULL || json == NULL || ref == NULL || revision == NULL ||
        !namespace_valid(name_space) || length == 0 || length > FYODOR_STORE_PACKAGE_LIMIT ||
        expected_revision >= INT64_MAX || memchr(json, 0, length) != NULL) return FYODOR_STORE_INVALID;
    char *copy = malloc(length + 1);
    if (copy == NULL) return FYODOR_STORE_NOMEM;
    memcpy(copy, json, length); copy[length] = 0;
    if (!text_valid(copy, length, 0)) { free(copy); return FYODOR_STORE_INVALID; }
    int rc = json_object(store->db, copy);
    if (rc != SQLITE_OK) { free(copy); return rc == SQLITE_MISMATCH ? FYODOR_STORE_INVALID : sql_result(rc); }
    /* Validate shape and decoded keys before extracting C strings. json_tree
     * catches duplicates at every object depth and escaped U+0000 in strings.
     * A package cannot supply a namespace, revision policy or access grant. */
    sqlite3_stmt *statement = NULL;
    rc = sqlite3_prepare_v2(store->db,
        "SELECT (SELECT count(*) FROM json_each(?1))=6 "
        "AND NOT EXISTS(SELECT 1 FROM json_each(?1) WHERE key NOT IN('schema','uri','title','content','metadata','provenance')) "
        "AND json_type(?1,'$.schema')='integer' AND json_extract(?1,'$.schema')=1 "
        "AND json_type(?1,'$.uri')='text' AND json_type(?1,'$.title')='text' "
        "AND json_type(?1,'$.content')='text' AND json_type(?1,'$.metadata')='object' "
        "AND json_type(?1,'$.provenance')='object' "
        "AND NOT EXISTS(SELECT 1 FROM json_tree(?1) WHERE typeof(key)='text' GROUP BY parent,key HAVING count(*)>1) "
        "AND NOT EXISTS(SELECT 1 FROM json_tree(?1) WHERE (type='text' AND instr(atom,char(0))>0) "
        "OR (typeof(key)='text' AND instr(key,char(0))>0))", -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_text(statement, 1, copy);
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(statement);
        if (rc == SQLITE_ROW) rc = sqlite3_column_int(statement, 0) == 1 ? SQLITE_OK : SQLITE_MISMATCH;
    }
    (void)sqlite3_finalize(statement); statement = NULL;
    fyodor_resource_record record = {0};
    char *uri = NULL;
    if (rc == SQLITE_OK) rc = sqlite3_prepare_v2(store->db,
        "SELECT json_extract(?1,'$.uri'),json_extract(?1,'$.title'),json_extract(?1,'$.content'),"
        "json_extract(?1,'$.metadata'),json_extract(?1,'$.provenance')", -1, &statement, NULL);
    if (rc == SQLITE_OK) rc = bind_text(statement, 1, copy);
    if (rc == SQLITE_OK) {
        rc = sqlite3_step(statement);
        if (rc == SQLITE_ROW) rc = column_copy(statement, 0, FYODOR_RESOURCE_URI_CAPACITY - 1, &uri);
        if (rc == SQLITE_OK) rc = column_copy(statement, 1, 1024, &record.title);
        if (rc == SQLITE_OK) rc = column_copy(statement, 2, FYODOR_STORE_CONTENT_LIMIT, &record.content);
        if (rc == SQLITE_OK) rc = column_copy(statement, 3, FYODOR_STORE_METADATA_LIMIT, &record.metadata);
        if (rc == SQLITE_OK) rc = column_copy(statement, 4, FYODOR_STORE_METADATA_LIMIT, &record.provenance);
        if (rc == SQLITE_OK && fyodor_resource_parse(uri, strlen(uri), &record.ref) != FYODOR_RESOURCE_OK) rc = SQLITE_MISMATCH;
    }
    int closed = sqlite3_finalize(statement);
    if (rc == SQLITE_OK) rc = closed;
    free(copy); free(uri);
    fyodor_store_result result;
    if (rc == SQLITE_OK) {
        fyodor_resource_input input = {record.ref, name_space, record.title, record.content, record.metadata, record.provenance};
        uint64_t next = 0;
        result = fyodor_store_put(store, &input, expected_revision, &next);
        if (result == FYODOR_STORE_OK) { *ref = record.ref; *revision = next; }
    } else result = rc == SQLITE_MISMATCH || rc == SQLITE_CORRUPT ? FYODOR_STORE_INVALID : sql_result(rc);
    fyodor_resource_record_free(&record);
    return result;
}

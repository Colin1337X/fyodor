#include "fyodor_store.h"
#include "sqlite3.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(test) do { if (!(test)) { fprintf(stderr, "line %d: %s\n", __LINE__, #test); return 1; } } while (0)

static fyodor_resource_input input_for(fyodor_resource_kind kind)
{
    fyodor_resource_input input = {0};
    input.ref.kind = kind;
    if (fyodor_uuid_generate(&input.ref.id) != FYODOR_RESOURCE_OK) abort();
    input.name_space = "workspace";
    input.title = "First document";
    input.content = "A world begins.\n\xed\x95\x9c\xea\xb8\x80 \xf0\x9f\x8c\x8d";
    input.metadata = "{\"tags\":[\"writing\"],\"format\":\"markdown\"}";
    input.provenance = "{\"source\":\"local\"}";
    return input;
}

static int lifecycle(const char *path)
{
    fyodor_store *first = NULL, *second = NULL;
    CHECK(fyodor_store_open(path, &first) == FYODOR_STORE_OK);
    CHECK(fyodor_store_open(path, &second) == FYODOR_STORE_OK);
    fyodor_resource_input input = input_for(FYODOR_RESOURCE_DOCUMENT);
    uint64_t revision = 123;
    CHECK(fyodor_store_put(first, &input, 0, &revision) == FYODOR_STORE_OK && revision == 1);
    CHECK(fyodor_store_put(second, &input, 0, &revision) == FYODOR_STORE_CONFLICT && revision == 1);
    fyodor_resource_record record = {0};
    CHECK(fyodor_store_get(second, input.name_space, &input.ref, 0, &record) == FYODOR_STORE_OK);
    CHECK(record.revision == 1 && record.created_ms > 0 && record.modified_ms >= record.created_ms);
    int64_t created = record.created_ms;
    CHECK(strcmp(record.content, input.content) == 0 && strcmp(record.metadata, input.metadata) == 0);
    CHECK(strcmp(record.provenance, input.provenance) == 0 && fyodor_resource_equal(&record.ref, &input.ref));
    fyodor_resource_record_free(&record);
    const char *original = input.content;
    input.content = "A second revision";
    CHECK(fyodor_store_put(first, &input, 1, &revision) == FYODOR_STORE_OK && revision == 2);
    input.content = "Stale competing update";
    CHECK(fyodor_store_put(second, &input, 1, &revision) == FYODOR_STORE_CONFLICT && revision == 2);
    CHECK(fyodor_store_get(second, input.name_space, &input.ref, 0, &record) == FYODOR_STORE_OK);
    CHECK(record.revision == 2 && record.created_ms == created && strcmp(record.content, "A second revision") == 0);
    fyodor_resource_record_free(&record);
    CHECK(fyodor_store_get(second, "another", &input.ref, 0, &record) == FYODOR_STORE_NOT_FOUND);
    CHECK(fyodor_store_delete(second, input.name_space, &input.ref, 1) == FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_get(first, input.name_space, &input.ref, 1, &record) == FYODOR_STORE_OK);
    CHECK(strcmp(record.content, original) == 0);
    fyodor_resource_record_free(&record);
    fyodor_store_close(first);
    fyodor_store_close(second);
    CHECK(fyodor_store_open(path, &first) == FYODOR_STORE_OK);
    CHECK(fyodor_store_get(first, input.name_space, &input.ref, 0, &record) == FYODOR_STORE_OK);
    CHECK(record.revision == 2 && strcmp(record.content, "A second revision") == 0);
    fyodor_resource_record_free(&record);
    CHECK(fyodor_store_delete(first, input.name_space, &input.ref, 2) == FYODOR_STORE_OK);
    CHECK(fyodor_store_get(first, input.name_space, &input.ref, 0, &record) == FYODOR_STORE_NOT_FOUND);
    CHECK(fyodor_store_get(first, input.name_space, &input.ref, 3, &record) == FYODOR_STORE_OK);
    CHECK(record.deleted == 1 && record.revision == 3 && strcmp(record.content, "A second revision") == 0);
    fyodor_resource_record_free(&record);
    CHECK(fyodor_store_put(first, &input, 0, &revision) == FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_put(first, &input, 3, &revision) == FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_delete(first, input.name_space, &input.ref, 3) == FYODOR_STORE_CONFLICT);
    /* Same identity can be isolated in a separate namespace. */
    input.name_space = "another";
    CHECK(fyodor_store_put(first, &input, 0, &revision) == FYODOR_STORE_OK && revision == 1);
    fyodor_store_close(first);
    return 0;
}

static int hierarchy(const char *path)
{
    fyodor_store *store = NULL;
    CHECK(fyodor_store_open(path, &store) == FYODOR_STORE_OK);
    fyodor_resource_input world = input_for(FYODOR_RESOURCE_WORLD);
    fyodor_resource_input save = input_for(FYODOR_RESOURCE_WORLD_SAVE);
    save.ref.parent_id = world.ref.id;
    uint64_t revision = 0;
    CHECK(fyodor_store_put(store, &save, 0, &revision) == FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_put(store, &world, 0, &revision) == FYODOR_STORE_OK);
    save.name_space = "another";
    CHECK(fyodor_store_put(store, &save, 0, &revision) == FYODOR_STORE_CONFLICT);
    save.name_space = "workspace";
    CHECK(fyodor_store_put(store, &save, 0, &revision) == FYODOR_STORE_OK);
    CHECK(fyodor_store_delete(store, world.name_space, &world.ref, 1) == FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_delete(store, save.name_space, &save.ref, 1) == FYODOR_STORE_OK);
    CHECK(fyodor_store_delete(store, world.name_space, &world.ref, 1) == FYODOR_STORE_OK);
    CHECK(fyodor_uuid_generate(&save.ref.id) == FYODOR_RESOURCE_OK);
    CHECK(fyodor_store_put(store, &save, 0, &revision) == FYODOR_STORE_CONFLICT);
    fyodor_store_close(store);
    return 0;
}

static int invalid_inputs(const char *path)
{
    fyodor_store *store = NULL;
    CHECK(fyodor_store_open(path, &store) == FYODOR_STORE_OK);
    fyodor_resource_input original = input_for(FYODOR_RESOURCE_NOTE), input;
    uint64_t revision = 700;
    const char *bad_json[] = {"", "{", "[]", "null", "17", "{\"a\":NaN}", "{} trailing", "{\"a\":1,}"};
    for (size_t i = 0; i < sizeof(bad_json)/sizeof(bad_json[0]); ++i) {
        input = original; input.metadata = bad_json[i];
        CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_INVALID && revision == 700);
        input = original; input.provenance = bad_json[i];
        CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_INVALID);
    }
    const char *bad_text[] = {"\xc0\x80", "\xed\xa0\x80", "\xf4\x90\x80\x80", "\xf0", "\x80", "\xe2\x82"};
    for (size_t i = 0; i < sizeof(bad_text)/sizeof(bad_text[0]); ++i) {
        input = original; input.content = bad_text[i];
        CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_INVALID);
    }
    input = original; input.title = "bad\nname";
    CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_INVALID);
    input = original; input.name_space = "../workspace";
    CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_INVALID);
    char *large = malloc(FYODOR_STORE_CONTENT_LIMIT + 2);
    CHECK(large != NULL);
    memset(large, 'x', FYODOR_STORE_CONTENT_LIMIT + 1); large[FYODOR_STORE_CONTENT_LIMIT + 1] = 0;
    input = original; input.content = large;
    CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_INVALID);
    large[FYODOR_STORE_CONTENT_LIMIT] = 0;
    CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_OK);
    fyodor_resource_record record = {0};
    CHECK(fyodor_store_get(store, input.name_space, &input.ref, 0, &record) == FYODOR_STORE_OK);
    CHECK(strlen(record.content) == FYODOR_STORE_CONTENT_LIMIT);
    fyodor_resource_record_free(&record);
    free(large);
    CHECK(fyodor_store_put(store, &original, UINT64_MAX, &revision) == FYODOR_STORE_INVALID);
    CHECK(fyodor_store_delete(store, original.name_space, &original.ref, 0) == FYODOR_STORE_INVALID);
    CHECK(fyodor_store_open(":memory:", &store) == FYODOR_STORE_INVALID);
    CHECK(fyodor_store_open("file:test.db?mode=memory", &store) == FYODOR_STORE_INVALID);
    fyodor_store_close(store);
    return 0;
}

static int database_guards(const char *path)
{
    sqlite3 *db = NULL;
    fyodor_store *store = NULL;
    CHECK(sqlite3_open(path, &db) == SQLITE_OK);
    CHECK(sqlite3_exec(db, "BEGIN IMMEDIATE", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(fyodor_store_open(path, &store) == FYODOR_STORE_BUSY && store == NULL);
    CHECK(sqlite3_exec(db, "ROLLBACK; PRAGMA user_version=999;", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(fyodor_store_open(path, &store) == FYODOR_STORE_SCHEMA_ERROR && store == NULL);
    CHECK(sqlite3_exec(db, "PRAGMA user_version=1; PRAGMA application_id=123;", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(fyodor_store_open(path, &store) == FYODOR_STORE_SCHEMA_ERROR);
    CHECK(sqlite3_exec(db, "PRAGMA user_version=0; PRAGMA application_id=0;", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(fyodor_store_open(path, &store) == FYODOR_STORE_SCHEMA_ERROR);
    CHECK(sqlite3_exec(db, "PRAGMA user_version=4; PRAGMA application_id=1180255314;", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(sqlite3_exec(db, "CREATE TRIGGER surprise AFTER INSERT ON revisions BEGIN SELECT 1; END", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(fyodor_store_open(path, &store) == FYODOR_STORE_SCHEMA_ERROR);
    CHECK(sqlite3_exec(db, "DROP TRIGGER surprise", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(sqlite3_close(db) == SQLITE_OK);
    CHECK(fyodor_store_open(path, &store) == FYODOR_STORE_OK);
    fyodor_resource_input input = input_for(FYODOR_RESOURCE_DOCUMENT);
    uint64_t revision = 0;
    CHECK(sqlite3_open(path, &db) == SQLITE_OK);
    CHECK(sqlite3_exec(db, "BEGIN IMMEDIATE", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_BUSY);
    CHECK(sqlite3_exec(db, "ROLLBACK", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_OK && revision == 1);
    /* Inject a history failure after the head write: the whole transaction
     * must roll back. This raw connection is test-only, never public API. */
    CHECK(sqlite3_exec(db, "CREATE TRIGGER fail_history BEFORE INSERT ON revisions "
        "BEGIN SELECT RAISE(ABORT,'injected history failure'); END", NULL, NULL, NULL) == SQLITE_OK);
    input.content = "must roll back";
    CHECK(fyodor_store_put(store, &input, 1, &revision) == FYODOR_STORE_CONFLICT);
    fyodor_resource_record record = {0};
    CHECK(fyodor_store_get(store, input.name_space, &input.ref, 0, &record) == FYODOR_STORE_OK);
    CHECK(record.revision == 1 && strcmp(record.content, "must roll back") != 0);
    fyodor_resource_record_free(&record);
    CHECK(fyodor_store_get(store, input.name_space, &input.ref, 2, &record) == FYODOR_STORE_NOT_FOUND);
    CHECK(fyodor_store_delete(store, input.name_space, &input.ref, 1) == FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_get(store, input.name_space, &input.ref, 0, &record) == FYODOR_STORE_OK);
    CHECK(record.revision == 1 && record.deleted == 0);
    fyodor_resource_record_free(&record);
    CHECK(sqlite3_exec(db, "DROP TRIGGER fail_history", NULL, NULL, NULL) == SQLITE_OK);
    CHECK(sqlite3_close(db) == SQLITE_OK);
    fyodor_store_close(store);
    return 0;
}

/* Separate-process durability fixture. The crash path leaves a deliberately
 * uncommitted, spilled transaction and exits without SQLite cleanup. */
static int process_fixture(const char *mode, const char *path)
{
    if (strcmp(path, "__unicode__") == 0) path = "workspace-\xed\x95\x9c\xea\xb8\x80-\xf0\x9f\x8c\x8d.db";
    fyodor_resource_input input = input_for(FYODOR_RESOURCE_DOCUMENT);
    CHECK(fyodor_uuid_parse("01234567-89ab-4cde-8f01-23456789abcd", 36, &input.ref.id) == FYODOR_RESOURCE_OK);
    input.content = "durable baseline";
    if (strcmp(mode, "--crash") == 0) {
        sqlite3 *db = NULL;
        CHECK(sqlite3_open(path, &db) == SQLITE_OK);
        CHECK(sqlite3_exec(db, "PRAGMA cache_size=2; BEGIN IMMEDIATE; "
            "UPDATE resources SET content=printf('%.*c',1048576,120),title='uncommitted';",
            NULL, NULL, NULL) == SQLITE_OK);
        CHECK(sqlite3_db_cacheflush(db) == SQLITE_OK);
        _Exit(23);
    }
    fyodor_store *store = NULL;
    CHECK(fyodor_store_open(path, &store) == FYODOR_STORE_OK);
    if (strcmp(mode, "--seed") == 0) {
        uint64_t revision = 0;
        CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_OK && revision == 1);
    } else if (strcmp(mode, "--update") == 0) {
        uint64_t revision = 0;
        input.content = "winner";
        fyodor_store_result result = fyodor_store_put(store, &input, 1, &revision);
        fyodor_store_close(store);
        if (result == FYODOR_STORE_CONFLICT) return 24;
        CHECK(result == FYODOR_STORE_OK && revision == 2);
        return 0;
    } else if (strcmp(mode, "--verify") == 0 || strcmp(mode, "--verify-updated") == 0) {
        int updated = strcmp(mode, "--verify-updated") == 0;
        fyodor_resource_record record = {0};
        CHECK(fyodor_store_get(store, input.name_space, &input.ref, 0, &record) == FYODOR_STORE_OK);
        CHECK(record.revision == (updated ? 2u : 1u) && strcmp(record.content, updated ? "winner" : input.content) == 0);
        CHECK(strcmp(record.title, input.title) == 0);
        fyodor_resource_record_free(&record);
        CHECK(fyodor_store_get(store, input.name_space, &input.ref, 1, &record) == FYODOR_STORE_OK);
        CHECK(strcmp(record.content, input.content) == 0);
        fyodor_resource_record_free(&record);
    } else { fyodor_store_close(store); return 2; }
    fyodor_store_close(store);
    return 0;
}

static int history_pages(const char *path)
{
    fyodor_store *store=NULL;CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    fyodor_resource_input input=input_for(FYODOR_RESOURCE_CHARACTER);input.name_space="history";
    uint64_t revision=0;
    for(uint64_t i=0;i<25;++i) {
        input.content=i==0?"first text":"later text";
        CHECK(fyodor_store_put(store,&input,i,&revision)==FYODOR_STORE_OK&&revision==i+1);
    }
    fyodor_revision_summary *page=NULL;size_t count=0;
    CHECK(fyodor_store_history(store,"history",&input.ref,0,20,&page,&count)==FYODOR_STORE_OK&&count==20);
    for(size_t i=0;i<count;++i) CHECK(page[i].revision==25-i&&!page[i].deleted);
    uint64_t before=page[count-1].revision;free(page);page=NULL;
    CHECK(fyodor_store_history(store,"history",&input.ref,before,20,&page,&count)==FYODOR_STORE_OK&&count==5);
    CHECK(page[0].revision==5&&page[4].revision==1);free(page);page=NULL;
    CHECK(fyodor_store_history(store,"history",&input.ref,1,20,&page,&count)==FYODOR_STORE_OK&&count==0);free(page);page=NULL;
    CHECK(fyodor_store_history(store,"elsewhere",&input.ref,0,20,&page,&count)==FYODOR_STORE_OK&&count==0);free(page);page=NULL;
    count=999;
    CHECK(fyodor_store_history(store,"history",&input.ref,0,0,&page,&count)==FYODOR_STORE_INVALID&&page==NULL&&count==999);
    CHECK(fyodor_store_delete(store,"history",&input.ref,25)==FYODOR_STORE_OK);
    CHECK(fyodor_store_history(store,"history",&input.ref,0,1,&page,&count)==FYODOR_STORE_OK&&count==1&&page[0].deleted&&page[0].revision==26);free(page);
    char *package=NULL;size_t length=0;
    CHECK(fyodor_store_export_revision(store,"history",&input.ref,1,&package,&length)==FYODOR_STORE_OK);
    CHECK(strstr(package,"first text")!=NULL);fyodor_resource_package_free(package);package=NULL;
    CHECK(fyodor_store_export(store,"history",&input.ref,&package,&length)==FYODOR_STORE_NOT_FOUND&&package==NULL);
    CHECK(fyodor_store_export_revision(store,"history",&input.ref,27,&package,&length)==FYODOR_STORE_NOT_FOUND);
    /* Filtering is applied before the page limit, not after fetching a mixed page. */
    input=input_for(FYODOR_RESOURCE_WORLD);input.name_space="writing-page";
    CHECK(fyodor_store_put(store,&input,0,&revision)==FYODOR_STORE_OK);
    input=input_for(FYODOR_RESOURCE_CHARACTER);input.name_space="writing-page";
    CHECK(fyodor_store_put(store,&input,0,&revision)==FYODOR_STORE_OK);
    fyodor_resource_summary *items=NULL;
    CHECK(fyodor_store_list_writing(store,"writing-page",NULL,1,&items,&count)==FYODOR_STORE_OK&&count==1&&items[0].ref.kind==FYODOR_RESOURCE_CHARACTER);
    fyodor_resource_summaries_free(items,count);fyodor_store_close(store);return 0;
}

static int writing_folders(const char *path)
{
    fyodor_store *store=NULL;CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    fyodor_resource_input project=input_for(FYODOR_RESOURCE_PROJECT),child=input_for(FYODOR_RESOURCE_PROJECT),doc=input_for(FYODOR_RESOURCE_DOCUMENT);
    project.name_space=child.name_space=doc.name_space="folders";
    doc.metadata="{\"large\":9007199254740993}";
    uint64_t revision=0;
    CHECK(fyodor_store_put(store,&project,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_put(store,&child,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_put(store,&doc,0,&revision)==FYODOR_STORE_OK);
    fyodor_resource_input large=input_for(FYODOR_RESOURCE_NOTE);large.name_space="folder-limit";
    char *metadata=malloc(FYODOR_STORE_METADATA_LIMIT+1);CHECK(metadata);
    memcpy(metadata,"{\"blob\":\"",9);memset(metadata+9,'x',FYODOR_STORE_METADATA_LIMIT-11);memcpy(metadata+FYODOR_STORE_METADATA_LIMIT-2,"\"}",3);
    large.metadata=metadata;
    CHECK(fyodor_store_put(store,&large,0,&revision)==FYODOR_STORE_OK);
    revision=999;
    CHECK(fyodor_store_move_writing(store,"folder-limit",&large.ref,1,&project.ref,&revision)==FYODOR_STORE_INVALID&&revision==999);
    free(metadata);
    CHECK(fyodor_store_move_writing(store,"folders",&child.ref,1,&project.ref,&revision)==FYODOR_STORE_OK&&revision==2);
    CHECK(fyodor_store_move_writing(store,"folders",&doc.ref,1,&child.ref,&revision)==FYODOR_STORE_OK&&revision==2);
    revision=999;
    CHECK(fyodor_store_move_writing(store,"folders",&project.ref,1,&child.ref,&revision)==FYODOR_STORE_CONFLICT&&revision==999);
    CHECK(fyodor_store_move_writing(store,"folders",&project.ref,1,&project.ref,&revision)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_move_writing(store,"folders",&doc.ref,1,NULL,&revision)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_move_writing(store,"folders",&doc.ref,2,&doc.ref,&revision)==FYODOR_STORE_INVALID);
    CHECK(fyodor_store_delete(store,"folders",&project.ref,1)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_delete(store,"folders",&child.ref,2)==FYODOR_STORE_CONFLICT);
    fyodor_resource_summary *items=NULL;size_t count=0;
    CHECK(fyodor_store_list_folder(store,"folders",NULL,NULL,1,&items,&count)==FYODOR_STORE_OK&&count==1&&fyodor_resource_equal(&items[0].ref,&project.ref));
    fyodor_resource_summaries_free(items,count);
    CHECK(fyodor_store_list_folder(store,"folders",&child.ref,NULL,1,&items,&count)==FYODOR_STORE_OK&&count==1&&fyodor_resource_equal(&items[0].ref,&doc.ref));
    fyodor_resource_summaries_free(items,count);
    char uri[FYODOR_RESOURCE_URI_CAPACITY];CHECK(fyodor_resource_format(&doc.ref,uri,sizeof(uri))==FYODOR_RESOURCE_OK);
    CHECK(fyodor_store_list_folder(store,"folders",&child.ref,uri,1,&items,&count)==FYODOR_STORE_OK&&count==0);fyodor_resource_summaries_free(items,count);
    fyodor_resource_record record={0};
    CHECK(fyodor_store_get(store,"folders",&doc.ref,0,&record)==FYODOR_STORE_OK);
    CHECK(strstr(record.metadata,"9007199254740993")&&strstr(record.metadata,"writing_parent"));
    CHECK(strcmp(record.content,doc.content)==0&&strcmp(record.provenance,doc.provenance)==0);fyodor_resource_record_free(&record);
    CHECK(fyodor_store_get(store,"folders",&doc.ref,1,&record)==FYODOR_STORE_OK);
    CHECK(!strstr(record.metadata,"writing_parent"));fyodor_resource_record_free(&record);
    char *package=NULL;size_t length=0;fyodor_resource_ref imported;
    CHECK(fyodor_store_export(store,"folders",&doc.ref,&package,&length)==FYODOR_STORE_OK);
    CHECK(fyodor_store_import(store,"other-folders",package,length,0,&imported,&revision)==FYODOR_STORE_CONFLICT);
    child.name_space="other-folders";
    CHECK(fyodor_store_put(store,&child,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_import(store,"other-folders",package,length,0,&imported,&revision)==FYODOR_STORE_OK);
    fyodor_resource_package_free(package);
    doc.metadata="{\"writing_parent\":42}";
    CHECK(fyodor_store_put(store,&doc,2,&revision)==FYODOR_STORE_CONFLICT);
    /* Inject a history failure after the metadata/index write. */
    sqlite3 *db=NULL;CHECK(sqlite3_open(path,&db)==SQLITE_OK);
    CHECK(sqlite3_exec(db,"CREATE TRIGGER fail_move BEFORE INSERT ON revisions BEGIN SELECT RAISE(ABORT,'injected'); END",NULL,NULL,NULL)==SQLITE_OK);
    revision=999;
    CHECK(fyodor_store_move_writing(store,"folders",&doc.ref,2,NULL,&revision)==FYODOR_STORE_CONFLICT&&revision==999);
    CHECK(fyodor_store_get(store,"folders",&doc.ref,0,&record)==FYODOR_STORE_OK&&record.revision==2&&strstr(record.metadata,"writing_parent"));fyodor_resource_record_free(&record);
    CHECK(sqlite3_exec(db,"DROP TRIGGER fail_move",NULL,NULL,NULL)==SQLITE_OK);CHECK(sqlite3_close(db)==SQLITE_OK);
    CHECK(fyodor_store_move_writing(store,"folders",&doc.ref,2,NULL,&revision)==FYODOR_STORE_OK&&revision==3);
    CHECK(fyodor_store_delete(store,"folders",&child.ref,2)==FYODOR_STORE_OK);
    CHECK(fyodor_store_move_writing(store,"folders",&doc.ref,3,&child.ref,&revision)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_delete(store,"folders",&project.ref,1)==FYODOR_STORE_OK);
    fyodor_store_close(store);CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    CHECK(fyodor_store_get(store,"folders",&doc.ref,0,&record)==FYODOR_STORE_OK&&record.revision==3&&!strstr(record.metadata,"writing_parent"));
    fyodor_resource_record_free(&record);fyodor_store_close(store);return 0;
}

int main(int argc, char **argv)
{
    if (argc == 3) return process_fixture(argv[1], argv[2]);
    if (argc != 2) return 2;
    /* This exact path belongs to the test, never a user-selected resource DB. */
    (void)remove(argv[1]);
    if (lifecycle(argv[1]) || hierarchy(argv[1]) || invalid_inputs(argv[1]) || database_guards(argv[1]) || history_pages(argv[1]) || writing_folders(argv[1])) return 1;
    CHECK(remove(argv[1]) == 0);
    puts("resource store: persistence, history, CAS, namespaces, parent lifecycle, limits and schema guards passed");
    return 0;
}

#include "fyodor_context.h"
#include "sqlite3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while(0)

/* Independent scan oracle: compare ordered IDs and revisions, not just counts. */
static int compare(fyodor_store *store,sqlite3 *db,const fyodor_uuid *principal,
    const char *name_space,const char *query,size_t limit)
{
    char who[FYODOR_UUID_TEXT_CAPACITY];
    CHECK(fyodor_uuid_format(principal,who,sizeof(who))==FYODOR_RESOURCE_OK);
    fyodor_resource_summary *items=NULL;size_t count=0;
    CHECK(fyodor_context_search(store,principal,name_space,query,limit,&items,&count)==FYODOR_STORE_OK);
    sqlite3_stmt *s=NULL;
    CHECK(sqlite3_prepare_v2(db,"SELECT r.uri,r.revision FROM resources r JOIN context_permissions p "
        "USING(namespace,uri) WHERE r.namespace=?1 AND p.principal=?2 AND (p.permissions&5)=5 AND r.deleted=0 "
        "AND (instr(lower(r.title),lower(?3))>0 OR instr(lower(r.content),lower(?3))>0) "
        "ORDER BY (instr(lower(r.title),lower(?3))>0) DESC,r.uri LIMIT ?4",-1,&s,NULL)==SQLITE_OK);
    CHECK(sqlite3_bind_text(s,1,name_space,-1,SQLITE_STATIC)==SQLITE_OK);
    CHECK(sqlite3_bind_text(s,2,who,-1,SQLITE_STATIC)==SQLITE_OK);
    CHECK(sqlite3_bind_text(s,3,query,-1,SQLITE_STATIC)==SQLITE_OK);
    CHECK(sqlite3_bind_int(s,4,(int)limit)==SQLITE_OK);
    size_t i=0;int rc;
    while((rc=sqlite3_step(s))==SQLITE_ROW) {
        CHECK(i<count);
        char uri[FYODOR_RESOURCE_URI_CAPACITY];
        CHECK(fyodor_resource_format(&items[i].ref,uri,sizeof(uri))==FYODOR_RESOURCE_OK);
        CHECK(strcmp(uri,(const char *)sqlite3_column_text(s,0))==0);
        CHECK(items[i].revision==(uint64_t)sqlite3_column_int64(s,1));++i;
    }
    CHECK(rc==SQLITE_DONE&&i==count);
    CHECK(sqlite3_finalize(s)==SQLITE_OK);fyodor_resource_summaries_free(items,count);return 0;
}

static int corpus(fyodor_store *store,sqlite3 *db,const fyodor_uuid *principal)
{
    const char *queries[]={"a","ab","abc","ABC","avalon","VAL","%","_","100%", "a_b",
        "\"","\"qu","\"quoted\"","OR"," OR ","NOT","NEAR(","title:","*","%_",
        "\xed\x95\x9c","\xed\x95\x9c\xea\xb8\x80","\xed\x95\x9c\xea\xb8\x80\xeb\xa7\x90",
        "\xc3\x84" "bc","\xc3\xa4" "bc","\xf0\x9f\x8c\x8d" "ab","missingneedle","   "};
    const size_t limits[]={1,7,100};
    for(size_t i=0;i<sizeof(queries)/sizeof(queries[0]);++i)
        for(size_t j=0;j<sizeof(limits)/sizeof(limits[0]);++j)
            CHECK(compare(store,db,principal,"workspace",queries[i],limits[j])==0);
    CHECK(compare(store,db,principal,"other","abc",100)==0);
    char quotes[129];memset(quotes,'"',128);quotes[128]=0;
    CHECK(compare(store,db,principal,"workspace",quotes,100)==0);
    return 0;
}

int main(int argc,char **argv)
{
    if(argc!=2) return 2;
    (void)remove(argv[1]);fyodor_store *store=NULL;sqlite3 *db=NULL;
    CHECK(fyodor_store_open(argv[1],&store)==FYODOR_STORE_OK);
    CHECK(sqlite3_open(argv[1],&db)==SQLITE_OK);
    fyodor_uuid principal;CHECK(fyodor_uuid_generate(&principal)==FYODOR_RESOURCE_OK);
    fyodor_resource_input input={0};input.ref.kind=FYODOR_RESOURCE_DOCUMENT;
    input.name_space="workspace";input.metadata="{}";input.provenance="{}";
    for(int i=0;i<42;++i) {
        CHECK(fyodor_uuid_generate(&input.ref.id)==FYODOR_RESOURCE_OK);
        input.name_space=i==41?"other":"workspace";
        input.title=i%3?"ordinary":"Avalon ABC \"quoted\"";
        input.content=i%2?"Avalon abc 100% a_b OR NOT NEAR( title: *   ":
            "\xed\x95\x9c\xea\xb8\x80\xeb\xa7\x90 \xc3\x84" "bc \xc3\xa4" "bc \xf0\x9f\x8c\x8d" "ab";
        uint64_t revision=0;
        CHECK(fyodor_store_put(store,&input,0,&revision)==FYODOR_STORE_OK);
        CHECK(fyodor_context_permissions_set(store,&principal,input.name_space,&input.ref,i%5?5:1)==FYODOR_STORE_OK);
    }
    CHECK(corpus(store,db,&principal)==0);
    CHECK(sqlite3_exec(db,"CREATE TRIGGER fail_index_key BEFORE INSERT ON resource_search_keys "
        "BEGIN SELECT RAISE(ABORT,'injected index failure'); END",NULL,NULL,NULL)==SQLITE_OK);
    fyodor_resource_input rejected=input;
    CHECK(fyodor_uuid_generate(&rejected.ref.id)==FYODOR_RESOURCE_OK);
    uint64_t unchanged=999;
    CHECK(fyodor_store_put(store,&rejected,0,&unchanged)==FYODOR_STORE_CONFLICT&&unchanged==999);
    fyodor_resource_record absent={0};
    CHECK(fyodor_store_get(store,rejected.name_space,&rejected.ref,0,&absent)==FYODOR_STORE_NOT_FOUND);
    CHECK(fyodor_store_get(store,rejected.name_space,&rejected.ref,1,&absent)==FYODOR_STORE_NOT_FOUND);
    CHECK(sqlite3_exec(db,"DROP TRIGGER fail_index_key",NULL,NULL,NULL)==SQLITE_OK);
    input.name_space="workspace";input.title="atomic";input.content="oldneedle";
    CHECK(fyodor_uuid_generate(&input.ref.id)==FYODOR_RESOURCE_OK);
    uint64_t revision=0;
    CHECK(fyodor_store_put(store,&input,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&input.ref,63)==FYODOR_STORE_OK);
    CHECK(sqlite3_exec(db,"CREATE TRIGGER fail_index_history BEFORE INSERT ON revisions "
        "BEGIN SELECT RAISE(ABORT,'injected'); END",NULL,NULL,NULL)==SQLITE_OK);
    input.content="newneedle";revision=777;
    CHECK(fyodor_store_put(store,&input,1,&revision)==FYODOR_STORE_CONFLICT&&revision==777);
    CHECK(fyodor_context_append(store,&principal,"workspace",&input.ref,"newneedle",1,&revision)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_delete(store,"workspace",&input.ref,1)==FYODOR_STORE_CONFLICT);
    CHECK(compare(store,db,&principal,"workspace","oldneedle",100)==0);
    CHECK(compare(store,db,&principal,"workspace","newneedle",100)==0);
    CHECK(sqlite3_exec(db,"DROP TRIGGER fail_index_history",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(fyodor_context_write(store,&principal,&input,1,&revision)==FYODOR_STORE_OK&&revision==2);
    CHECK(compare(store,db,&principal,"workspace","oldneedle",100)==0);
    CHECK(compare(store,db,&principal,"workspace","newneedle",100)==0);
    CHECK(fyodor_context_append(store,&principal,"workspace",&input.ref,"appendneedle",2,&revision)==FYODOR_STORE_OK);
    CHECK(compare(store,db,&principal,"workspace","appendneedle",100)==0);
    CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&input.ref,1)==FYODOR_STORE_OK);
    CHECK(compare(store,db,&principal,"workspace","newneedle",100)==0);
    CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&input.ref,63)==FYODOR_STORE_OK);
    CHECK(fyodor_context_delete(store,&principal,"workspace",&input.ref,3)==FYODOR_STORE_OK);
    CHECK(compare(store,db,&principal,"workspace","newneedle",100)==0);
    /* Check the index itself, since final live-head filtering can hide stale postings. */
    sqlite3_stmt *s=NULL;
    CHECK(sqlite3_prepare_v2(db,"SELECT count(*) FROM resource_search('newneedle')",-1,&s,NULL)==SQLITE_OK);
    CHECK(sqlite3_step(s)==SQLITE_ROW&&sqlite3_column_int(s,0)==0);CHECK(sqlite3_finalize(s)==SQLITE_OK);
    CHECK(sqlite3_exec(db,"INSERT INTO resource_search(resource_search) VALUES('integrity-check')",NULL,NULL,NULL)==SQLITE_OK);
    fyodor_store_close(store);store=NULL;
    CHECK(sqlite3_exec(db,"VACUUM",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(fyodor_store_open(argv[1],&store)==FYODOR_STORE_OK);
    CHECK(corpus(store,db,&principal)==0);
    fyodor_store_close(store);store=NULL;
    /* Reconstruct exact v3 storage and migrate populated data with existing grants. */
    CHECK(sqlite3_exec(db,"DROP TABLE resource_search; DROP TABLE resource_search_keys; PRAGMA user_version=3",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(fyodor_store_open(argv[1],&store)==FYODOR_STORE_OK);
    CHECK(corpus(store,db,&principal)==0);
    CHECK(compare(store,db,&principal,"workspace","newneedle",100)==0);
    CHECK(sqlite3_exec(db,"INSERT INTO resource_search(resource_search) VALUES('integrity-check')",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(sqlite3_close(db)==SQLITE_OK);fyodor_store_close(store);
    CHECK(remove(argv[1])==0);
    puts("indexed context: scan parity, literal/Unicode matching, grants, transactional mutations, VACUUM and v3 migration passed");
    return 0;
}

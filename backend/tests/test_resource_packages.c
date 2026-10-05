#include "fyodor_store.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while (0)

static const char *uri = "fyodor://writing/documents/01234567-89ab-4cde-8f01-23456789abcd";

static int roundtrip(fyodor_store *store)
{
    fyodor_resource_input input = {0};
    CHECK(fyodor_resource_parse(uri, strlen(uri), &input.ref) == FYODOR_RESOURCE_OK);
    input.name_space = "source"; input.title = "Quote \" & <script>";
    input.content = "new\nline\ttab\\slash\x01\xed\x95\x9c";
    input.metadata = "{\"tags\":[\"a\",\"b\"],\"nested\":{\"ok\":true}}";
    input.provenance = "{\"source\":\"untrusted reference\"}";
    uint64_t revision = 0;
    CHECK(fyodor_store_put(store, &input, 0, &revision) == FYODOR_STORE_OK);
    char *package = NULL; size_t length = 0;
    CHECK(fyodor_store_export(store, "source", &input.ref, &package, &length) == FYODOR_STORE_OK);
    CHECK(strlen(package) == length && strstr(package,"namespace") == NULL);
    fyodor_resource_ref imported = {0};
    /* Deliberately pass a non-NUL-terminated input allocation. */
    char *bounded = malloc(length);
    CHECK(bounded != NULL); memcpy(bounded, package, length);
    CHECK(fyodor_store_import(store,"destination",bounded,length,0,&imported,&revision) == FYODOR_STORE_OK);
    free(bounded);
    CHECK(fyodor_resource_equal(&imported,&input.ref) && revision == 1);
    CHECK(fyodor_store_import(store,"destination",package,length,0,&imported,&revision) == FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_import(store,"destination",package,length,1,&imported,&revision) == FYODOR_STORE_OK && revision == 2);
    fyodor_resource_record record = {0};
    CHECK(fyodor_store_get(store,"destination",&imported,0,&record) == FYODOR_STORE_OK);
    CHECK(strcmp(record.content,input.content) == 0 && strcmp(record.title,input.title) == 0);
    CHECK(strcmp(record.metadata,input.metadata) == 0 && strcmp(record.provenance,input.provenance) == 0);
    fyodor_resource_record_free(&record);
    fyodor_resource_package_free(package);
    /* Worst-case JSON expansion: six output bytes per content byte. */
    char *large = malloc(FYODOR_STORE_CONTENT_LIMIT + 1);
    CHECK(large != NULL); memset(large,1,FYODOR_STORE_CONTENT_LIMIT); large[FYODOR_STORE_CONTENT_LIMIT] = 0;
    input.content = large;
    CHECK(fyodor_store_put(store,&input,1,&revision) == FYODOR_STORE_OK);
    CHECK(fyodor_store_export(store,"source",&input.ref,&package,&length) == FYODOR_STORE_OK);
    CHECK(length > 6u * FYODOR_STORE_CONTENT_LIMIT);
    CHECK(fyodor_store_import(store,"large",package,length,0,&imported,&revision) == FYODOR_STORE_OK);
    CHECK(fyodor_store_get(store,"large",&imported,0,&record) == FYODOR_STORE_OK);
    CHECK(strlen(record.content) == FYODOR_STORE_CONTENT_LIMIT && memcmp(record.content,large,FYODOR_STORE_CONTENT_LIMIT) == 0);
    fyodor_resource_record_free(&record); free(large); fyodor_resource_package_free(package);
    return 0;
}

static int malformed(fyodor_store *store)
{
    char json[2048];
    const char *bad[] = {
        "\"schema\":2", "\"schema\":1.0", "\"schema\":\"1\"", "\"schema\":1,\"namespace\":\"stolen\"",
        "\"schema\":1,\"revision\":99", "\"schema\":1,\"schema\":1",
        "\"schema\":1,\"\\u0073chema\":1", "\"schema\":1,\"permissions\":[\"all\"]"
    };
    fyodor_resource_ref output = {0}; output.kind = FYODOR_RESOURCE_NODE;
    uint64_t revision = 777;
    for (size_t i=0;i<sizeof(bad)/sizeof(bad[0]);++i) {
        snprintf(json,sizeof(json),"{%s,\"uri\":\"%s\",\"title\":\"t\",\"content\":\"c\",\"metadata\":{},\"provenance\":{}}",bad[i],uri);
        CHECK(fyodor_store_import(store,"rejected",json,strlen(json),0,&output,&revision) == FYODOR_STORE_INVALID);
        CHECK(revision == 777 && output.kind == FYODOR_RESOURCE_NODE);
    }
    const char *fields[] = {
        "\"title\":\"t\",\"content\":\"a\\u0000b\",\"metadata\":{},\"provenance\":{}",
        "\"title\":\"t\",\"content\":\"c\",\"metadata\":{\"a\":1,\"a\":2},\"provenance\":{}",
        "\"title\":\"t\",\"content\":\"c\",\"metadata\":{\"nested\":{\"a\":1,\"\\u0061\":2}},\"provenance\":{}",
        "\"title\":\"t\",\"content\":\"c\",\"metadata\":{\"a\\u0000b\":1},\"provenance\":{}",
        "\"title\":\"t\",\"content\":{},\"metadata\":{},\"provenance\":{}",
        "\"title\":\"t\",\"content\":\"c\",\"metadata\":[],\"provenance\":{}",
        "\"title\":\"t\",\"content\":\"c\",\"metadata\":{}",
        "\"title\":\"bad\\nname\",\"content\":\"c\",\"metadata\":{},\"provenance\":{}"
    };
    for(size_t i=0;i<sizeof(fields)/sizeof(fields[0]);++i) {
        snprintf(json,sizeof(json),"{\"schema\":1,\"uri\":\"%s\",%s}",uri,fields[i]);
        CHECK(fyodor_store_import(store,"rejected",json,strlen(json),0,&output,&revision) == FYODOR_STORE_INVALID);
    }
    CHECK(fyodor_store_import(store,"rejected","{}\0evil",7,0,&output,&revision) == FYODOR_STORE_INVALID);
    CHECK(fyodor_store_import(store,"rejected","{}",SIZE_MAX,0,&output,&revision) == FYODOR_STORE_INVALID);
    CHECK(fyodor_store_import(store,"rejected","{\"x\":\"\xc0\x80\"}",10,0,&output,&revision) == FYODOR_STORE_INVALID);
    fyodor_resource_summary *items = NULL; size_t count = 999;
    CHECK(fyodor_store_list(store,"rejected",NULL,100,&items,&count) == FYODOR_STORE_OK && count == 0);
    fyodor_resource_summaries_free(items,count);
    return 0;
}

static int listing(fyodor_store *store)
{
    fyodor_resource_input input = {0}; input.ref.kind = FYODOR_RESOURCE_DOCUMENT;
    input.name_space="pages"; input.title="title"; input.content="body"; input.metadata="{}"; input.provenance="{}";
    for (unsigned i=1;i<=105;++i) {
        memset(input.ref.id.bytes,0,16); input.ref.id.bytes[15]=(uint8_t)i;
        uint64_t revision=0;
        CHECK(fyodor_store_put(store,&input,0,&revision) == FYODOR_STORE_OK);
        if (i%10==0) CHECK(fyodor_store_delete(store,"pages",&input.ref,1) == FYODOR_STORE_OK);
    }
    size_t total=0; char cursor[FYODOR_RESOURCE_URI_CAPACITY]="";
    for (;;) {
        fyodor_resource_summary *items=NULL; size_t count=0;
        CHECK(fyodor_store_list(store,"pages",cursor[0]?cursor:NULL,17,&items,&count) == FYODOR_STORE_OK);
        if (count==0) { fyodor_resource_summaries_free(items,count); break; }
        CHECK(count<=17);
        for(size_t i=0;i<count;++i) {
            char next[FYODOR_RESOURCE_URI_CAPACITY];
            CHECK(fyodor_resource_format(&items[i].ref,next,sizeof(next)) == FYODOR_RESOURCE_OK);
            CHECK(strcmp(next,cursor)>0 && items[i].revision==1 && strcmp(items[i].title,"title")==0);
            memcpy(cursor,next,strlen(next)+1); ++total;
        }
        fyodor_resource_summaries_free(items,count);
    }
    CHECK(total==95);
    fyodor_resource_summary *items=NULL; size_t count=456;
    CHECK(fyodor_store_list(store,"pages",NULL,101,&items,&count) == FYODOR_STORE_INVALID && count==456 && items==NULL);
    CHECK(fyodor_store_list(store,"pages","not-a-uri",1,&items,&count) == FYODOR_STORE_INVALID);
    return 0;
}

int main(int argc,char **argv)
{
    if(argc!=2) return 2;
    (void)remove(argv[1]);
    fyodor_store *store=NULL;
    CHECK(fyodor_store_open(argv[1],&store) == FYODOR_STORE_OK);
    if(roundtrip(store)||malformed(store)||listing(store)) return 1;
    fyodor_store_close(store);
    CHECK(remove(argv[1])==0);
    puts("resource packages: bounded pages, roundtrip, expansion limits and malformed-input rejection passed");
    return 0;
}

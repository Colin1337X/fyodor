#include "fyodor_context.h"
#include "sqlite3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); return 1; } } while(0)

static fyodor_resource_input document(const char *title,const char *content)
{
    fyodor_resource_input input={0}; input.ref.kind=FYODOR_RESOURCE_DOCUMENT;
    if(fyodor_uuid_generate(&input.ref.id)!=FYODOR_RESOURCE_OK) abort();
    input.name_space="workspace"; input.title=title; input.content=content;
    input.metadata="{}"; input.provenance="{}"; return input;
}

static int exercise(const char *path)
{
    fyodor_store *store=NULL;
    CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    fyodor_uuid alice,bob; CHECK(fyodor_uuid_generate(&alice)==FYODOR_RESOURCE_OK);
    CHECK(fyodor_uuid_generate(&bob)==FYODOR_RESOURCE_OK);
    fyodor_resource_input a=document("Avalon title","abc\xed\x95\x9c\xf0\x9f\x8c\x8d");
    fyodor_resource_input b=document("ordinary title","Avalon secret");
    uint64_t revision=0;
    CHECK(fyodor_store_put(store,&a,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_put(store,&b,0,&revision)==FYODOR_STORE_OK);
    fyodor_resource_record record={0};
    CHECK(fyodor_context_read(store,&alice,"workspace",&a.ref,&record)==FYODOR_STORE_DENIED && record.content==NULL);
    CHECK(fyodor_context_permissions_set(store,&alice,"workspace",&a.ref,FYODOR_PERMISSION_READ)==FYODOR_STORE_OK);
    CHECK(fyodor_context_read(store,&alice,"workspace",&a.ref,&record)==FYODOR_STORE_OK);
    CHECK(strcmp(record.content,a.content)==0); fyodor_resource_record_free(&record);
    CHECK(fyodor_context_read(store,&bob,"workspace",&a.ref,&record)==FYODOR_STORE_DENIED);
    CHECK(fyodor_context_read(store,&alice,"elsewhere",&a.ref,&record)==FYODOR_STORE_DENIED);
    fyodor_resource_summary *items=NULL; size_t count=99;
    CHECK(fyodor_context_search(store,&alice,"workspace","avalon",10,&items,&count)==FYODOR_STORE_OK && count==0);
    fyodor_resource_summaries_free(items,count);
    CHECK(fyodor_context_permissions_set(store,&alice,"workspace",&a.ref,FYODOR_PERMISSION_SEARCH)==FYODOR_STORE_OK);
    CHECK(fyodor_context_read(store,&alice,"workspace",&a.ref,&record)==FYODOR_STORE_DENIED);
    CHECK(fyodor_context_search(store,&alice,"workspace","avalon",10,&items,&count)==FYODOR_STORE_OK && count==0);
    fyodor_resource_summaries_free(items,count);
    CHECK(fyodor_context_permissions_set(store,&alice,"workspace",&a.ref,5)==FYODOR_STORE_OK);
    CHECK(fyodor_context_search(store,&alice,"workspace","avalon",10,&items,&count)==FYODOR_STORE_OK && count==1);
    CHECK(fyodor_resource_equal(&items[0].ref,&a.ref)); fyodor_resource_summaries_free(items,count);
    CHECK(fyodor_context_permissions_set(store,&alice,"workspace",&b.ref,5)==FYODOR_STORE_OK);
    CHECK(fyodor_context_search(store,&alice,"workspace","AVALON",10,&items,&count)==FYODOR_STORE_OK && count==2);
    CHECK(fyodor_resource_equal(&items[0].ref,&a.ref)); fyodor_resource_summaries_free(items,count);
    CHECK(fyodor_context_search(store,&bob,"workspace","secret",10,&items,&count)==FYODOR_STORE_OK && count==0);
    fyodor_resource_summaries_free(items,count);
    fyodor_resource_ref refs[]={a.ref,b.ref};
    for(size_t budget=0;budget<30;++budget) {
        fyodor_context_bundle bundle={0};
        CHECK(fyodor_context_assemble(store,&alice,"workspace",refs,2,budget,&bundle)==FYODOR_STORE_OK);
        CHECK(bundle.length<=budget && strlen(bundle.text)==bundle.length && bundle.source_count==2);
        for(size_t i=0;i<2;++i) {
            CHECK(bundle.sources[i].revision==1 && bundle.sources[i].offset+bundle.sources[i].length<=bundle.length);
            CHECK(fyodor_resource_equal(&bundle.sources[i].ref,&refs[i]));
            const char *text=i==0?a.content:b.content;
            CHECK(bundle.sources[i].original_length==strlen(text));
            CHECK(memcmp(bundle.text+bundle.sources[i].offset,text,bundle.sources[i].length)==0);
            if(bundle.sources[i].length<strlen(text)) CHECK(((unsigned char)text[bundle.sources[i].length]&0xc0u)!=0x80u);
        }
        fyodor_context_bundle_free(&bundle);
    }
    fyodor_context_bundle bundle={0};
    CHECK(fyodor_context_assemble(store,&alice,"workspace",refs,2,100,&bundle)==FYODOR_STORE_OK);
    char expected[128]; snprintf(expected,sizeof(expected),"%s\n%s",a.content,b.content);
    CHECK(strcmp(bundle.text,expected)==0); fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_context_permissions_set(store,&alice,"workspace",&b.ref,0)==FYODOR_STORE_OK);
    /* Denied sources cannot be concealed after the budget is exhausted. */
    CHECK(fyodor_context_assemble(store,&alice,"workspace",refs,2,0,&bundle)==FYODOR_STORE_DENIED && bundle.text==NULL);
    CHECK(fyodor_context_assemble(store,&alice,"workspace",refs,2,100,&bundle)==FYODOR_STORE_DENIED && bundle.text==NULL);
    CHECK(fyodor_context_read(store,&alice,"workspace",&b.ref,&record)==FYODOR_STORE_DENIED);
    refs[1]=refs[0];
    CHECK(fyodor_context_assemble(store,&alice,"workspace",refs,2,100,&bundle)==FYODOR_STORE_INVALID);
    CHECK(fyodor_context_assemble(store,&alice,"workspace",NULL,65,100,&bundle)==FYODOR_STORE_INVALID);
    CHECK(fyodor_context_search(store,&alice,"workspace","",10,&items,&count)==FYODOR_STORE_INVALID);
    CHECK(fyodor_context_permissions_set(store,&alice,"workspace",&a.ref,64)==FYODOR_STORE_INVALID);
    fyodor_store_close(store);
    sqlite3 *v2=NULL;
    CHECK(sqlite3_open(path,&v2)==SQLITE_OK);
    CHECK(sqlite3_exec(v2,"DROP TABLE resource_search; DROP TABLE resource_search_keys; DROP TABLE context_receipts; PRAGMA user_version=2",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(sqlite3_close(v2)==SQLITE_OK);
    CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    unsigned mask=999;
    CHECK(fyodor_context_permissions_get(store,&alice,"workspace",&a.ref,&mask)==FYODOR_STORE_OK && mask==5);
    CHECK(fyodor_context_permissions_get(store,&alice,"workspace",&b.ref,&mask)==FYODOR_STORE_OK && mask==0);
    CHECK(fyodor_store_delete(store,"workspace",&a.ref,1)==FYODOR_STORE_OK);
    CHECK(fyodor_context_read(store,&alice,"workspace",&a.ref,&record)==FYODOR_STORE_DENIED);
    CHECK(fyodor_context_permissions_get(store,&alice,"workspace",&a.ref,&mask)==FYODOR_STORE_OK && mask==0);
    CHECK(fyodor_context_permissions_set(store,&alice,"workspace",&a.ref,5)==FYODOR_STORE_CONFLICT);
    fyodor_store_close(store);

    /* Recreate the exact v1 schema by removing only the v2 table and marker.
     * Reopening must migrate atomically, retain history and create no grants. */
    sqlite3 *db=NULL; CHECK(sqlite3_open(path,&db)==SQLITE_OK);
    CHECK(sqlite3_exec(db,"DROP TABLE resource_search; DROP TABLE resource_search_keys; DROP TABLE context_receipts; DROP TABLE context_permissions; PRAGMA user_version=1",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(sqlite3_close(db)==SQLITE_OK);
    CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    CHECK(fyodor_store_get(store,"workspace",&a.ref,1,&record)==FYODOR_STORE_OK);
    CHECK(strcmp(record.content,a.content)==0); fyodor_resource_record_free(&record);
    CHECK(fyodor_store_get(store,"workspace",&b.ref,0,&record)==FYODOR_STORE_OK);
    CHECK(strcmp(record.content,b.content)==0); fyodor_resource_record_free(&record);
    CHECK(fyodor_context_permissions_get(store,&alice,"workspace",&b.ref,&mask)==FYODOR_STORE_OK && mask==0);
    CHECK(fyodor_context_read(store,&alice,"workspace",&b.ref,&record)==FYODOR_STORE_DENIED);
    fyodor_store_close(store);
    return 0;
}
static int mutations(const char *path)
{
    fyodor_store *store=NULL;
    CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    fyodor_uuid principal,other;
    CHECK(fyodor_uuid_generate(&principal)==FYODOR_RESOURCE_OK);
    CHECK(fyodor_uuid_generate(&other)==FYODOR_RESOURCE_OK);
    for(unsigned mask=0;mask<=63;++mask) {
        fyodor_resource_input input=document("original title","initial");
        uint64_t revision=0,next=999;
        CHECK(fyodor_store_put(store,&input,0,&revision)==FYODOR_STORE_OK);
        CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&input.ref,mask)==FYODOR_STORE_OK);
        input.title="new title"; input.content="replacement"; input.metadata="{\"permissions\":63}";
        CHECK(fyodor_context_write(store,&other,&input,revision,&next)==FYODOR_STORE_DENIED&&next==999);
        CHECK(fyodor_context_write(store,&principal,&input,0,&next)==FYODOR_STORE_INVALID);
        fyodor_store_result result=fyodor_context_write(store,&principal,&input,revision,&next);
        CHECK(result==((mask&2)?FYODOR_STORE_OK:FYODOR_STORE_DENIED));
        if(mask&2) { CHECK(next==revision+1); revision=next; }
        else CHECK(next==999);
        next=999;
        result=fyodor_context_append(store,&principal,"workspace",&input.ref,"-suffix",revision,&next);
        CHECK(result==((mask&8)?FYODOR_STORE_OK:FYODOR_STORE_DENIED));
        if(mask&8) { CHECK(next==revision+1); revision=next; }
        else CHECK(next==999);
        fyodor_resource_record record={0};
        CHECK(fyodor_store_get(store,"workspace",&input.ref,0,&record)==FYODOR_STORE_OK);
        const char *body=(mask&2)?((mask&8)?"replacement-suffix":"replacement"):((mask&8)?"initial-suffix":"initial");
        CHECK(strcmp(record.content,body)==0&&strcmp(record.title,(mask&2)?"new title":"original title")==0);
        CHECK(strcmp(record.metadata,(mask&2)?"{\"permissions\":63}":"{}")==0);
        fyodor_resource_record_free(&record);
        result=fyodor_context_read(store,&principal,"workspace",&input.ref,&record);
        CHECK(result==((mask&1)?FYODOR_STORE_OK:FYODOR_STORE_DENIED));
        fyodor_resource_record_free(&record);
        if(mask&8) CHECK(fyodor_context_append(store,&principal,"workspace",&input.ref,"stale",revision+1,&next)==FYODOR_STORE_CONFLICT);
        CHECK(fyodor_context_delete(store,&principal,"elsewhere",&input.ref,revision)==FYODOR_STORE_DENIED);
        result=fyodor_context_delete(store,&principal,"workspace",&input.ref,revision);
        CHECK(result==((mask&16)?FYODOR_STORE_OK:FYODOR_STORE_DENIED));
        CHECK(fyodor_store_get(store,"workspace",&input.ref,1,&record)==FYODOR_STORE_OK&&strcmp(record.content,"initial")==0);
        fyodor_resource_record_free(&record);
        CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&input.ref,0)==FYODOR_STORE_OK);
        CHECK(fyodor_context_write(store,&principal,&input,revision,&next)==FYODOR_STORE_DENIED);
        CHECK(fyodor_context_append(store,&principal,"workspace",&input.ref,"revoked",revision,&next)==FYODOR_STORE_DENIED);
    }
    fyodor_resource_input input=document("atomic","original"); uint64_t revision=0,next=888;
    CHECK(fyodor_store_put(store,&input,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&input.ref,63)==FYODOR_STORE_OK);
    sqlite3 *db=NULL; CHECK(sqlite3_open(path,&db)==SQLITE_OK);
    CHECK(sqlite3_exec(db,"CREATE TRIGGER fail_scoped BEFORE INSERT ON revisions "
        "BEGIN SELECT RAISE(ABORT,'injected'); END",NULL,NULL,NULL)==SQLITE_OK);
    input.content="replacement";
    CHECK(fyodor_context_write(store,&principal,&input,1,&next)==FYODOR_STORE_CONFLICT&&next==888);
    CHECK(fyodor_context_append(store,&principal,"workspace",&input.ref,"append",1,&next)==FYODOR_STORE_CONFLICT&&next==888);
    CHECK(fyodor_context_delete(store,&principal,"workspace",&input.ref,1)==FYODOR_STORE_CONFLICT);
    fyodor_resource_record record={0};
    CHECK(fyodor_store_get(store,"workspace",&input.ref,0,&record)==FYODOR_STORE_OK);
    CHECK(record.revision==1&&strcmp(record.content,"original")==0);
    fyodor_resource_record_free(&record);
    CHECK(fyodor_store_get(store,"workspace",&input.ref,2,&record)==FYODOR_STORE_NOT_FOUND);
    CHECK(sqlite3_exec(db,"DROP TRIGGER fail_scoped",NULL,NULL,NULL)==SQLITE_OK);
    CHECK(sqlite3_close(db)==SQLITE_OK);
    fyodor_store_close(store); return 0;
}

static int layers(const char *path)
{
    fyodor_store *store=NULL; fyodor_uuid principal,other;
    CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    CHECK(fyodor_uuid_generate(&principal)==FYODOR_RESOURCE_OK);
    CHECK(fyodor_uuid_generate(&other)==FYODOR_RESOURCE_OK);
    fyodor_context_entry entries[5];
    for(int i=0;i<5;++i) {
        fyodor_resource_input input=document("layer",fyodor_context_layer_name((fyodor_context_layer)i));
        uint64_t revision=0;
        CHECK(fyodor_store_put(store,&input,0,&revision)==FYODOR_STORE_OK);
        CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&input.ref,5)==FYODOR_STORE_OK);
        entries[4-i]=(fyodor_context_entry){input.ref,(fyodor_context_layer)i};
    }
    fyodor_context_bundle bundle={0};
    CHECK(fyodor_context_assemble_layers(store,&principal,"workspace",entries,5,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"explicit\nworkspace\nsession\nretrieved\nglobal")==0);
    for(int i=0;i<5;++i) CHECK(bundle.sources[i].layer==(fyodor_context_layer)i&&fyodor_resource_equal(&bundle.sources[i].ref,&entries[4-i].ref));
    fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_context_assemble_layers(store,&principal,"workspace",entries,5,10,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"explicit\nw")==0&&bundle.source_count==5&&bundle.sources[4].length==0);
    fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_context_assemble_layers(store,&other,"workspace",entries,5,0,&bundle)==FYODOR_STORE_DENIED);
    CHECK(fyodor_context_assemble_layers(store,&principal,"other",entries,5,100,&bundle)==FYODOR_STORE_DENIED);
    fyodor_context_entry same_layer[]={entries[0],entries[4]};
    same_layer[0].layer=FYODOR_CONTEXT_SESSION; same_layer[1].layer=FYODOR_CONTEXT_SESSION;
    CHECK(fyodor_context_assemble_layers(store,&principal,"workspace",same_layer,2,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"global\nexplicit")==0&&bundle.sources[0].layer==FYODOR_CONTEXT_SESSION);
    fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&entries[1].ref,1)==FYODOR_STORE_OK);
    CHECK(fyodor_context_assemble_layers(store,&principal,"workspace",entries,5,0,&bundle)==FYODOR_STORE_DENIED&&bundle.text==NULL);
    CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&entries[1].ref,5)==FYODOR_STORE_OK);
    CHECK(fyodor_context_permissions_set(store,&principal,"workspace",&entries[0].ref,0)==FYODOR_STORE_OK);
    CHECK(fyodor_context_assemble_layers(store,&principal,"workspace",entries,5,1,&bundle)==FYODOR_STORE_DENIED&&bundle.text==NULL);
    entries[0].layer=(fyodor_context_layer)99;
    CHECK(fyodor_context_assemble_layers(store,&principal,"workspace",entries,5,100,&bundle)==FYODOR_STORE_INVALID);
    entries[0]=entries[1]; entries[0].layer=FYODOR_CONTEXT_GLOBAL;
    CHECK(fyodor_context_assemble_layers(store,&principal,"workspace",entries,5,100,&bundle)==FYODOR_STORE_INVALID);
    CHECK(fyodor_context_assemble_layers(store,&principal,"workspace",NULL,65,0,&bundle)==FYODOR_STORE_INVALID);
    fyodor_store_close(store); return 0;
}

static int writing_projects(const char *path)
{
    fyodor_store *store=NULL;CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    fyodor_uuid principal;CHECK(fyodor_uuid_generate(&principal)==FYODOR_RESOURCE_OK);
    fyodor_resource_input root=document("Root","root"),folder=document("Folder","folder"),target=document("Target","doc");
    root.ref.kind=folder.ref.kind=FYODOR_RESOURCE_PROJECT;
    root.name_space=folder.name_space=target.name_space="projects";
    uint64_t revision=0;
    CHECK(fyodor_store_put(store,&root,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_put(store,&folder,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_put(store,&target,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_move_writing(store,"projects",&folder.ref,1,&root.ref,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_move_writing(store,"projects",&target.ref,1,&folder.ref,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_context_permissions_set(store,&principal,"projects",&target.ref,1)==FYODOR_STORE_OK);
    CHECK(fyodor_context_permissions_set(store,&principal,"projects",&root.ref,1)==FYODOR_STORE_OK);
    fyodor_context_bundle bundle={0};
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,NULL,0,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc")==0&&bundle.source_count==1);fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_context_permissions_set(store,&principal,"projects",&folder.ref,1)==FYODOR_STORE_OK);
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,NULL,0,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc\nfolder\nroot")==0&&bundle.source_count==3);
    CHECK(bundle.sources[0].layer==FYODOR_CONTEXT_EXPLICIT&&bundle.sources[1].layer==FYODOR_CONTEXT_WORKSPACE);
    CHECK(bundle.sources[1].revision==2&&bundle.sources[2].revision==1);fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,NULL,0,3,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc")==0&&bundle.source_count==3&&bundle.sources[1].length==0&&bundle.sources[2].length==0);fyodor_context_bundle_free(&bundle);
    fyodor_context_entry extra={root.ref,FYODOR_CONTEXT_GLOBAL};
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,&extra,1,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc\nfolder\nroot")==0&&bundle.source_count==3&&bundle.sources[2].layer==FYODOR_CONTEXT_GLOBAL);fyodor_context_bundle_free(&bundle);
    extra.layer=FYODOR_CONTEXT_EXPLICIT;
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,&extra,1,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc\nroot\nfolder")==0&&bundle.sources[1].layer==FYODOR_CONTEXT_EXPLICIT);fyodor_context_bundle_free(&bundle);
    extra.layer=FYODOR_CONTEXT_RETRIEVED;
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,&extra,1,0,&bundle)==FYODOR_STORE_DENIED&&bundle.text==NULL);
    CHECK(fyodor_context_permissions_set(store,&principal,"projects",&root.ref,0)==FYODOR_STORE_OK);
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,NULL,0,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc\nfolder")==0&&bundle.source_count==2);fyodor_context_bundle_free(&bundle);
    extra.layer=FYODOR_CONTEXT_EXPLICIT;
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,&extra,1,0,&bundle)==FYODOR_STORE_DENIED);
    extra.ref=target.ref;
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,&extra,1,100,&bundle)==FYODOR_STORE_INVALID);
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,1,NULL,0,100,&bundle)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_writing_assemble(store,&principal,"elsewhere",&target.ref,2,NULL,0,100,&bundle)==FYODOR_STORE_DENIED);
    /* Caller selections reserve capacity; automatic context cannot evict them. */
    fyodor_context_entry many[63];
    for(size_t i=0;i<63;++i) {
        fyodor_resource_input item=document("Extra","x");item.name_space="projects";
        CHECK(fyodor_store_put(store,&item,0,&revision)==FYODOR_STORE_OK);
        CHECK(fyodor_context_permissions_set(store,&principal,"projects",&item.ref,1)==FYODOR_STORE_OK);
        many[i]=(fyodor_context_entry){item.ref,FYODOR_CONTEXT_EXPLICIT};
    }
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,many,63,0,&bundle)==FYODOR_STORE_OK);
    CHECK(bundle.source_count==64&&bundle.sources[63].layer==FYODOR_CONTEXT_EXPLICIT);fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,many,62,0,&bundle)==FYODOR_STORE_OK);
    CHECK(bundle.source_count==64&&fyodor_resource_equal(&bundle.sources[63].ref,&folder.ref));fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_store_move_writing(store,"projects",&target.ref,2,NULL,&revision)==FYODOR_STORE_OK&&revision==3);
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,2,NULL,0,100,&bundle)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_writing_assemble(store,&principal,"projects",&target.ref,3,NULL,0,100,&bundle)==FYODOR_STORE_OK&&bundle.source_count==1);fyodor_context_bundle_free(&bundle);
    fyodor_store_close(store);return 0;
}

static int writing_lore(const char *path)
{
    fyodor_store *store=NULL;CHECK(fyodor_store_open(path,&store)==FYODOR_STORE_OK);
    fyodor_uuid principal;CHECK(fyodor_uuid_generate(&principal)==FYODOR_RESOURCE_OK);
    fyodor_resource_input world=document("World","world"),lore=document("Lore","lore"),target=document("Target","doc"),project=document("Project","project");
    world.ref.kind=FYODOR_RESOURCE_WORLD;lore.ref.kind=FYODOR_RESOURCE_WORLD_LORE;lore.ref.parent_id=world.ref.id;project.ref.kind=FYODOR_RESOURCE_PROJECT;
    world.name_space=lore.name_space=target.name_space=project.name_space="lore";target.metadata="{\"large\":9007199254740993}";
    uint64_t revision=0;
    CHECK(fyodor_store_put(store,&world,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_put(store,&lore,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_put(store,&project,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_put(store,&target,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_set_lore(store,"lore",&target.ref,1,&lore.ref,1,&revision)==FYODOR_STORE_OK&&revision==2);
    fyodor_resource_record record={0};CHECK(fyodor_store_get(store,"lore",&target.ref,0,&record)==FYODOR_STORE_OK);
    CHECK(strstr(record.metadata,"9007199254740993")&&strstr(record.metadata,"writing_lore"));fyodor_resource_record_free(&record);
    CHECK(fyodor_store_get(store,"lore",&target.ref,1,&record)==FYODOR_STORE_OK&&strcmp(record.metadata,target.metadata)==0);fyodor_resource_record_free(&record);
    CHECK(fyodor_store_set_lore(store,"lore",&target.ref,1,NULL,0,&revision)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_set_lore(store,"lore",&target.ref,2,&lore.ref,33,&revision)==FYODOR_STORE_INVALID);
    fyodor_resource_ref duplicate[]={lore.ref,lore.ref};
    CHECK(fyodor_store_set_lore(store,"lore",&target.ref,2,duplicate,2,&revision)==FYODOR_STORE_INVALID);
    CHECK(fyodor_store_set_lore(store,"lore",&target.ref,2,&world.ref,1,&revision)==FYODOR_STORE_INVALID);
    CHECK(fyodor_store_delete(store,"lore",&lore.ref,1)==FYODOR_STORE_CONFLICT);
    target.metadata="{\"writing_lore\":null}";CHECK(fyodor_store_put(store,&target,2,&revision)==FYODOR_STORE_CONFLICT);
    target.metadata="{\"writing_lore\":[42]}";CHECK(fyodor_store_put(store,&target,2,&revision)==FYODOR_STORE_CONFLICT);
    fyodor_resource_summary *page=NULL;size_t count=0;
    CHECK(fyodor_store_list_lore(store,"lore",NULL,1,&page,&count)==FYODOR_STORE_OK&&count==1&&fyodor_resource_equal(&page[0].ref,&lore.ref));fyodor_resource_summaries_free(page,count);
    CHECK(fyodor_context_permissions_set(store,&principal,"lore",&target.ref,1)==FYODOR_STORE_OK);
    fyodor_context_bundle bundle={0};
    CHECK(fyodor_writing_assemble(store,&principal,"lore",&target.ref,2,NULL,0,100,&bundle)==FYODOR_STORE_OK&&bundle.source_count==1);fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_context_permissions_set(store,&principal,"lore",&lore.ref,1)==FYODOR_STORE_OK);
    CHECK(fyodor_writing_assemble(store,&principal,"lore",&target.ref,2,NULL,0,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc\nlore")==0&&bundle.source_count==2&&bundle.sources[1].layer==FYODOR_CONTEXT_WORKSPACE);fyodor_context_bundle_free(&bundle);
    fyodor_context_entry extra={lore.ref,FYODOR_CONTEXT_EXPLICIT};
    CHECK(fyodor_writing_assemble(store,&principal,"lore",&target.ref,2,&extra,1,0,&bundle)==FYODOR_STORE_OK&&bundle.source_count==2&&bundle.sources[1].layer==FYODOR_CONTEXT_EXPLICIT);fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_store_set_lore(store,"lore",&project.ref,1,&lore.ref,1,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_move_writing(store,"lore",&target.ref,2,&project.ref,&revision)==FYODOR_STORE_OK&&revision==3);
    CHECK(fyodor_context_permissions_set(store,&principal,"lore",&project.ref,1)==FYODOR_STORE_OK);
    CHECK(fyodor_writing_assemble(store,&principal,"lore",&target.ref,3,NULL,0,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc\nlore\nproject")==0&&bundle.source_count==3);fyodor_context_bundle_free(&bundle);
    sqlite3 *db=NULL;CHECK(sqlite3_open(path,&db)==SQLITE_OK);
    CHECK(sqlite3_exec(db,"CREATE TRIGGER fail_lore BEFORE INSERT ON revisions BEGIN SELECT RAISE(ABORT,'injected'); END",NULL,NULL,NULL)==SQLITE_OK);
    revision=999;CHECK(fyodor_store_set_lore(store,"lore",&target.ref,3,NULL,0,&revision)==FYODOR_STORE_CONFLICT&&revision==999);
    CHECK(sqlite3_exec(db,"DROP TRIGGER fail_lore",NULL,NULL,NULL)==SQLITE_OK);CHECK(sqlite3_close(db)==SQLITE_OK);
    CHECK(fyodor_store_set_lore(store,"lore",&target.ref,3,NULL,0,&revision)==FYODOR_STORE_OK&&revision==4);
    CHECK(fyodor_writing_assemble(store,&principal,"lore",&target.ref,4,NULL,0,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc\nproject\nlore")==0);fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_context_permissions_set(store,&principal,"lore",&lore.ref,0)==FYODOR_STORE_OK);
    CHECK(fyodor_writing_assemble(store,&principal,"lore",&target.ref,4,NULL,0,100,&bundle)==FYODOR_STORE_OK);
    CHECK(strcmp(bundle.text,"doc\nproject")==0);fyodor_context_bundle_free(&bundle);
    CHECK(fyodor_store_delete(store,"lore",&lore.ref,1)==FYODOR_STORE_CONFLICT);
    CHECK(fyodor_store_set_lore(store,"lore",&project.ref,2,NULL,0,&revision)==FYODOR_STORE_OK);
    CHECK(fyodor_store_delete(store,"lore",&lore.ref,1)==FYODOR_STORE_OK);
    CHECK(fyodor_store_set_lore(store,"lore",&target.ref,4,&lore.ref,1,&revision)==FYODOR_STORE_CONFLICT);
    fyodor_store_close(store);return 0;
}

int main(int argc,char **argv)
{
    if(argc!=2) return 2;
    (void)remove(argv[1]);
    if(exercise(argv[1])||mutations(argv[1])||layers(argv[1])||writing_projects(argv[1])||writing_lore(argv[1])) return 1;
    CHECK(remove(argv[1])==0);
    puts("context: persisted permissions, isolation, retrieval, bounded assembly and v1 migration passed");
    return 0;
}

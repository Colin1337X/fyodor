// Human-selected local authoring sources use administrative resource reads.
// This is not the permission-scoped Context ingestion service: do not infer
// model access from a visible source, create grants, or invent native chat IDs.
import {datasetLimit, inspectDataset, newDataset} from './dataset-data.js';
import {exploreKind} from './explore-data.js';

const uuid = '[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}';
const writing = new RegExp(`^fyodor://writing/(documents|notes|characters|projects)/${uuid}$`);
const encoder = new TextEncoder();
export const sourceLimit = 32;
export const sourceKind = uri => writing.test(uri) ? 'writing' : exploreKind(uri) ? 'explore' : null;

export async function sourcePage(api, namespace, kind, after = '') {
  if (!['writing','explore'].includes(kind)) throw Error('Choose Writing or Explore sources.');
  const resources = [], seen = new Set(); let cursor = after;
  for (let requests=0; requests<25; requests++) {
    seen.add(cursor);
    const page = await api.resources(namespace,cursor,kind==='writing'?'writing':'all');
    resources.push(...page.resources.filter(item=>sourceKind(item.uri)===kind));
    cursor = page.next;
    if (cursor && seen.has(cursor)) throw Error('Repeated source cursor. Browse again.');
    if (!cursor || resources.length>=20) break;
  }
  return {resources,next:cursor};
}

export function resourceSource(item, namespace) {
  if (!sourceKind(item.uri) || typeof item.revision!=='string' || !/^[1-9][0-9]*$/.test(item.revision) || !/^[A-Za-z0-9_-]{1,64}$/.test(namespace))
    throw Error('Select a saved Writing or Explore revision.');
  return {kind:sourceKind(item.uri),uri:item.uri,revision:item.revision,namespace,title:item.title||'Untitled'};
}

export function chatSource(chat) {
  if (!chat || typeof chat.id!=='string' || chat.id.length>160 || !Array.isArray(chat.messages) || chat.messages.length>2000)
    throw Error('Choose a supported local chat.');
  let bytes=0;
  const messages = chat.messages.map(message=>{
    if (!message || !['user','assistant','tool'].includes(message.role) || typeof message.text!=='string') throw Error('Unsupported chat message.');
    bytes+=encoder.encode(message.text).length;
    if (bytes>datasetLimit) throw Error('Selected chat text exceeds the Studio’s 1 MiB limit.');
    return {role:message.role,text:message.text};
  });
  // Freeze text at selection, including partial assistant outputs as ordinary
  // text. Selection never labels it verified or silently adds a system prompt.
  return {kind:'local-chat',local_id:chat.id,title:chat.title||'Untitled chat',created_ms:chat.created,messages};
}

async function digest(text) {
  const hash = await crypto.subtle.digest('SHA-256',encoder.encode(text));
  return Array.from(new Uint8Array(hash),byte=>byte.toString(16).padStart(2,'0')).join('');
}
function field(text, policy) {
  if (text.includes('\0')) throw Error('Source text contains null characters.');
  if (!/[\t\r\n]/.test(text)) return text;
  if (policy==='spaces') return text.replace(/\r\n|[\t\r\n]/g,' ');
  throw Error('SFT fields contain tabs or line breaks. Choose their explicit replacement policy or use a corpus format.');
}

export async function datasetFromSources(api, sources, {mode='cpt',delimiterPolicy='reject',prompt=''} = {}) {
  if (!['pretrain','cpt','sft'].includes(mode)) throw Error('Source ingestion supports corpus or SFT. DPO needs explicit chosen/rejected records.');
  if (!['reject','spaces'].includes(delimiterPolicy) || typeof prompt!=='string') throw Error('Choose a supported field mapping.');
  if (!sources.length || sources.length>sourceLimit) throw Error('Select 1–32 sources.');
  const seen = new Set(), pieces = [], provenanceSources = [];
  let bytes = 0, records = 0;
  const append = text => {
    const start = bytes + (pieces.length ? (mode==='sft'?1:2) : 0);
    const length = encoder.encode(text).length;
    if (start+length>datasetLimit) throw Error('Selected output exceeds the Studio’s 1 MiB limit. Select fewer sources.');
    pieces.push(text); bytes=start+length;
    return {byte_start:start,byte_length:length};
  };
  for (const selected of sources) {
    const key = selected.kind==='local-chat' ? 'chat:'+selected.local_id : selected.namespace+':'+selected.uri;
    if (seen.has(key)) throw Error('A source was selected more than once.');
    seen.add(key);
    let texts, origin;
    if (selected.kind==='local-chat') {
      const chat = chatSource({id:selected.local_id,title:selected.title,created:selected.created_ms,messages:selected.messages});
      const indices=[]; texts=[];
      if (mode==='sft') {
        for (let index=0; index+1<chat.messages.length; index++) {
          const user=chat.messages[index], assistant=chat.messages[index+1];
          if (user.role==='user' && assistant.role==='assistant') {
            texts.push(field(user.text,delimiterPolicy)+'\t'+field(assistant.text,delimiterPolicy));
            indices.push([index,index+1]);
          }
        }
        if (!texts.length) throw Error(`“${chat.title}” has no adjacent user/assistant pairs.`);
      } else {
        texts = [chat.messages.map(message=>message.role+':\n'+message.text).join('\n\n')];
        if (!chat.messages.length) throw Error(`“${chat.title}” has no messages.`);
      }
      // IDs are window-local and may change after restore. The hash identifies
      // exact role/text snapshot bytes; it is not a native revision or receipt.
      origin = {kind:'local-chat',local_id:chat.local_id,created_ms:chat.created_ms,
        snapshot_sha256:await digest(JSON.stringify(chat.messages)),message_count:chat.messages.length,
        ...(mode==='sft'?{message_pairs:indices}:{})};
    } else {
      const ref = resourceSource(selected,selected.namespace);
      // Request the selected revision, never the moving live head. Each read is
      // independent; this is a reproducible revision set, not one DB snapshot.
      const loaded = await api.resource(ref.namespace,ref.uri,ref.revision);
      if (loaded.deleted || loaded.revision!==ref.revision || loaded.resource?.uri!==ref.uri || typeof loaded.resource.content!=='string')
        throw Error('A selected revision is unavailable. Browse and select it again.');
      const text = loaded.resource.content;
      if (!text.length) throw Error(`“${ref.title}” has no saved text.`);
      texts = [mode==='sft'?field(prompt,delimiterPolicy)+'\t'+field(text,delimiterPolicy):text];
      origin = {kind:ref.kind,namespace:ref.namespace,uri:ref.uri,revision:ref.revision,content_sha256:await digest(text)};
    }
    const first=records, spans=texts.map(text=>{++records;return append(text);});
    provenanceSources.push({...origin,first_unit:first,unit_count:texts.length,byte_start:spans[0].byte_start,
      byte_length:bytes-spans[0].byte_start});
  }
  const content=pieces.join(mode==='sft'?'\n':'\n\n');
  const report=inspectDataset(content,mode);
  if (report.issues.length) throw Error(report.issues[0].message);
  const provenance={dataset_studio:{version:1,operation:'source-ingestion',output_sha256:await digest(content),mapping:mode==='sft'?'resource-prompt-content;adjacent-user-assistant':'resource-content;role-labelled-chat',
    delimiter_policy:mode==='sft'?delimiterPolicy:'preserve',...(mode==='sft'?{resource_prompt:prompt}:{}),sources:provenanceSources}};
  if (encoder.encode(JSON.stringify(provenance)).length>65536) throw Error('Source provenance exceeds 64 KiB. Select fewer chat messages/sources.');
  return newDataset(mode,sources.length===1?sources[0].title+' · dataset':'Selected sources · dataset',content,provenance);
}

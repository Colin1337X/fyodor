// Receipt offsets are UTF-8 byte ranges in the executed prompt, not JavaScript
// character indices. Keep native integer revisions exact and reject unsupported
// shapes rather than slicing user instructions or emitting replacement text.
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8',{fatal:true,ignoreBOM:true});
const limit = 1024 * 1024;
const layers = ['explicit','workspace','session','retrieved','global'];
export const isContextIdentity = value => typeof value==='string' && /^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(value) && value!=='00000000-0000-0000-0000-000000000000';
export const contextSourceKey = source => `context:${source.namespace}:${source.principal}:${source.receipt_id}:${source.index}`;

export function inspectContextReceipt(receipt) {
  if (!receipt || typeof receipt.prompt!=='string' || receipt.prompt.includes('\0') ||
      typeof receipt.sources_json!=='string' || typeof receipt.metadata_json!=='string' ||
      encoder.encode(receipt.sources_json).length>65536 || encoder.encode(receipt.metadata_json).length>65536)
    throw Error('This receipt has an unsupported source record.');
  const prompt = encoder.encode(receipt.prompt);
  if (prompt.length>2*limit+2) throw Error('This receipt exceeds the supported prompt size.');
  const metadata = JSON.parse(receipt.metadata_json);
  const contextBytes = metadata?.context_bytes;
  if (metadata?.schema!==1 || !Number.isSafeInteger(contextBytes) || contextBytes<0 || contextBytes>limit || contextBytes>prompt.length)
    throw Error('This receipt has an invalid context boundary.');
  try { decoder.decode(prompt.subarray(0,contextBytes)); }
  catch { throw Error('The receipt context boundary splits a UTF-8 character.'); }
  const rows = JSON.parse(receipt.sources_json,(key,value,context)=>{
    if (key!=='revision') return value;
    // JSON.parse's source context is available in current engines. Older
    // engines may use the safe-integer fallback, never a rounded large revision.
    let revision;
    if (typeof value==='string') revision=value;
    else if (typeof context?.source==='string') revision=context.source;
    else if (Number.isSafeInteger(value)) revision=String(value);
    else throw Error('This browser cannot read the receipt revision exactly.');
    if (!/^[1-9][0-9]{0,18}$/.test(revision) || BigInt(revision)>9223372036854775807n)
      throw Error('This receipt has an invalid source revision.');
    return revision;
  });
  if (!Array.isArray(rows) || rows.length>64) throw Error('This receipt has too many context sources.');
  const seen=new Set();let end=0;
  const sources=rows.map((row,index)=>{
    if (!row || typeof row.uri!=='string' || row.uri.length>320 || !/^fyodor:\/\/[a-z0-9/_-]+$/.test(row.uri) ||
        typeof row.revision!=='string' || !layers.includes(row.layer) ||
        ![row.offset,row.length,row.original_length].every(Number.isSafeInteger) ||
        row.offset<end || row.offset<0 || row.length<0 || row.original_length<row.length || row.original_length>limit ||
        row.offset+row.length>contextBytes || seen.has(row.uri)) throw Error('This receipt has an invalid source range.');
    seen.add(row.uri);end=row.offset+row.length;
    let content;
    try { content=decoder.decode(prompt.subarray(row.offset,end)); }
    catch { throw Error('A receipt source range splits a UTF-8 character.'); }
    const status=row.length===0?(row.original_length===0?'empty':'omitted'):row.length<row.original_length?'truncated':'included';
    return {...row,index,content,status};
  });
  return {sources,contextBytes};
}

export function contextSource(row, namespace, principal, receiptId) {
  if (!/^[A-Za-z0-9_-]{1,64}$/.test(namespace) || !isContextIdentity(principal) || !isContextIdentity(receiptId) ||
      !Number.isInteger(row.index) || row.index<0 || row.index>=64 || !row.length || typeof row.revision!=='string')
    throw Error('Choose an included source from a readable Context receipt.');
  return {kind:'context-receipt',namespace,principal,receipt_id:receiptId,index:row.index,
    uri:row.uri,revision:row.revision,title:'Context · '+row.uri};
}

export async function readContextSource(api, selected, cache = new Map()) {
  const key=selected.namespace+':'+selected.principal+':'+selected.receipt_id;
  if (!cache.has(key)) {
    // Always authorize anew for this preparation. The preview is not a grant,
    // and no administrative resource get is used after a separate grant query.
    const receipt=await api.contextReceipt(selected.namespace,selected.principal,selected.receipt_id);
    cache.set(key,inspectContextReceipt(receipt));
  }
  const row=cache.get(key).sources[selected.index];
  if (!row || row.uri!==selected.uri || row.revision!==selected.revision || !row.length)
    throw Error('The selected receipt source is unavailable. Read the receipt again.');
  return row;
}

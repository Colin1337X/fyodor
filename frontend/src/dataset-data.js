// Mirror the native CLI's structural grammar, not its model tokenizer. See
// backend/core/train_main.c: dataset_parse/encode_record. Token/context fit is
// checked by the trainer; character counts must never masquerade as tokens.
export const datasetModes = {pretrain:'Pretraining', cpt:'Continued pretraining', sft:'Supervised fine-tuning', dpo:'Preference / DPO'};
export const datasetLimit = 1024 * 1024;
const encoder = new TextEncoder();
const datasetPattern = /^fyodor:\/\/datasets\/[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/;
export const isDataset = uri => datasetPattern.test(uri);
export const savedDatasetMode = resource => resource.metadata?.dataset_studio?.version === 1 && Object.hasOwn(datasetModes, resource.metadata.dataset_studio.mode) ? resource.metadata.dataset_studio.mode : null;

export function inspectDataset(content, mode) {
  if (!Object.hasOwn(datasetModes, mode)) throw new Error('Choose a supported training format.');
  const issues = [], rows = [];
  const bytes = encoder.encode(content).length;
  if (!bytes) issues.push({line:0,message:'Add training text or records.'});
  if (bytes > datasetLimit) issues.push({line:0,message:'Studio datasets must be 1 MiB or smaller.'});
  if (content.includes('\0')) issues.push({line:0,message:'Null characters are not accepted by the trainer.'});
  const structured = mode === 'sft' || mode === 'dpo';
  const lines = content.split('\n');
  if (structured && lines.length > 100000) issues.push({line:0,message:'The trainer accepts at most 100000 lines, counting the empty line after a final newline.'});
  lines.forEach((source, index) => {
    const raw = source.endsWith('\r') ? source.slice(0,-1) : source;
    if (!raw.length) return;
    const fields = structured ? raw.split('\t') : [raw];
    const errors = [];
    if (structured) {
      if (fields.length !== (mode === 'sft' ? 2 : 3)) errors.push(mode === 'sft' ? 'Use prompt<TAB>completion.' : 'Use prompt<TAB>chosen<TAB>rejected.');
      else if (fields.slice(1).some(field => !field.length)) errors.push('Completion fields must be nonempty.');
    }
    errors.forEach(message => issues.push({line:index+1,message}));
    rows.push({line:index+1,raw,fields,valid:!errors.length});
  });
  if (structured && !rows.length && bytes) issues.push({line:0,message:'Add at least one nonempty structured record.'});
  const unique = new Set(rows.map(row => row.raw));
  return {rows,issues,bytes,characters:[...content].length,duplicates:rows.length-unique.size,structured};
}

export function decodeDataset(bytes, name) {
  if (!/\.(txt|tsv)$/i.test(name)) throw new Error('Choose a UTF-8 .txt or .tsv file.');
  if (bytes.byteLength > datasetLimit) throw new Error('Studio imports must be 1 MiB or smaller.');
  let text;
  try { text = new TextDecoder('utf-8',{fatal:true}).decode(bytes); }
  catch { throw new Error('Save this dataset as UTF-8 text and try again.'); }
  if (text.includes('\0')) throw new Error('The file contains null characters.');
  return text;
}

export function seededShuffle(rows, seed) {
  if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) throw new Error('Seed must be an integer from 0 to 4294967295.');
  let value = seed >>> 0;
  const result = [...rows];
  const random = () => {
    value = (value + 0x6d2b79f5) >>> 0;
    let n = Math.imul(value ^ value >>> 15, value | 1);
    n ^= n + Math.imul(n ^ n >>> 7, n | 61);
    return ((n ^ n >>> 14) >>> 0) / 4294967296;
  };
  for (let i=result.length-1;i>0;i--) {
    const j = Math.floor(random()*(i+1)); [result[i],result[j]] = [result[j],result[i]];
  }
  return result;
}

export function transformDataset(content, mode, operation, options = {}) {
  const inspected = inspectDataset(content,mode);
  if (inspected.issues.length) throw new Error('Fix format issues before transforming this dataset.');
  let rows = inspected.rows;
  if (operation === 'filter') rows = rows.filter(row => row.raw.toLowerCase().includes(String(options.query || '').toLowerCase()));
  else if (operation === 'deduplicate') {
    const seen = new Set(); rows = rows.filter(row => {if(seen.has(row.raw))return false;seen.add(row.raw);return true;});
  } else if (operation === 'shuffle' || operation === 'sample' || operation === 'split') rows = seededShuffle(rows, options.seed);
  else throw new Error('Unknown dataset operation.');
  if (operation === 'sample') {
    if (!Number.isInteger(options.count) || options.count < 1 || options.count > rows.length) throw new Error('Sample size must fit the available records.');
    rows = rows.slice(0,options.count);
  }
  if (operation === 'split') {
    if (rows.length < 2 || !Number.isInteger(options.validationPercent) || options.validationPercent < 1 || options.validationPercent > 99)
      throw new Error('Splits need at least two records and a validation share from 1 to 99%.');
    const count = Math.max(1,Math.min(rows.length-1,Math.round(rows.length*options.validationPercent/100)));
    return {training:rows.slice(count).map(row=>row.raw).join('\n'),validation:rows.slice(0,count).map(row=>row.raw).join('\n')};
  }
  if (!rows.length) throw new Error('This operation would create an empty dataset.');
  return {content:rows.map(row=>row.raw).join('\n')};
}

export function newDataset(mode, title = 'Untitled dataset', content = '', provenance = {}) {
  if (!Object.hasOwn(datasetModes,mode)) throw new Error('Unsupported training format.');
  return {schema:1,uri:'fyodor://datasets/'+crypto.randomUUID(),title,content,metadata:{dataset_studio:{version:1,mode}},provenance};
}

export async function datasetPage(api, namespace, after = '') {
  const resources = [], seen = new Set(); let cursor = after;
  for(let count=0;count<25;count++){
    seen.add(cursor);const page=await api.resources(namespace,cursor,'all');
    resources.push(...page.resources.filter(item=>isDataset(item.uri)));cursor=page.next;
    if(cursor&&seen.has(cursor))throw new Error('Repeated library cursor. Refresh and try again.');
    if(!cursor||resources.length>=20)break;
  }
  return {resources,next:cursor};
}

import { getBackendConnection } from "./platform.js";
import {readCompletionStream} from './stream.js';

// Local calls use the launcher's in-memory token. Tauri never proxies inference.
async function request(path, options = {}) {
  const connection = await getBackendConnection();
  const headers = new Headers(options.headers);
  if (connection.token) headers.set("Authorization", `Bearer ${connection.token}`);
  if (options.body) headers.set("Content-Type", "application/json");
  const response = await fetch(`${connection.baseUrl}${path}`, { ...options, headers });
  if(options.rawResponse && response.ok)return response.text();
  const data = await response.json().catch(() => null);
  if (!response.ok || data?.ok === false) throw new Error(data?.error?.message || `Backend returned ${response.status}`);
  if (!data) throw new Error("The backend returned an empty response.");
  return data;
}
const post = (path, body) => request(path, { method: "POST", body: JSON.stringify(body) });
async function external(provider, path, body, options={}) {
  const base = new URL(provider.endpoint);
  if (!["http:", "https:"].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw new Error("Use an HTTP(S) base URL without credentials, query or fragment.");
  const headers = new Headers({ "Content-Type": "application/json" });
  if (provider.apiKey) headers.set("Authorization", `Bearer ${provider.apiKey}`);
  const response = await fetch(base.href.replace(/\/+$/, "") + path, {
    method: body ? "POST" : "GET", headers, signal:options.signal, ...(body ? { body: JSON.stringify(body) } : {}),
  });
  if (response.ok && options.onUpdate && response.body && response.headers.get('content-type')?.includes('text/event-stream'))
    return readCompletionStream(response, options.onUpdate);
  const data = await response.json().catch(() => null);
  if (!response.ok || !data) throw new Error(data?.error?.message || `Endpoint returned ${response.status}`);
  if(options.onUpdate){const text=data.choices?.[0]?.message?.content;if(typeof text!=='string')throw new Error('Endpoint returned no assistant text.');options.onUpdate(text);return{text,tokens:data.usage?.completion_tokens||0};}
  return data;
}
export const backend = {
  contextSearch: (namespace,principal,query) => post('/api/v1/context/search',{namespace,principal,query}),
  contextReceipts: (namespace,principal,after='') => post('/api/v1/context/receipts',{namespace,principal,after}),
  contextPermissions: (namespace,principal,uri,permissions) => post('/api/v1/context/permissions',{namespace,principal,uri,...(permissions===undefined?{}:{permissions})}),
  writingGenerate: payload => post('/api/v1/context/writing',payload),
  contextGenerate: payload => post('/api/v1/context/generate',payload),
  contextReceipt: (namespace,principal,receipt_id) => post('/api/v1/context/receipt',{namespace,principal,receipt_id}),
  resources: (namespace,after='',scope='all',folder=null) => post('/api/v1/resources/list',{namespace,after,scope,...(folder===null?{}:{folder})}),
  setWritingLore: (namespace,uri,expected_revision,resources) => post('/api/v1/resources/lore',{namespace,uri,expected_revision,resources}),
  moveWriting: (namespace,uri,expected_revision,folder) => post('/api/v1/resources/move',{namespace,uri,expected_revision,folder}),
  resourceHistory: (namespace,uri,before='0') => post('/api/v1/resources/history',{namespace,uri,before}),
  resource: (namespace,uri,revision='0') => post('/api/v1/resources/get',{namespace,uri,revision}),
  saveResource: (namespace,resource,expected_revision) => post('/api/v1/resources/put',{namespace,resource,expected_revision}),
  updateResource: (namespace,resource,expected_revision,accepted=null) => post('/api/v1/resources/update',{namespace,uri:resource.uri,title:resource.title,content:resource.content,expected_revision,...(accepted?{receipt_id:accepted.receipt_id,principal:accepted.principal}:{})}),
  importResource: (namespace,text) => {JSON.parse(text);return request('/api/v1/resources/put',{method:'POST',body:`{"namespace":${JSON.stringify(namespace)},"expected_revision":"0","resource":${text}}`});},
  exportResource: (namespace,uri) => request('/api/v1/resources/export',{method:'POST',body:JSON.stringify({namespace,uri}),rawResponse:true}),
  deleteResource: (namespace,uri,expected_revision) => post('/api/v1/resources/delete',{namespace,uri,expected_revision}),
  health: () => request("/api/v1/health"), models: () => request("/api/v1/models"), runtime: () => request("/api/v1/runtime"),
  loadModel: path => post("/api/v1/model/load", { path }),
  unloadModel: model_id => post("/api/v1/model/unload", { model_id }),
  setCompute: (model_id, compute) => post("/api/v1/model/compute", { model_id, compute }),
  generate: payload => post("/api/v1/generate", payload),
  modelInfo: model_id => post('/api/v1/model/info',{model_id}),
  tokenize: (model_id,text) => post('/api/v1/tokenize',{model_id,text}),
  openaiModels: provider => external(provider, "/models"),
  openaiChat: (provider, payload, options) => external(provider, "/chat/completions", payload, options),
};

import { getBackendConnection } from "./platform.js";

// Local calls use the launcher's in-memory token. Tauri never proxies inference.
async function request(path, options = {}) {
  const connection = await getBackendConnection();
  const headers = new Headers(options.headers);
  if (connection.token) headers.set("Authorization", `Bearer ${connection.token}`);
  if (options.body) headers.set("Content-Type", "application/json");
  const response = await fetch(`${connection.baseUrl}${path}`, { ...options, headers });
  const data = await response.json().catch(() => null);
  if (!response.ok || data?.ok === false) throw new Error(data?.error?.message || `Backend returned ${response.status}`);
  if (!data) throw new Error("The backend returned an empty response.");
  return data;
}
const post = (path, body) => request(path, { method: "POST", body: JSON.stringify(body) });
async function external(provider, path, body) {
  const base = new URL(provider.endpoint);
  if (!["http:", "https:"].includes(base.protocol) || base.username || base.password || base.search || base.hash)
    throw new Error("Use an HTTP(S) base URL without credentials, query or fragment.");
  const headers = new Headers({ "Content-Type": "application/json" });
  if (provider.apiKey) headers.set("Authorization", `Bearer ${provider.apiKey}`);
  const response = await fetch(base.href.replace(/\/+$/, "") + path, {
    method: body ? "POST" : "GET", headers, ...(body ? { body: JSON.stringify(body) } : {}),
  });
  const data = await response.json().catch(() => null);
  if (!response.ok || !data) throw new Error(data?.error?.message || `Endpoint returned ${response.status}`);
  return data;
}
export const backend = {
  health: () => request("/api/v1/health"), models: () => request("/api/v1/models"), runtime: () => request("/api/v1/runtime"),
  loadModel: path => post("/api/v1/model/load", { path }),
  unloadModel: model_id => post("/api/v1/model/unload", { model_id }),
  setCompute: (model_id, compute) => post("/api/v1/model/compute", { model_id, compute }),
  generate: payload => post("/api/v1/generate", payload),
  openaiModels: provider => external(provider, "/models"),
  openaiChat: (provider, payload) => external(provider, "/chat/completions", payload),
};

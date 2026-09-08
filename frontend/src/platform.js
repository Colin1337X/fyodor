let connection;
export async function getBackendConnection(){if(connection)return connection;if(window.__TAURI_INTERNALS__){const{invoke}=await import("@tauri-apps/api/core");connection=await invoke("backend_connection")}else connection={baseUrl:(import.meta.env.VITE_FYODOR_URL||"http://127.0.0.1:9473").replace(/\/+$/,""),token:import.meta.env.VITE_FYODOR_TOKEN||""};return connection}
async function dialog(kind,options){if(!window.__TAURI_INTERNALS__)return null;const api=await import("@tauri-apps/plugin-dialog");return api[kind](options)}
export const chooseModel=()=>dialog("open",{multiple:false,filters:[{name:"AI models",extensions:["gguf","onnx","safetensors"]}]});
export const chooseDataset=()=>dialog("open",{multiple:false,filters:[{name:"Training data",extensions:["txt","tsv"]}]});
export const chooseCheckpoint=(output=false)=>dialog(output?"save":"open",{filters:[{name:"Fyodor checkpoint",extensions:["ckpt"]}]});
export const chooseOutput=()=>dialog("save",{filters:[{name:"GGUF model",extensions:["gguf"]}]});
export async function invokeDesktop(command,args={}){if(!window.__TAURI_INTERNALS__)throw new Error("This feature runs in the desktop app");const{invoke}=await import("@tauri-apps/api/core");return invoke(command,args)}

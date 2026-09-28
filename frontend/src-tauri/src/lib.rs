use serde::{Deserialize, Serialize};
use std::{
    collections::VecDeque,
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
    path::{Path, PathBuf},
    process::{Child, Command, Stdio},
    sync::{atomic::{AtomicBool, Ordering}, Arc, Mutex},
    time::Duration,
};
use tauri::{Manager, RunEvent};

type Logs = Arc<Mutex<VecDeque<String>>>;

// A GUI-owned sidecar still uses stdin/stdout pipes, but never creates a second
// console window on Windows. Running fyodor-backend directly keeps normal CLI
// console behavior, including --verbose diagnostics on stderr.
fn background_command(executable: &Path) -> Command {
    let mut command = Command::new(executable);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    command
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct Connection {
    base_url: String,
    token: String,
}

struct Backend {
    connection: Connection,
    child: Mutex<Option<Child>>,
}

struct Trainer {
    executable: PathBuf,
    child: Mutex<Option<TrainingProcess>>,
    last_exit: Mutex<Option<i32>>,
    exit_waiter: AtomicBool,
    logs: Logs,
}

struct TrainingProcess {
    child: Child,
    stopping: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrainArgs {
    mode: String,
    data: String,
    output: String,
    base: Option<String>,
    checkpoint: Option<String>,
    resume: Option<String>,
    steps: u32,
    #[serde(default = "default_accumulation")]
    accumulate: u32,
    #[serde(default)]
    threads: u32,
    #[serde(default)]
    eval_data: Option<String>,
    #[serde(default = "default_eval_interval")]
    eval_every: u32,
    #[serde(default)]
    eval_records: u32,
    learning_rate: f32,
    rank: u32,
    beta: f32,
    context: u32,
    dimension: u32,
    feed_forward: u32,
    layers: u32,
    heads: u32,
    kv_heads: u32,
    seed: u32,
    memory_mib: u32,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct TrainStatus {
    running: bool,
    stopping: bool,
    exit_code: Option<i32>,
}

fn default_accumulation() -> u32 {
    1
}

fn default_eval_interval() -> u32 { 10 }

#[cfg(test)]
mod training_tests {
    use super::*;

    #[test]
    fn trainer_safe_stop_and_failed_save() {
        let Some(executable) = std::env::var_os("FYODOR_TEST_TRAINER") else {
            eprintln!("Native process integration requires FYODOR_TEST_TRAINER");
            return;
        };
        let root = std::env::temp_dir().join(format!("fyodor-stop-test-\u{d55c}\u{ae00}-\u{1f43e}-{}-{}", std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        std::fs::create_dir(&root).unwrap();
        let data = root.join("corpus.txt");
        std::fs::write(&data, "abcabcabcabc").unwrap();
        for fail in [false, true] {
            let output = root.join(if fail { "existing.gguf" } else { "new.gguf" });
            let checkpoint = root.join(if fail { "failed-save.ckpt" } else { "saved.ckpt" });
            if fail { std::fs::write(&output, b"keep me").unwrap(); }
            let child = background_command(Path::new(&executable))
                .args(["--control-stdin", "1", "--steps", "100000", "--dimension", "16",
                    "--ff", "32", "--layers", "1", "--heads", "2", "--kv-heads", "1",
                    "--context", "16", "--accumulate", "3", "--data"])
                .arg(&data).arg("--output").arg(&output).arg("--checkpoint").arg(&checkpoint)
                .stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
            let trainer = Trainer {
                executable: PathBuf::from(&executable), child: Mutex::new(Some(TrainingProcess { child, stopping: false })),
                last_exit: Mutex::new(None), exit_waiter: AtomicBool::new(false),
                logs: Arc::new(Mutex::new(VecDeque::new())),
            };
            assert!(request_training_stop(&trainer).unwrap());
            request_training_stop(&trainer).unwrap(); // Duplicate requests never kill the child.
            let deadline = std::time::Instant::now() + Duration::from_secs(20);
            let status = loop {
                let status = collect_training_status(&trainer);
                if !status.running { break status; }
                assert!(status.stopping);
                if std::time::Instant::now() >= deadline {
                    if let Some(process) = trainer.child.lock().unwrap().as_mut() {
                        let _ = process.child.kill(); let _ = process.child.wait();
                    }
                    panic!("safe stop timed out");
                }
                std::thread::sleep(Duration::from_millis(10));
            };
            assert_eq!(status.exit_code, Some(if fail { 1 } else { 0 }));
            assert!(!status.stopping);
            assert!(!request_training_stop(&trainer).unwrap());
            assert!(std::fs::metadata(&checkpoint).unwrap().len() > 16);
            if fail { assert_eq!(std::fs::read(&output).unwrap(), b"keep me"); }
            else { assert!(std::fs::metadata(&output).unwrap().len() > 100); }
        }
        std::fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn accumulation_deserialization_and_cli_boundary() {
        // Existing saved/UI arguments omit accumulation. Keep their meaning.
        let mut input = serde_json::json!({
            "mode":"pretrain", "data":"corpus.txt", "output":"model.gguf",
            "steps":10, "learningRate":0.001, "rank":0, "beta":0.1,
            "context":16, "dimension":16, "feedForward":32, "layers":1,
            "heads":2, "kvHeads":1, "seed":42, "memoryMib":32
        });
        let old: TrainArgs = serde_json::from_value(input.clone()).unwrap();
        assert_eq!(old.accumulate, 1);
        assert_eq!(old.threads, 0);
        for count in [1, 7, 1024] {
            input["accumulate"] = count.into();
            let args = checked_training_args(serde_json::from_value(input.clone()).unwrap()).unwrap();
            let flag = args.iter().position(|s| s == "--accumulate").unwrap();
            assert_eq!(args[flag + 1], count.to_string());
            let control = args.iter().position(|s| s == "--control-stdin").unwrap();
            assert_eq!(args[control + 1], "1");
        }
        for count in [0, 1025] {
            input["accumulate"] = count.into();
            assert!(checked_training_args(serde_json::from_value(input.clone()).unwrap()).is_err());
        }
        for invalid in [serde_json::json!(-1), serde_json::json!(1.5)] {
            input["accumulate"] = invalid;
            assert!(serde_json::from_value::<TrainArgs>(input.clone()).is_err());
        }
        input["accumulate"] = 1.into();
        for threads in [0,1,6,64] {
            input["threads"] = threads.into();
            let args = checked_training_args(serde_json::from_value(input.clone()).unwrap()).unwrap();
            let flag = args.iter().position(|s| s == "--threads").unwrap();
            assert_eq!(args[flag+1],threads.to_string());
        }
        input["threads"] = 65.into();
        assert!(checked_training_args(serde_json::from_value(input.clone()).unwrap()).is_err());
        input["threads"] = serde_json::json!(-1);
        assert!(serde_json::from_value::<TrainArgs>(input.clone()).is_err());
        input["threads"] = 0.into();
        let old: TrainArgs = serde_json::from_value(input.clone()).unwrap();
        assert_eq!(old.eval_every,10); assert_eq!(old.eval_records,0); assert!(old.eval_data.is_none());
        assert!(!checked_training_args(old).unwrap().contains(&"--eval-data".into()));
        input["evalData"] = "validation data.tsv".into();
        input["evalEvery"] = 2.into(); input["evalRecords"] = 3.into();
        let args = checked_training_args(serde_json::from_value(input.clone()).unwrap()).unwrap();
        for (flag,value) in [("--eval-data","validation data.tsv"),("--eval-every","2"),("--eval-records","3")] {
            let i = args.iter().position(|s| s == flag).unwrap(); assert_eq!(args[i+1],value);
        }
        input["evalEvery"] = 0.into();
        assert!(checked_training_args(serde_json::from_value(input.clone()).unwrap()).is_err());
        input["evalEvery"] = serde_json::json!(-1);
        assert!(serde_json::from_value::<TrainArgs>(input).is_err());
    }
}

fn executable_name(stem: &str) -> String {
    format!("{}{}", stem, if cfg!(windows) { ".exe" } else { "" })
}

fn packaged_executable(app: &tauri::App, stem: &str) -> Option<PathBuf> {
    app.path()
        .resource_dir()
        .ok()
        .map(|p| p.join("backend").join(executable_name(stem)))
        .filter(|p| p.exists())
}

fn development_executable(stem: &str, env_name: &str) -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(env_name)
        .map(PathBuf::from)
        .filter(|p| p.exists())
    {
        return Some(path);
    }
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let name = executable_name(stem);
    [
        root.join("build-cuda").join(&name),
        root.join("build-cuda/Release").join(&name),
        root.join("build-vulkan").join(&name),
        root.join("build-cpu").join(&name),
        root.join("build-cpu/Release").join(&name),
    ]
    .into_iter()
    .find(|p| p.exists())
}

fn append_log(logs: &Logs, line: impl Into<String>) {
    if let Ok(mut entries) = logs.lock() {
        entries.push_back(line.into());
        while entries.len() > 1500 {
            entries.pop_front();
        }
    }
}

fn pipe_logs<R: std::io::Read + Send + 'static>(reader: R, logs: Logs, source: &'static str) {
    std::thread::spawn(move || {
        for line in BufReader::new(reader).lines().map_while(Result::ok) {
            append_log(&logs, format!("[{source}] {line}"));
        }
    });
}

fn spawn_backend(app: &tauri::App, logs: Logs) -> Result<Backend, String> {
    let exe = packaged_executable(app, "fyodor-backend")
        .or_else(|| development_executable("fyodor-backend", "FYODOR_BACKEND_BIN"))
        .ok_or("Fyodor backend executable was not bundled")?;
    let cwd = exe
        .parent()
        .and_then(Path::parent)
        .map(Path::to_path_buf)
        .unwrap_or_default();
    let mut command = background_command(&exe);
    command.args(["--port", "0"]);
    // Each model starts with the requested accelerator, with native CPU fallback.
    // The REST compute endpoint can later switch a single loaded model safely.
    command.env(
        "NYA_COMPUTE",
        std::env::var("NYA_COMPUTE").unwrap_or_else(|_| "cuda".into()),
    );
    if std::env::args().any(|arg| arg == "--verbose")
        || std::env::var("FYODOR_VERBOSE").as_deref() == Ok("1")
    {
        command.arg("--verbose");
    }
    // Prefer the packaged compiler over a build-machine absolute path. Driver
    // loading remains the C provider's responsibility (system nvcuda.dll).
    if std::env::var_os("NYA_CUDA_NVRTC").is_none() {
        if let Some(parent) = exe.parent() {
            if let Ok(entries) = std::fs::read_dir(parent) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().into_owned();
                    if name.starts_with("nvrtc64_") && name.ends_with(".dll") {
                        command.env("NYA_CUDA_NVRTC", entry.path());
                        break;
                    }
                }
            }
        }
    }
    let mut child = command
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Could not start {}: {e}", exe.display()))?;
    if let Some(stderr) = child.stderr.take() {
        pipe_logs(stderr, logs.clone(), "engine");
    }
    let stdout = child
        .stdout
        .take()
        .ok_or("Backend discovery pipe is unavailable")?;
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut line = String::new();
        let result = BufReader::new(stdout)
            .take(4096)
            .read_line(&mut line)
            .map(|_| line);
        let _ = tx.send(result);
    });
    let discovery = rx
        .recv_timeout(Duration::from_secs(12))
        .map_err(|_| "Backend startup timed out".to_string())
        .and_then(|result| result.map_err(|e| e.to_string()));
    let line = match discovery {
        Ok(line) => line,
        Err(error) => {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
    };
    let fields: Vec<_> = line.split_whitespace().collect();
    let valid_token = fields
        .get(3)
        .is_some_and(|token| token.len() == 64 && token.bytes().all(|c| c.is_ascii_hexdigit()));
    let port = fields
        .get(2)
        .and_then(|v| v.parse::<u16>().ok())
        .filter(|p| *p > 0);
    if fields.len() != 4
        || fields.first() != Some(&"FYODOR_READY")
        || fields.get(1) != Some(&"127.0.0.1")
        || port.is_none()
        || !valid_token
    {
        let _ = child.kill();
        let _ = child.wait();
        return Err("Backend returned an invalid discovery handshake".into());
    }
    let port = port.unwrap();
    append_log(&logs, format!("[desktop] engine ready on port {port}"));
    Ok(Backend {
        connection: Connection {
            base_url: format!("http://127.0.0.1:{port}"),
            token: fields[3].to_string(),
        },
        child: Mutex::new(Some(child)),
    })
}

#[tauri::command]
fn backend_connection(state: tauri::State<'_, Backend>) -> Connection {
    state.connection.clone()
}

#[tauri::command]
fn runtime_logs(state: tauri::State<'_, Trainer>) -> Vec<String> {
    state
        .logs
        .lock()
        .map(|v| v.iter().cloned().collect())
        .unwrap_or_default()
}

#[tauri::command]
fn clear_runtime_logs(state: tauri::State<'_, Trainer>) {
    if let Ok(mut logs) = state.logs.lock() {
        logs.clear();
    }
}

fn checked_training_args(input: TrainArgs) -> Result<Vec<String>, String> {
    if !matches!(input.mode.as_str(), "pretrain" | "cpt" | "sft" | "dpo") {
        return Err("Unsupported training mode".into());
    }
    if input.data.trim().is_empty() || input.output.trim().is_empty() {
        return Err("Dataset and output paths are required".into());
    }
    if input.mode != "pretrain" && input.base.as_deref().unwrap_or("").trim().is_empty() {
        return Err("This training mode requires a base model".into());
    }
    if input.steps == 0
        || !(1..=1024).contains(&input.accumulate)
        || input.threads > 64
        || input.eval_every == 0
        || input.rank > 256
        || input.memory_mib == 0
        || !input.learning_rate.is_finite()
        || input.learning_rate <= 0.0
        || !input.beta.is_finite()
        || input.beta <= 0.0
    {
        return Err("Training configuration is outside supported limits".into());
    }
    let mut args = vec![
        "--control-stdin".into(),
        "1".into(),
        "--mode".into(),
        input.mode,
        "--data".into(),
        input.data,
        "--output".into(),
        input.output,
        "--steps".into(),
        input.steps.to_string(),
        "--accumulate".into(),
        input.accumulate.to_string(),
        "--threads".into(),
        input.threads.to_string(),
        "--lr".into(),
        input.learning_rate.to_string(),
        "--rank".into(),
        input.rank.to_string(),
        "--beta".into(),
        input.beta.to_string(),
        "--context".into(),
        input.context.to_string(),
        "--dimension".into(),
        input.dimension.to_string(),
        "--ff".into(),
        input.feed_forward.to_string(),
        "--layers".into(),
        input.layers.to_string(),
        "--heads".into(),
        input.heads.to_string(),
        "--kv-heads".into(),
        input.kv_heads.to_string(),
        "--seed".into(),
        input.seed.to_string(),
        "--memory-mib".into(),
        input.memory_mib.to_string(),
    ];
    for (flag, value) in [
        ("--base", input.base),
        ("--checkpoint", input.checkpoint),
        ("--resume", input.resume),
    ] {
        if let Some(value) = value.filter(|v| !v.trim().is_empty()) {
            args.extend([flag.into(), value]);
        }
    }
    if let Some(path) = input.eval_data.filter(|v| !v.trim().is_empty()) {
        args.extend(["--eval-data".into(), path, "--eval-every".into(), input.eval_every.to_string(),
                     "--eval-records".into(), input.eval_records.to_string()]);
    }
    Ok(args)
}

#[tauri::command]
fn start_training(state: tauri::State<'_, Trainer>, args: TrainArgs) -> Result<(), String> {
    let args = checked_training_args(args)?;
    let mut slot = state
        .child
        .lock()
        .map_err(|_| "Training state is unavailable")?;
    if slot
        .as_mut()
        .is_some_and(|process| process.child.try_wait().ok().flatten().is_none())
    {
        return Err("A training run is already active".into());
    }
    if state.exit_waiter.load(Ordering::SeqCst) {
        return Err("The app is waiting for training to save before closing".into());
    }
    append_log(
        &state.logs,
        format!("[trainer] starting {}", args.join(" ")),
    );
    let mut child = background_command(&state.executable)
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("Could not start trainer: {e}"))?;
    if let Some(stdout) = child.stdout.take() {
        pipe_logs(stdout, state.logs.clone(), "trainer");
    }
    if let Some(stderr) = child.stderr.take() {
        pipe_logs(stderr, state.logs.clone(), "trainer");
    }
    *state
        .last_exit
        .lock()
        .map_err(|_| "Training state is unavailable")? = None;
    *slot = Some(TrainingProcess { child, stopping: false });
    Ok(())
}

#[tauri::command]
fn training_status(state: tauri::State<'_, Trainer>) -> TrainStatus {
    collect_training_status(&state)
}

fn collect_training_status(state: &Trainer) -> TrainStatus {
    let mut running = false;
    let mut stopping = false;
    if let Ok(mut slot) = state.child.lock() {
        if let Some(process) = slot.as_mut() {
            match process.child.try_wait() {
                Ok(None) => { running = true; stopping = process.stopping; },
                Ok(Some(status)) => {
                    let code = status.code().unwrap_or(-1);
                    if let Ok(mut exit) = state.last_exit.lock() {
                        *exit = Some(code);
                    }
                    append_log(&state.logs, format!("[trainer] exited with code {code}"));
                    *slot = None;
                }
                Err(error) => {
                    // A failed observation does not prove the child exited.
                    running = true;
                    stopping = process.stopping;
                    append_log(&state.logs, format!("[trainer] status error: {error}"));
                },
            }
        }
    } else {
        // Poisoned ownership is not evidence that it is safe to exit.
        running = true;
    }
    let exit_code = state.last_exit.lock().ok().and_then(|value| *value);
    TrainStatus { running, stopping, exit_code }
}

#[tauri::command]
fn stop_training(state: tauri::State<'_, Trainer>) -> Result<(), String> {
    request_training_stop(&state).map(|_| ())
}

fn request_training_stop(state: &Trainer) -> Result<bool, String> {
    let mut slot = state
        .child
        .lock()
        .map_err(|_| "Training state is unavailable")?;
    let Some(process) = slot.as_mut() else {
        return Ok(false);
    };
    if process.child.try_wait().map_err(|e| format!("Could not inspect trainer: {e}"))?.is_some() {
        return Ok(false);
    }
    if !process.stopping {
        let pipe = process.child.stdin.as_mut().ok_or("Trainer control pipe is unavailable")?;
        if let Err(error) = pipe.write_all(b"S") {
            // The child may finish between try_wait and the write.
            if process.child.try_wait().ok().flatten().is_some() { return Ok(false); }
            return Err(format!("Could not request a safe stop: {error}"));
        }
        process.stopping = true;
        append_log(&state.logs, "[trainer] stop requested; finishing the update and saving outputs");
    }
    Ok(true)
}

fn stop_backend(backend: &Backend) {
    if let Ok(mut stream) =
        TcpStream::connect(backend.connection.base_url.trim_start_matches("http://"))
    {
        let request = format!("POST /shutdown HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer {}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n", backend.connection.token);
        let _ = stream.set_write_timeout(Some(Duration::from_secs(1)));
        let _ = stream.write_all(request.as_bytes());
    }
    if let Ok(mut guard) = backend.child.lock() {
        if let Some(mut child) = guard.take() {
            for _ in 0..20 {
                if child.try_wait().ok().flatten().is_some() {
                    return;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .invoke_handler(tauri::generate_handler![
            backend_connection,
            runtime_logs,
            clear_runtime_logs,
            start_training,
            training_status,
            stop_training
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if let Some(trainer) = window.app_handle().try_state::<Trainer>() {
                    if collect_training_status(&trainer).running {
                        // Keep the window and logs available during saving,
                        // including when saving fails. ExitRequested owns the
                        // single asynchronous waiter for the whole app.
                        api.prevent_close();
                        window.app_handle().exit(0);
                    }
                }
            }
        })
        .setup(|app| {
            let logs = Arc::new(Mutex::new(VecDeque::new()));
            let trainer = packaged_executable(app, "fyodor-train")
                .or_else(|| development_executable("fyodor-train", "FYODOR_TRAIN_BIN"))
                .ok_or_else(|| {
                    std::io::Error::other("Fyodor trainer executable was not bundled")
                })?;
            let backend = spawn_backend(app, logs.clone()).map_err(std::io::Error::other)?;
            app.manage(backend);
            app.manage(Trainer {
                executable: trainer,
                child: Mutex::new(None),
                last_exit: Mutex::new(None),
                exit_waiter: AtomicBool::new(false),
                logs,
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building Fyodor");
    app.run(|handle, event| {
        if let RunEvent::ExitRequested { api, .. } = &event {
            if let Some(trainer) = handle.try_state::<Trainer>() {
                match request_training_stop(&trainer) {
                    Ok(true) => {
                        api.prevent_exit();
                        if !trainer.exit_waiter.swap(true, Ordering::SeqCst) {
                            let handle = handle.clone();
                            std::thread::spawn(move || {
                                let trainer = handle.state::<Trainer>();
                                let status = loop {
                                    let status = collect_training_status(&trainer);
                                    if !status.running { break status; }
                                    std::thread::sleep(Duration::from_millis(50));
                                };
                                if status.exit_code == Some(0) {
                                    handle.exit(0);
                                } else {
                                    trainer.exit_waiter.store(false, Ordering::SeqCst);
                                    append_log(&trainer.logs, "[trainer] app remains open because training/save failed; inspect the log");
                                }
                            });
                        }
                    },
                    Err(error) => {
                        api.prevent_exit();
                        append_log(&trainer.logs, format!("[trainer] close delayed: {error}"));
                    },
                    Ok(false) => {},
                }
            }
        }
        if matches!(event, RunEvent::Exit) {
            if let Some(backend) = handle.try_state::<Backend>() {
                stop_backend(&backend);
            }
        }
    })
}

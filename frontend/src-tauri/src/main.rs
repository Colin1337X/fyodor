// Release desktop launches own their sidecar pipes without a console window.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    fyodor_desktop_lib::run()
}

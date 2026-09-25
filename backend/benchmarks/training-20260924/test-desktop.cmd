@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
set "PATH=C:\Users\Colin\.cargo\bin;%PATH%"
cd /d "C:\Users\Colin\fyodor\frontend\src-tauri"
cargo test --release --lib
exit /b %errorlevel%

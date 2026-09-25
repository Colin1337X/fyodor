@echo off
rem The newer VS 18 installation lacks its CRT headers on this host.
rem Select the complete VS 2022 toolchain for the existing Tauri dependencies.
call "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b 1
set "PATH=C:\Program Files\nodejs;C:\Users\Colin\.cargo\bin;%PATH%"
cd /d "C:\Users\Colin\fyodor\frontend"
"C:\Program Files\nodejs\node.exe" node_modules/@tauri-apps/cli/tauri.js build --bundles nsis
exit /b %errorlevel%

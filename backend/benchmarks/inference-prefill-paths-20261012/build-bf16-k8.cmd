@echo off
call C:\BuildTools\VC\Auxiliary\Build\vcvars64.bat
if errorlevel 1 exit /b 1
set PATH=C:\Users\Colin\fyodor\.tools\cuda-toolkit\bin;%PATH%
C:\Users\Colin\fyodor\.tools\w64devkit\bin\cmake.exe -S C:\Users\Colin\.codex\worktrees\resident-training\fyodor\CUDA\cutlass -B D:\fyodor-validation\cutlass-bf16-k8-20261012 -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DNYA_CUTLASS_ROOT=C:\Users\Colin\fyodor\CUDA\cutlass\vendor\cutlass -DNYA_CUTLASS_ARCHITECTURES=120 -DNYA_CUTLASS_BF16X6=ON
if errorlevel 1 exit /b 1
C:\Users\Colin\fyodor\.tools\w64devkit\bin\cmake.exe --build D:\fyodor-validation\cutlass-bf16-k8-20261012 --parallel 4
exit /b %errorlevel%

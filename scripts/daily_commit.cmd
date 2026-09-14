@echo off
@chcp 65001 >nul
rem Commit and push only when the working tree has changes. Never creates empty commits.
setlocal
cd /d "%~dp0.."
set LOG=%~dp0daily_commit.log

git status --porcelain > "%TEMP%\mmrag_status.txt"
for %%A in ("%TEMP%\mmrag_status.txt") do if %%~zA==0 (
  echo %date% %time% no changes, skipped>> "%LOG%"
  exit /b 0
)

git add -A
git commit -m "일일 체크포인트 %date:~0,10%"
if errorlevel 1 ( echo %date% %time% commit failed>> "%LOG%" & exit /b 1 )

git push origin main
if errorlevel 1 ( echo %date% %time% push failed, commit kept locally>> "%LOG%" & exit /b 1 )
echo %date% %time% committed and pushed>> "%LOG%"

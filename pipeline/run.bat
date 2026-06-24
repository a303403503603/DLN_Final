@echo off
cd /d "%~dp0.."
set MODE=%1
if "%MODE%"=="" (
    echo Usage: run.bat ^<daily^|trade^|bot^|reconcile^|fetch^|gru^|status^>
    echo.
    echo   daily      Fetch + GRU + reconcile
    echo   trade      PPO trading
    echo   bot        Discord bot (background, pythonw)
    echo   reconcile  核對掛單 vs API
    echo   fetch      API data fetch only
    echo   gru        GRU cache update only
    echo   status     Pipeline status
    exit /b 1
)

if "%MODE%"=="bot" (
    C:\ProgramData\anaconda3\envs\dl_final\pythonw.exe -m pipeline.discord_bot
) else (
    C:\ProgramData\anaconda3\envs\dl_final\python.exe -m pipeline.run_daily_pipeline %MODE% >> logs\task_%MODE%.log 2>&1
)

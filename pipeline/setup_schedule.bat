@echo off
schtasks /create /tn "PPO_Daily" /tr "\"C:\Users\a3034\Desktop\DL_Final\grullr_stock\pipeline\run.bat\" daily" /sc weekly /d MON,TUE,WED,THU,FRI /st 16:30 /f
schtasks /delete /tn "PPO_Trade" /f 2>nul
echo.
echo Daily schedule created at 16:30 (fetch + GRU + reconciliation).
echo Trade is handled by the Discord bot.
echo.
echo To start the bot on login, run as Admin:
echo   schtasks /create /tn "PPO_Bot" /tr "\"%%CD%%\..\run.bat\" bot" /sc onlogon /rl highest /f
pause

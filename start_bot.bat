@echo off
cd /d "C:\Users\a3034\Desktop\DL_Final\grullr_stock"
start /B /MIN "" "C:\ProgramData\Anaconda3\envs\dl_final\python.exe" -X utf8 "pipeline\discord_bot.py" > "logs\discord_bot_stdout.log" 2>&1

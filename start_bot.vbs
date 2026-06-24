Set objShell = CreateObject("Wscript.Shell")
objShell.Run "cmd /c cd /d """ & CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName) & """ && C:\ProgramData\anaconda3\envs\dl_final\pythonw.exe main.py bot", 0, False

@echo off
rem Launch player-one without leaving a console window open
start "" "%~dp0node_modules\electron\dist\electron.exe" "%~dp0."

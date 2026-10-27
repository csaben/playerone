# player-one: `just` lists these. Windows only (the dock and its recorder are Win32).
set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

default:
    @just --list

# Build the recorder sidecar (native/rec), which the dock runs
build:
    @cd native; cargo build --release

# Start the dock in the background (it keeps the controller passthrough plugged in while it runs)
start: build
    @Start-Process "{{justfile_directory()}}\node_modules\electron\dist\electron.exe" -ArgumentList "{{justfile_directory()}}"

# Whether anything is recording right now: restart or stop the dock only when this says idle
status:
    @$live = Get-ChildItem "$env:APPDATA\player-one\recordings" -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddSeconds(-20) }; if ($live) { "recording (files written in the last 20 s)" } else { "idle" }; $d = Get-CimInstance Win32_Process -Filter "Name='electron.exe'" | Where-Object { $_.CommandLine -notmatch '--type=' -and $_.CommandLine -match 'player-one' }; if ($d) { "dock running (pid $($d.ProcessId))" } else { "dock not running" }

# Stop the dock. Refuses while recording, since a stop from outside cuts the recording off; --force skips that
stop force="":
    @if ("{{force}}" -ne "--force") { $live = Get-ChildItem "$env:APPDATA\player-one\recordings" -Recurse -File -ErrorAction SilentlyContinue | Where-Object { $_.LastWriteTime -gt (Get-Date).AddSeconds(-20) }; if ($live) { throw "recording in progress: stop it in the panel first (or: just stop --force)" } }; Get-CimInstance Win32_Process -Filter "Name='electron.exe'" | Where-Object { $_.CommandLine -notmatch '--type=' -and $_.CommandLine -match 'player-one' } | ForEach-Object { Stop-Process -Id $_.ProcessId }; Start-Sleep 2; & "C:\Program Files\Nefarius Software Solutions\HidHide\x64\HidHideCLI.exe" --cloak-off 2>$null; "stopped"

# Stop (if idle), rebuild and start again
restart: stop start

# The recorder's log (sidecar events and errors)
log:
    @Get-Content "$env:APPDATA\player-one\rec.log" -Tail 40 -Wait

# Unit tests (dock and recorder)
test:
    @npm test
    @cd native; cargo test --release -p player-one-rec

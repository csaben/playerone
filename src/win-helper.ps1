# Long-lived helper: reads one JSON request per line on stdin, answers one JSON line.
#   {"op":"list","self":1234}                  -> top-level app windows, front to back
#   {"op":"fg"}                                -> the foreground window's handle
#   {"op":"type","hwnd":123,"text":"hi"}       -> bring that window forward, then type text into it
[Console]::InputEncoding = [Text.Encoding]::UTF8
[Console]::OutputEncoding = [Text.Encoding]::UTF8
Add-Type @"
using System; using System.Collections.Generic; using System.Text; using System.Threading; using System.Runtime.InteropServices;
public static class PoWin {
  public delegate bool EnumProc(IntPtr h, IntPtr l);
  [DllImport("user32.dll")] static extern bool EnumWindows(EnumProc f, IntPtr l);
  [DllImport("user32.dll")] static extern bool IsWindowVisible(IntPtr h);
  [DllImport("user32.dll")] public static extern bool IsWindow(IntPtr h);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetWindowText(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern int GetClassName(IntPtr h, StringBuilder s, int n);
  [DllImport("user32.dll")] static extern IntPtr GetWindow(IntPtr h, uint cmd);
  [DllImport("user32.dll")] static extern int GetWindowLong(IntPtr h, int i);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, out uint pid);
  [DllImport("user32.dll")] static extern uint GetWindowThreadProcessId(IntPtr h, IntPtr p);
  [DllImport("dwmapi.dll")] static extern int DwmGetWindowAttribute(IntPtr h, int a, out int v, int size);
  [DllImport("kernel32.dll")] static extern IntPtr OpenProcess(uint access, bool inherit, uint pid);
  [DllImport("kernel32.dll")] static extern bool CloseHandle(IntPtr h);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode)] static extern bool QueryFullProcessImageName(IntPtr h, int f, StringBuilder s, ref int n);
  [DllImport("kernel32.dll")] static extern uint GetCurrentThreadId();
  [DllImport("user32.dll")] static extern bool AttachThreadInput(uint a, uint b, bool attach);
  [DllImport("user32.dll")] static extern bool BringWindowToTop(IntPtr h);
  [DllImport("user32.dll")] static extern bool SetForegroundWindow(IntPtr h);
  [DllImport("user32.dll")] static extern IntPtr SetActiveWindow(IntPtr h);
  [DllImport("user32.dll")] static extern IntPtr SetFocus(IntPtr h);
  [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr h, int c);
  [DllImport("user32.dll")] static extern bool IsIconic(IntPtr h);
  [DllImport("user32.dll")] public static extern IntPtr GetForegroundWindow();
  [DllImport("user32.dll")] static extern void keybd_event(byte vk, byte scan, uint flags, UIntPtr extra);
  [DllImport("user32.dll", SetLastError = true)] static extern uint SendInput(uint n, INPUT[] inputs, int size);

  [StructLayout(LayoutKind.Sequential)] struct MOUSEINPUT { public int dx, dy; public uint data, flags, time; public IntPtr extra; }
  [StructLayout(LayoutKind.Sequential)] struct KEYBDINPUT { public ushort vk, scan; public uint flags, time; public IntPtr extra; }
  [StructLayout(LayoutKind.Explicit)] struct UNION { [FieldOffset(0)] public MOUSEINPUT mi; [FieldOffset(0)] public KEYBDINPUT ki; }
  [StructLayout(LayoutKind.Sequential)] struct INPUT { public uint type; public UNION u; }

  public class Win { public long Hwnd; public string Title; public string Cls; public uint Pid; public string Exe; }

  static readonly HashSet<string> SHELL = new HashSet<string> { "Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd" };

  // What Alt+Tab would show: visible, titled, unowned, not a tool window, not cloaked
  // (hidden UWP frames and windows on other virtual desktops are cloaked).
  public static List<Win> List(uint self) {
    var res = new List<Win>();
    EnumWindows((h, l) => {
      if (!IsWindowVisible(h) || GetWindow(h, 4) != IntPtr.Zero) return true;
      if ((GetWindowLong(h, -20) & 0x80) != 0) return true;
      int cloaked; if (DwmGetWindowAttribute(h, 14, out cloaked, 4) == 0 && cloaked != 0) return true;
      var t = new StringBuilder(512); GetWindowText(h, t, 512);
      if (t.Length == 0) return true;
      var c = new StringBuilder(256); GetClassName(h, c, 256);
      if (SHELL.Contains(c.ToString())) return true;
      uint pid; GetWindowThreadProcessId(h, out pid);
      if (pid == self) return true;
      res.Add(new Win { Hwnd = h.ToInt64(), Title = t.ToString(), Cls = c.ToString(), Pid = pid, Exe = ExePath(pid) });
      return true;
    }, IntPtr.Zero);
    return res;
  }

  static string ExePath(uint pid) {
    IntPtr p = OpenProcess(0x1000, false, pid);
    if (p == IntPtr.Zero) return "";
    try { var s = new StringBuilder(1024); int n = 1024; return QueryFullProcessImageName(p, 0, s, ref n) ? s.ToString() : ""; }
    finally { CloseHandle(p); }
  }

  // Windows only lets the foreground app hand over focus. Joining the input queue of the
  // window that has it (the dock) makes this thread count as that app for a moment.
  public static bool Activate(IntPtr h) {
    if (IsIconic(h)) ShowWindow(h, 9);
    for (int i = 0; i < 3 && GetForegroundWindow() != h; i++) {
      // A key tap lifts the foreground lock. Not Alt: it reaches the target after the
      // switch and opens its menu bar, so the typing goes to the menu. F24 does nothing.
      if (i > 0) { keybd_event(0x87, 0, 0, UIntPtr.Zero); keybd_event(0x87, 0, 2, UIntPtr.Zero); }
      IntPtr fg = GetForegroundWindow();
      uint fgThread = GetWindowThreadProcessId(fg, IntPtr.Zero), me = GetCurrentThreadId();
      bool joined = fgThread != 0 && fgThread != me && AttachThreadInput(me, fgThread, true);
      try { BringWindowToTop(h); SetForegroundWindow(h); SetActiveWindow(h); SetFocus(h); }
      finally { if (joined) AttachThreadInput(me, fgThread, false); }
      if (GetForegroundWindow() != h) Thread.Sleep(80);
    }
    return GetForegroundWindow() == h;
  }

  // Types as unicode key events, so layout and modifiers don't matter; "\n" is Enter.
  // Stops if the window loses the foreground, so keys never land somewhere else.
  public static int Type(IntPtr h, string text, int delayMs) {
    int size = Marshal.SizeOf(typeof(INPUT)), sent = 0;
    foreach (char ch in text) {
      if (GetForegroundWindow() != h) break;
      var down = new INPUT { type = 1 };
      var up = new INPUT { type = 1 };
      if (ch == '\n') { down.u.ki.vk = 0x0D; up.u.ki.vk = 0x0D; up.u.ki.flags = 2; }
      else { down.u.ki.scan = ch; down.u.ki.flags = 4; up.u.ki.scan = ch; up.u.ki.flags = 4 | 2; }
      if (SendInput(2, new[] { down, up }, size) != 2) break;
      sent++;
      if (delayMs > 0) Thread.Sleep(delayMs);
    }
    return sent;
  }
}
"@

function Reply($obj) { [Console]::Out.WriteLine((ConvertTo-Json -Compress -Depth 4 -InputObject $obj)); [Console]::Out.Flush() }

while ($true) {
  $line = [Console]::In.ReadLine()
  if ($null -eq $line) { break }
  try {
    $req = $line | ConvertFrom-Json
    if ($req.op -eq 'list') {
      $out = @(foreach ($w in [PoWin]::List([uint32]$req.self)) { @{ hwnd = $w.Hwnd; title = $w.Title; cls = $w.Cls; pid = $w.Pid; exe = $w.Exe } })
      Reply @{ ok = $true; windows = $out }
    }
    elseif ($req.op -eq 'fg') { Reply @{ ok = $true; hwnd = [PoWin]::GetForegroundWindow().ToInt64() } }
    elseif ($req.op -eq 'type') {
      $h = [IntPtr][int64]$req.hwnd
      if (-not [PoWin]::IsWindow($h)) { Reply @{ ok = $false; error = 'that window is gone' }; continue }
      if (-not [PoWin]::Activate($h)) { Reply @{ ok = $false; error = "couldn't bring it to the front" }; continue }
      Start-Sleep -Milliseconds 120
      $text = [string]$req.text
      $n = [PoWin]::Type($h, $text, [int]$req.delay)
      if ($n -lt $text.Length) { Reply @{ ok = $false; error = "stopped after $n of $($text.Length) keys (focus moved, or the window runs as admin)"; sent = $n }; continue }
      Reply @{ ok = $true; sent = $n }
    }
    else { Reply @{ ok = $false; error = "unknown op $($req.op)" } }
  } catch {
    Reply @{ ok = $false; error = $_.Exception.Message }
  }
}

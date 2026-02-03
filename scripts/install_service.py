#!/usr/bin/env python3
"""Install launchd service for zikaron daemon."""

import subprocess
import sys
from pathlib import Path

PLIST_CONTENT = '''<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.zikaron.daemon</string>
    
    <key>ProgramArguments</key>
    <array>
        <string>{daemon_path}</string>
    </array>
    
    <key>RunAtLoad</key>
    <true/>
    
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    
    <key>StandardOutPath</key>
    <string>{log_dir}/zikaron-daemon.log</string>
    
    <key>StandardErrorPath</key>
    <string>{log_dir}/zikaron-daemon.error.log</string>
    
    <key>WorkingDirectory</key>
    <string>{home_dir}</string>
    
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:~/.local/bin:/opt/homebrew/bin</string>
    </dict>
</dict>
</plist>'''


def find_daemon_executable() -> Path:
    """Find zikaron-daemon executable."""
    # Try which command first
    try:
        result = subprocess.run(["which", "zikaron-daemon"], 
                              capture_output=True, text=True, check=True)
        return Path(result.stdout.strip())
    except subprocess.CalledProcessError:
        pass
    
    # Try common locations
    possible_paths = [
        Path.home() / ".local" / "bin" / "zikaron-daemon",
        Path("/usr/local/bin/zikaron-daemon"),
        Path("/opt/homebrew/bin/zikaron-daemon"),
    ]
    
    for path in possible_paths:
        if path.exists():
            return path
    
    raise FileNotFoundError("zikaron-daemon executable not found")


def install_service():
    """Install launchd service."""
    try:
        daemon_path = find_daemon_executable()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure zikaron is installed: pip install -e .")
        sys.exit(1)
    
    home_dir = Path.home()
    log_dir = home_dir / ".local" / "share" / "zikaron" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    plist_dir = home_dir / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    
    plist_path = plist_dir / "com.zikaron.daemon.plist"
    
    # Generate plist content
    plist_content = PLIST_CONTENT.format(
        daemon_path=daemon_path,
        home_dir=home_dir,
        log_dir=log_dir
    )
    
    # Write plist file
    with open(plist_path, 'w') as f:
        f.write(plist_content)
    
    print(f"Created plist at: {plist_path}")
    
    # Load service
    try:
        subprocess.run(["launchctl", "load", str(plist_path)], check=True)
        print("Service loaded successfully")
        
        # Start service
        subprocess.run(["launchctl", "start", "com.zikaron.daemon"], check=True)
        print("Service started successfully")
        
        print("\nLogs available at:")
        print(f"  Output: {log_dir}/zikaron-daemon.log")
        print(f"  Errors: {log_dir}/zikaron-daemon.error.log")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to load/start service: {e}")
        sys.exit(1)


def uninstall_service():
    """Uninstall launchd service."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.zikaron.daemon.plist"
    
    if not plist_path.exists():
        print("Service not installed")
        return
    
    # Stop service (ignore errors - might not be running)
    subprocess.run(["launchctl", "stop", "com.zikaron.daemon"],
                  capture_output=True)

    # Unload service
    result = subprocess.run(["launchctl", "unload", str(plist_path)],
                          capture_output=True, text=True)

    if result.returncode != 0 and "Could not find" not in result.stderr:
        print(f"Warning: Failed to unload service: {result.stderr}")
        print("Plist file will not be removed. Please unload manually.")
        sys.exit(1)

    # Remove plist
    plist_path.unlink()
    print("Service uninstalled successfully")


def main():
    """Main entry point."""
    if len(sys.argv) != 2 or sys.argv[1] not in ["install", "uninstall"]:
        print("Usage: python install_service.py [install|uninstall]")
        sys.exit(1)
    
    action = sys.argv[1]
    
    if action == "install":
        install_service()
    elif action == "uninstall":
        uninstall_service()


if __name__ == "__main__":
    main()

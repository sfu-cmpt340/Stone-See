#!/usr/bin/env python3
"""
KidneyNet - Cross-Platform Runner Script
Works on Windows, macOS, and Linux
"""

import os
import sys
import subprocess
import platform
import time
import signal
import webbrowser
from pathlib import Path

# Color codes for terminal output (works on Windows 10+)
class Colors:
    GREEN = '\033[92m' if platform.system() != 'Windows' else ''
    YELLOW = '\033[93m' if platform.system() != 'Windows' else ''
    RED = '\033[91m' if platform.system() != 'Windows' else ''
    BLUE = '\033[94m' if platform.system() != 'Windows' else ''
    RESET = '\033[0m' if platform.system() != 'Windows' else ''
    BOLD = '\033[1m' if platform.system() != 'Windows' else ''

# Enable ANSI colors on Windows 10+
if platform.system() == 'Windows':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except:
        pass

def print_header():
    """Print welcome header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*50}")
    print("  KidneyNet - Starting Web Interface")
    print(f"{'='*50}{Colors.RESET}\n")

def check_dependencies():
    """Check if required dependencies are installed"""
    print(f"{Colors.YELLOW}Checking dependencies...{Colors.RESET}")
    
    # Check Python
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 7):
        print(f"{Colors.RED}✗ Python 3.7+ required. Found Python {python_version.major}.{python_version.minor}{Colors.RESET}")
        return False
    print(f"{Colors.GREEN}✓ Python {python_version.major}.{python_version.minor}{Colors.RESET}")
    
    # Check if backend dependencies exist
    backend_dir = Path(__file__).parent / "backend"
    if not (backend_dir / "app.py").exists():
        print(f"{Colors.RED}✗ Backend not found at {backend_dir}{Colors.RESET}")
        return False
    print(f"{Colors.GREEN}✓ Backend found{Colors.RESET}")
    
    # Check if frontend exists
    frontend_dir = Path(__file__).parent / "frontend"
    if not frontend_dir.exists():
        print(f"{Colors.RED}✗ Frontend not found at {frontend_dir}{Colors.RESET}")
        return False
    print(f"{Colors.GREEN}✓ Frontend found{Colors.RESET}")
    
    # Check if node/npm is installed (for frontend)
    try:
        npm_version = subprocess.run(
            ["npm", "--version"],
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        print(f"{Colors.GREEN}✓ npm {npm_version} found{Colors.RESET}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"{Colors.YELLOW}⚠ npm not found. Frontend may not work properly.{Colors.RESET}")
        print(f"{Colors.YELLOW}  Install Node.js from https://nodejs.org/{Colors.RESET}")
    
    print()
    return True

def kill_existing_servers():
    """Kill any existing servers on the ports"""
    print(f"{Colors.YELLOW}Stopping any existing servers...{Colors.RESET}")
    
    system = platform.system()
    
    if system == "Windows":
        # Windows: Use netstat and taskkill
        try:
            # Kill processes on port 5000 (backend)
            subprocess.run(
                ["netstat", "-ano"],
                capture_output=True,
                check=False
            )
            subprocess.run(
                ["for", "/f", "tokens=5", "%a", "in", "('netstat", "-ano", "|", "findstr", ":5000')", "do", "taskkill", "/F", "/PID", "%a"],
                shell=True,
                capture_output=True,
                check=False
            )
            # Kill processes on port 3000 (frontend)
            subprocess.run(
                ["for", "/f", "tokens=5", "%a", "in", "('netstat", "-ano", "|", "findstr", ":3000')", "do", "taskkill", "/F", "/PID", "%a"],
                shell=True,
                capture_output=True,
                check=False
            )
        except:
            pass
    else:
        # Unix/macOS: Use lsof or pkill
        try:
            # Kill backend
            subprocess.run(
                ["pkill", "-f", "python.*app.py"],
                capture_output=True,
                check=False
            )
            # Kill frontend
            subprocess.run(
                ["pkill", "-f", "next dev"],
                capture_output=True,
                check=False
            )
            subprocess.run(
                ["pkill", "-f", "node.*next"],
                capture_output=True,
                check=False
            )
        except:
            pass
    
    time.sleep(2)

def start_backend():
    """Start the Flask backend server"""
    print(f"{Colors.YELLOW}Starting backend API...{Colors.RESET}")
    
    project_root = Path(__file__).parent
    backend_dir = project_root / "backend"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Determine Python command
    python_cmd = "python" if platform.system() == "Windows" else "python3"
    
    # Start backend
    log_file = logs_dir / "backend.log"
    
    if platform.system() == "Windows":
        # Windows: Use start to run in new window, or run in background
        process = subprocess.Popen(
            [python_cmd, "app.py"],
            cwd=backend_dir,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0
        )
    else:
        # Unix/macOS
        process = subprocess.Popen(
            [python_cmd, "app.py"],
            cwd=backend_dir,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    # Wait for backend to start
    time.sleep(3)
    
    # Check if backend is running
    try:
        import urllib.request
        response = urllib.request.urlopen("http://localhost:5000/health", timeout=2)
        if response.status == 200:
            print(f"{Colors.GREEN}✓ Backend API running at http://localhost:5000{Colors.RESET}")
            return process
    except:
        pass
    
    print(f"{Colors.RED}✗ Backend failed to start. Check logs/backend.log{Colors.RESET}")
    print(f"{Colors.YELLOW}  Make sure you have installed requirements: pip install -r requirements.txt{Colors.RESET}")
    return None

def start_frontend():
    """Start the Next.js frontend server"""
    print(f"{Colors.YELLOW}Starting frontend...{Colors.RESET}")
    
    project_root = Path(__file__).parent
    frontend_dir = project_root / "frontend"
    logs_dir = project_root / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Check if node_modules exists
    if not (frontend_dir / "node_modules").exists():
        print(f"{Colors.YELLOW}Installing frontend dependencies...{Colors.RESET}")
        try:
            subprocess.run(
                ["npm", "install"],
                cwd=frontend_dir,
                check=True,
                capture_output=True
            )
            print(f"{Colors.GREEN}✓ Frontend dependencies installed{Colors.RESET}")
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}✗ Failed to install frontend dependencies{Colors.RESET}")
            print(f"{Colors.RED}  Error: {e}{Colors.RESET}")
            return None
    
    # Start frontend
    log_file = logs_dir / "frontend.log"
    
    if platform.system() == "Windows":
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if hasattr(subprocess, 'CREATE_NEW_PROCESS_GROUP') else 0
        )
    else:
        process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=frontend_dir,
            stdout=open(log_file, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True
        )
    
    # Wait for frontend to start
    time.sleep(5)
    
    # Check if frontend is running
    try:
        import urllib.request
        response = urllib.request.urlopen("http://localhost:3000", timeout=2)
        if response.status == 200:
            print(f"{Colors.GREEN}✓ Frontend running at http://localhost:3000{Colors.RESET}")
            return process
    except:
        pass
    
    print(f"{Colors.YELLOW}⚠ Frontend may still be starting. Check logs/frontend.log{Colors.RESET}")
    print(f"{Colors.YELLOW}  It may take a few more seconds...{Colors.RESET}")
    return process  # Return anyway, might still be starting

def open_browser():
    """Open browser to the frontend"""
    url = "http://localhost:3000"
    print(f"\n{Colors.YELLOW}Opening browser...{Colors.RESET}")
    try:
        webbrowser.open(url)
    except:
        print(f"{Colors.YELLOW}Please open {url} in your browser{Colors.RESET}")

def cleanup(backend_process, frontend_process):
    """Clean up processes on exit"""
    print(f"\n{Colors.YELLOW}Stopping servers...{Colors.RESET}")
    
    if backend_process:
        try:
            if platform.system() == "Windows":
                backend_process.terminate()
                time.sleep(1)
                if backend_process.poll() is None:
                    backend_process.kill()
            else:
                backend_process.terminate()
                time.sleep(1)
                if backend_process.poll() is None:
                    backend_process.kill()
        except:
            pass
    
    if frontend_process:
        try:
            if platform.system() == "Windows":
                frontend_process.terminate()
                time.sleep(1)
                if frontend_process.poll() is None:
                    frontend_process.kill()
            else:
                frontend_process.terminate()
                time.sleep(1)
                if frontend_process.poll() is None:
                    frontend_process.kill()
        except:
            pass
    
    # Also kill by port (in case processes didn't die)
    kill_existing_servers()
    
    print(f"{Colors.GREEN}✓ Servers stopped{Colors.RESET}")

def main():
    """Main function"""
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        print(f"{Colors.RED}Dependency check failed. Please install required dependencies.{Colors.RESET}")
        sys.exit(1)
    
    # Kill existing servers
    kill_existing_servers()
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        cleanup(backend_process, None)
        sys.exit(1)
    
    # Open browser
    time.sleep(2)
    open_browser()
    
    # Print success message
    print(f"\n{Colors.BOLD}{Colors.GREEN}{'='*50}")
    print("  ✓ Website is running!")
    print(f"{'='*50}{Colors.RESET}\n")
    print(f"Backend:  {Colors.BLUE}http://localhost:5000{Colors.RESET}")
    print(f"Frontend: {Colors.BLUE}http://localhost:3000{Colors.RESET}")
    print(f"\n{Colors.YELLOW}Press Ctrl+C to stop servers{Colors.RESET}\n")
    
    # Set up signal handlers for cleanup
    def signal_handler(sig, frame):
        cleanup(backend_process, frontend_process)
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Wait for processes
    try:
        backend_process.wait()
    except KeyboardInterrupt:
        pass
    finally:
        cleanup(backend_process, frontend_process)

if __name__ == "__main__":
    main()


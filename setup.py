import os
import sys
import subprocess
import platform

def run_command(command):
    """Run shell command and handle errors"""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result.stdout

def main():
    print(f"Detected OS: {platform.system()}")
    
    # Install requirements
    print("\n📦 Installing dependencies...")
    run_command(f"{sys.executable} -m pip install -r requirements.txt")
    
    # Clone and install taming-transformers
    if not os.path.exists("taming-transformers"):
        print("\n📥 Cloning taming-transformers...")
        run_command("git clone https://github.com/CompVis/taming-transformers.git")
    
    print("\n📦 Installing taming-transformers...")
    original_dir = os.getcwd()
    os.chdir("taming-transformers")
    run_command(f"{sys.executable} -m pip install -e .")
    os.chdir(original_dir)
    
    print("\n✅ Setup complete!")

if __name__ == "__main__":
    main()
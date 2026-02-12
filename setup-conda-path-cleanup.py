# Automatic PATH Cleanup for Conda Environment
# ==============================================
# This creates activation scripts that automatically clean PATH when you
# activate your conda environment
#
# USAGE:
#   python setup-conda-path-cleanup.py bushfire-py313
#
# This will create scripts in:
#   <conda-env>/etc/conda/activate.d/remove-gdal.bat
#   <conda-env>/etc/conda/deactivate.d/restore-gdal.bat

import sys
import os
from pathlib import Path
import subprocess

def get_conda_env_path(env_name):
    """Get the path to a conda environment"""
    try:
        result = subprocess.run(
            ['conda', 'env', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.split('\n'):
            if line.strip().startswith(env_name):
                parts = line.split()
                if len(parts) >= 2:
                    return parts[-1]
        
        print(f"Error: Could not find conda environment '{env_name}'")
        print("\nAvailable environments:")
        print(result.stdout)
        return None
    except Exception as e:
        print(f"Error getting conda environments: {e}")
        return None

def create_activation_scripts(env_path, patterns=['gdal', 'osgeo']):
    """Create activation/deactivation scripts for PATH cleanup"""
    
    env_path = Path(env_path)
    
    # Create directories if they don't exist
    activate_dir = env_path / 'etc' / 'conda' / 'activate.d'
    deactivate_dir = env_path / 'etc' / 'conda' / 'deactivate.d'
    
    activate_dir.mkdir(parents=True, exist_ok=True)
    deactivate_dir.mkdir(parents=True, exist_ok=True)
    
    # Pattern for matching (case-insensitive)
    pattern_regex = '|'.join(patterns)
    
    # Create activation script (removes GDAL from PATH)
    activate_script = activate_dir / 'remove-gdal-path.bat'
    activate_content = f'''@echo off
REM Auto-generated script to remove GDAL from PATH
REM Removes paths containing: {', '.join(patterns)}

REM Save original PATH for restoration
set ORIGINAL_PATH_BACKUP=%PATH%

REM Filter PATH using PowerShell
for /f "delims=" %%i in ('powershell -Command "($env:Path -split ';' | Where-Object {{ $_ -notmatch '{pattern_regex}' }}) -join ';'"') do set PATH=%%i

REM Optional: Show what was removed
REM echo [Conda Env] Removed GDAL paths from PATH
'''
    
    # Create deactivation script (restores original PATH)
    deactivate_script = deactivate_dir / 'restore-gdal-path.bat'
    deactivate_content = '''@echo off
REM Auto-generated script to restore original PATH

if defined ORIGINAL_PATH_BACKUP (
    set PATH=%ORIGINAL_PATH_BACKUP%
    set ORIGINAL_PATH_BACKUP=
    REM echo [Conda Env] Restored original PATH
)
'''
    
    # Write scripts
    activate_script.write_text(activate_content)
    deactivate_script.write_text(deactivate_content)
    
    print(f"✓ Created activation script: {activate_script}")
    print(f"✓ Created deactivation script: {deactivate_script}")
    print()
    print(f"PATH cleanup is now automatic for environment: {env_path.name}")
    print(f"Patterns filtered: {', '.join(patterns)}")
    print()
    print("To test:")
    print(f"  1. conda activate {env_path.name}")
    print(f"  2. echo %PATH%  (should not contain GDAL paths)")
    print(f"  3. conda deactivate  (restores original PATH)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python setup-conda-path-cleanup.py <conda-env-name> [pattern1] [pattern2] ...")
        print()
        print("Examples:")
        print("  python setup-conda-path-cleanup.py bushfire-py313")
        print("  python setup-conda-path-cleanup.py bushfire-py313 gdal osgeo")
        print()
        print("This will create conda activation hooks that automatically")
        print("remove specified paths when you activate the environment.")
        sys.exit(1)
    
    env_name = sys.argv[1]
    patterns = sys.argv[2:] if len(sys.argv) > 2 else ['gdal', 'osgeo']
    
    env_path = get_conda_env_path(env_name)
    if env_path:
        create_activation_scripts(env_path, patterns)
    else:
        sys.exit(1)

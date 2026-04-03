import os
import shutil
import tempfile
import git
from pathlib import Path
from config import SUPPORTED_EXTENSIONS, IGNORED_DIRS, IGNORED_FILES

def is_valid_file(filepath: Path) -> bool:
    """Check if the file is valid to be processed based on extensions and ignore lists."""
    if filepath.name in IGNORED_FILES:
        return False
        
    if not filepath.suffix or filepath.suffix not in SUPPORTED_EXTENSIONS:
        return False
        
    # Check if any parent directory is in the ignore list
    for part in filepath.parts:
        if part in IGNORED_DIRS:
            return False
            
    return True

def clone_github_repo(repo_url: str) -> str:
    """Clone a GitHub repository to a temporary directory and return the path."""
    temp_dir = tempfile.mkdtemp(prefix="ai_code_explainer_")
    try:
        print(f"Cloning {repo_url} into {temp_dir}...")
        git.Repo.clone_from(repo_url, temp_dir)
        return temp_dir
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise Exception(f"Failed to clone repository: {str(e)}")

def cleanup_temp_dir(dir_path: str):
    """Remove the temporary directory."""
    if dir_path and os.path.exists(dir_path):
        try:
            # handle read-only files in .git
            def rm_error(func, path, exc_info):
                import stat
                os.chmod(path, stat.S_IWRITE)
                func(path)
            shutil.rmtree(dir_path, onerror=rm_error)
        except Exception as e:
            print(f"Warning: Failed to clean up temp directory {dir_path}: {e}")

import os

def load_env(env_path=None):
    """
    Manually loads key-value pairs from a .env file into os.environ.
    If env_path is not specified, searches in the current directory and parent directories.
    """
    if env_path is None:
        # Search paths: current directory, parent directory, and grandparent directory
        search_paths = [
            ".env",
            "../.env",
            "../../.env",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
        ]
        for path in search_paths:
            if os.path.exists(path):
                env_path = path
                break
                
    if env_path and os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, val = line.split('=', 1)
                        val = val.strip().strip('"').strip("'")
                        os.environ[key.strip()] = val

"""Upload a local file to the remote server via SSH/SCP."""
import sys, pathlib
from remote_exec import load_cfg, connect

def main():
    if len(sys.argv) != 3:
        print("Usage: python upload.py <local> <remote>")
        sys.exit(1)
    local, remote = sys.argv[1], sys.argv[2]
    cfg = load_cfg()
    c = connect(cfg)
    sftp = c.open_sftp()
    sftp.put(local, remote)
    sftp.close()
    c.close()
    print(f"uploaded {local} -> {remote}")

if __name__ == "__main__":
    main()

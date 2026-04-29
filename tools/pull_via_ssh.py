"""Pull a remote file to local via SSH (base64 over stdout — works when SFTP is restricted).

Usage:
    python tools/pull_via_ssh.py <remote_path> <local_path>
"""
import sys, base64, pathlib
from remote_exec import load_cfg, connect


def pull_one(c, remote, local):
    cmd = f"base64 -w 0 {remote}"
    stdin, stdout, stderr = c.exec_command(cmd, timeout=120)
    rc = stdout.channel.recv_exit_status()
    if rc != 0:
        err = stderr.read().decode()
        raise RuntimeError(f"base64 failed for {remote}: {err}")
    b64 = stdout.read().decode("ascii").strip()
    data = base64.b64decode(b64)
    p = pathlib.Path(local)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return len(data)


def main():
    if len(sys.argv) != 3:
        print("Usage: python pull_via_ssh.py <remote> <local>")
        sys.exit(1)
    remote, local = sys.argv[1], sys.argv[2]
    cfg = load_cfg()
    c = connect(cfg)
    try:
        n = pull_one(c, remote, local)
        print(f"pulled {remote} -> {local} ({n} bytes)")
    finally:
        c.close()


if __name__ == "__main__":
    main()

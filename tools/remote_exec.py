"""Minimal SSH helper for running commands on SeetaCloud GPU server."""
import json, sys, os, pathlib

CFG_PATH = pathlib.Path(__file__).resolve().parent.parent / ".ssh_config.json"

def load_cfg():
    with open(CFG_PATH) as f:
        return json.load(f)

def connect(cfg):
    import paramiko
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(
        hostname=cfg["host"],
        port=int(cfg["port"]),
        username=cfg["user"],
        password=cfg["password"],
        timeout=15,
        banner_timeout=30,
    )
    return c

def run(cmd):
    cfg = load_cfg()
    c = connect(cfg)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=600)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    c.close()
    return rc, out, err

def main():
    if len(sys.argv) < 2:
        print("Usage: python remote_exec.py '<command>'")
        sys.exit(1)
    cmd = sys.argv[1]
    rc, out, err = run(cmd)
    if out:
        sys.stdout.buffer.write(out.encode("utf-8", errors="replace"))
    if err:
        sys.stderr.buffer.write(err.encode("utf-8", errors="replace"))
    sys.exit(rc)

if __name__ == "__main__":
    main()

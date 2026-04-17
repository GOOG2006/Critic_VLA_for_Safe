# Remote GPU Server — SeetaCloud

## Access
```
ssh -p 49161 root@connect.westb.seetacloud.com
# Password in .ssh_config.json (gitignored)
```

Helper: `python tools/remote_exec.py "<command>"`  
Upload: `python tools/remote_exec.py --put local remote`  
Download: `python tools/remote_exec.py --get remote local`

## Hardware
- **GPU**: NVIDIA RTX 4090 D (24GB VRAM)
- **RAM**: 755 GB
- **CPU**: unmeasured (shared container)
- **CUDA driver**: 13.0 (CUDA 12.4 wheels compatible)
- **OS**: Ubuntu 22.04 (Linux 5.15)

## Storage
| Mount | Size | Free | Use |
|-------|------|------|-----|
| `/` (overlay) | 30G | 14G | system — don't write here |
| `/root/autodl-tmp` | 50G | 32G | **workspace** — put code + small data here |
| `/autodl-pub/data` | 20T | 18T | shared dataset mount (read mostly) |
| `/dev/shm` | 30G | 30G | DataLoader workers / tmpfs |

## Existing Envs (user's other projects — don't touch)
- `base` — system
- `Evo1` — another project
- `metaworld` — another project

## Safe-MoLe Workspace
- Code: `/root/autodl-tmp/safe-mole/MoLe-VLA-Pytorch`
- Conda env: `MoLe_VLA` (Python 3.10)
- HF cache (may reuse): `/root/autodl-tmp/huggingface_cache` (existing)

## Install Plan (in progress)
1. ✅ conda create -n MoLe_VLA python=3.10
2. 🚧 pip install torch==2.5.1 + cu124 wheels
3. pip install -r deps from environment.yml (pip section only)
4. pip install flash-attn (slow build)
5. Download CogACT-Base checkpoint from HuggingFace → /root/autodl-tmp/
6. RLBench / CoppeliaSim installation

## Notes
- PyTorch 2.5.1 + CUDA 12.4 wheels, driver 13.0 OK
- Persistent storage only on `/root/autodl-tmp` — always cd here
- Check disk before big downloads: `df -h /root/autodl-tmp`

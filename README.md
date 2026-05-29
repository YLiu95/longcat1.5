# longcat1.5

Backup of the Kaggle-patched LongCat Video Avatar 1.5 workflow for 2×T4 GPUs, preserved so the same code can be restored in future Kaggle sessions.

## What is in this repo
- [`run_kaggle_avatar.py`](run_kaggle_avatar.py)
- [`longcat_video/modules/attention.py`](longcat_video/modules/attention.py)
- [`longcat_video/modules/avatar/attention.py`](longcat_video/modules/avatar/attention.py)
- [`longcat_video/modules/quantization.py`](longcat_video/modules/quantization.py)
- [`plan.md`](plan.md)
- [`backup/`](backup)

The runtime code above was copied from the live Kaggle working tree and preserved without changing behavior.

## Expected Kaggle hardware
- 2× Tesla T4 GPUs
- enough space in `/kaggle/tmp` for model weights
- no CPU offloading assumed

## Recommended paths in Kaggle
- repo/code: `/kaggle/tmp/LongCat-Video`
- base weights: `/kaggle/tmp/weights/LongCat-Video`
- avatar weights: `/kaggle/tmp/weights/LongCat-Video-Avatar-1.5`
- input json: `/kaggle/tmp/input.json`
- outputs: `/kaggle/working`

## Kaggle code cell: clone the backup repo
```python
!rm -rf /kaggle/tmp/LongCat-Video
!git clone https://github.com/YLiu95/longcat1.5 /kaggle/tmp/LongCat-Video
!ls -R /kaggle/tmp/LongCat-Video | head -200
```

## Kaggle code cell: install dependencies
```python
!pip install -q flash_attn==2.8.3 --no-build-isolation
!pip install -q audio-separator soundfile soxr pyloudnorm onnxruntime
```

If [`flash_attn`](run_kaggle_avatar.py:1) fails to install on T4, keep the patched attention files from this repo and continue with the fallback path already preserved in [`longcat_video/modules/attention.py`](longcat_video/modules/attention.py) and [`longcat_video/modules/avatar/attention.py`](longcat_video/modules/avatar/attention.py).

## Kaggle code cell: create input json
```python
import json

payload = {
    "prompt": "Static camera. An important Chinese political leader giving a speech to a large group of top tier elite officials. He smiles slightly and gestures gently with his right hand while speaking.",
    "cond_image": "/kaggle/input/datasets/yliu95/trump-xi-voice/zhuxi_speech_832_480.png",
    "cond_audio": {
        "person1": "/kaggle/input/datasets/yliu95/trump-xi-voice/toushang sanchi you shenming 3s.wav"
    }
}

with open('/kaggle/tmp/input.json', 'w') as f:
    json.dump(payload, f, indent=2)

print('Wrote /kaggle/tmp/input.json')
```

## Kaggle code cell: run the script
Use the preserved script directly:
```python
!python /kaggle/tmp/LongCat-Video/run_kaggle_avatar.py
```

If you also keep a separate config flow in your notebook, check the argument handling implemented in [`run_kaggle_avatar.py`](run_kaggle_avatar.py) before switching to a `--config` invocation.

## How to restore the exact patched files later
If you start from a fresh upstream clone, copy these preserved files back into the matching repo-relative paths:
- [`run_kaggle_avatar.py`](run_kaggle_avatar.py)
- [`longcat_video/modules/attention.py`](longcat_video/modules/attention.py)
- [`longcat_video/modules/avatar/attention.py`](longcat_video/modules/avatar/attention.py)
- [`longcat_video/modules/quantization.py`](longcat_video/modules/quantization.py)

Older flat snapshots are also kept under [`backup/`](backup) as an emergency fallback.

## How to update this backup repo from Kaggle next time
Store a GitHub PAT in Kaggle secrets, then use a code cell like this:
```python
from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
token = user_secrets.get_secret("GITHUB_PERSONAL_ACCESS_TOKEN")
os.environ["GITHUB_TOKEN"] = token
```

Then run shell commands:
```bash
git clone https://github.com/YLiu95/longcat1.5 /kaggle/tmp/github_repo
cd /kaggle/tmp/github_repo
cp /kaggle/tmp/LongCat-Video/run_kaggle_avatar.py ./run_kaggle_avatar.py
cp /kaggle/tmp/LongCat-Video/longcat_video/modules/attention.py ./longcat_video/modules/attention.py
cp /kaggle/tmp/LongCat-Video/longcat_video/modules/avatar/attention.py ./longcat_video/modules/avatar/attention.py
cp /kaggle/tmp/LongCat-Video/longcat_video/modules/quantization.py ./longcat_video/modules/quantization.py
git add .
git commit -m "Backup latest Kaggle LongCat patches"
git push https://$GITHUB_TOKEN@github.com/YLiu95/longcat1.5 HEAD:main
```

## Persistence advice
Kaggle sessions are ephemeral. Keeping the patched files in GitHub is the main backup. For extra safety, also export the repo as a Kaggle Dataset or zip archive in `/kaggle/working` after each successful run.

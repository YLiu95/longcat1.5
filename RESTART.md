# Restart Instructions — LongCat-Video-Avatar-1.5 on Kaggle 2×T4

If the Kaggle session crashes/restarts, follow these steps to restore everything.

## Branches

- **`main`** — Last known stable state (Run 17)
- **`experiment`** — Latest code with configurable settings + async GPU prefetch

Use `experiment` for the latest features. Use `main` to roll back if `experiment` has issues.

## Step 1: Install Dependencies (~2 min)

```bash
pip install audio-separator soundfile soxr pyloudnorm onnxruntime
```

> Note: flash_attn is NOT needed — we use SDPA with EFFICIENT_ATTENTION backend.

## Step 2: Clone Repo (~1 min)

```bash
git clone --single-branch --branch main https://github.com/meituan-longcat/LongCat-Video /kaggle/tmp/LongCat-Video
```

## Step 3: Download Model Weights (~20-30 min)

```bash
# LongCat-Video base models (tokenizer, text encoder, VAE)
huggingface-cli download meituan-longcat/LongCat-Video \
  --include "tokenizer/*" "text_encoder/*" "vae/*" \
  --local-dir /kaggle/tmp/weights/LongCat-Video

# Avatar-1.5 models (DiT INT8, Whisper, vocal separator, LoRA, scheduler)  
huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --include "base_model_int8/*" "whisper-large-v3/*" "vocal_separator/*" "lora/*" "scheduler/*" \
  --local-dir /kaggle/tmp/weights/LongCat-Video-Avatar-1.5
```

## Step 4: Restore Modified Files from GitHub Backup

```bash
source /root/.env
cd /kaggle/tmp

# Clone the experiment branch (latest code with config support + async prefetch)
git clone -b experiment https://${GITHUB_TOKEN}@github.com/YLiu95/longcat1.5.git backup-restore
# Or use main branch for stable fallback:
# git clone -b main https://${GITHUB_TOKEN}@github.com/YLiu95/longcat1.5.git backup-restore

# Copy modified inference script
cp backup-restore/run_kaggle_avatar.py /kaggle/tmp/LongCat-Video/

# Copy patched attention modules (SDPA + fp16 fixes)
cp backup-restore/attention_patched.py /kaggle/tmp/LongCat-Video/longcat_video/modules/attention.py
cp backup-restore/avatar_attention_patched.py /kaggle/tmp/LongCat-Video/longcat_video/modules/avatar/attention.py

# Copy config and input files
cp backup-restore/config.json /kaggle/tmp/
cp backup-restore/input.json /kaggle/tmp/
```

## Step 5: (Optional) Edit Settings

Edit `/kaggle/tmp/config.json` to change output name, resolution, or other settings.
See the `__comment_*` keys in the JSON for documentation on each setting.

## Step 6: Run Inference

```bash
cd /kaggle/tmp/LongCat-Video
python -u run_kaggle_avatar.py --config /kaggle/tmp/config.json > /kaggle/working/run.log 2>&1
```

Monitor progress:
```bash
tail -f /kaggle/working/run.log
```

## ⚠️ Before Running Risky Processes

If you're about to run something that might OOM and crash the session:

1. **Save your work**: `cd /kaggle/tmp/backup-repo && git add -A && git commit -m "pre-risk checkpoint" && source /root/.env && git push https://YLiu95:${GITHUB_TOKEN}@github.com/YLiu95/longcat1.5.git experiment`
2. **Check memory headroom**: Run `cat /sys/fs/cgroup/memory.current` and `cat /sys/fs/cgroup/memory.max` — ensure at least 4GB free
3. **Drop page cache first**: `echo 3 > /proc/sys/vm/drop_caches` (if available)
4. **Phase 1-3 cache**: If `/kaggle/working/phase123_cache.pt` exists, the script will skip the 30GB model loading phases on restart

## Expected Output

- Video: `/kaggle/working/{output_name}.mp4` (default: `output.mp4`)
- Log: `/kaggle/working/run.log`
- Phase cache: `/kaggle/working/phase123_cache.pt` (reused on restart)

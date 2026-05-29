# LongCat-Video-Avatar-1.5 — Kaggle 2×T4 Runner

Generate talking-head videos from a single image + audio clip using AI, running on Kaggle's free 2×T4 GPUs.

## Quick Start (5 steps)

### 1. Install dependencies

```bash
pip install audio-separator soundfile soxr pyloudnorm onnxruntime
```

### 2. Clone the upstream repo + download model weights

```bash
# Clone the base LongCat-Video repo
git clone --single-branch --branch main \
  https://github.com/meituan-longcat/LongCat-Video /kaggle/tmp/LongCat-Video

# Download base model weights (~22GB)
huggingface-cli download meituan-longcat/LongCat-Video \
  --include "tokenizer/*" "text_encoder/*" "vae/*" \
  --local-dir /kaggle/tmp/weights/LongCat-Video

# Download avatar weights (~20GB)
huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 \
  --include "base_model_int8/*" "whisper-large-v3/*" "vocal_separator/*" "lora/*" "scheduler/*" \
  --local-dir /kaggle/tmp/weights/LongCat-Video-Avatar-1.5
```

### 3. Copy patched files from this repo

```bash
source /root/.env   # or set GITHUB_TOKEN manually
git clone -b experiment \
  https://${GITHUB_TOKEN}@github.com/YLiu95/longcat1.5.git /kaggle/tmp/backup-repo

# Copy the inference script + patched attention modules
cp /kaggle/tmp/backup-repo/run_kaggle_avatar.py /kaggle/tmp/LongCat-Video/
cp /kaggle/tmp/backup-repo/attention_patched.py \
   /kaggle/tmp/LongCat-Video/longcat_video/modules/attention.py
cp /kaggle/tmp/backup-repo/avatar_attention_patched.py \
   /kaggle/tmp/LongCat-Video/longcat_video/modules/avatar/attention.py

# Copy config + input files
cp /kaggle/tmp/backup-repo/config.json /kaggle/tmp/
cp /kaggle/tmp/backup-repo/input.json  /kaggle/tmp/
```

### 4. Edit your inputs and settings

**`/kaggle/tmp/input.json`** — what to generate:
```json
{
  "prompt": "A person giving a speech, gesturing gently while speaking.",
  "cond_image": "/path/to/face-image.png",
  "cond_audio": { "person1": "/path/to/speech.wav" }
}
```

**`/kaggle/tmp/config.json`** — how to generate (all optional, sensible defaults provided):

| Setting | Default | What it does |
|---------|---------|--------------|
| `output_name` | `"output"` | Output filename → `{output_name}.mp4` |
| `resolution` | `"480p"` | `"480p"` (480×832) or `"720p"` (768×1280) |
| `seed` | `42` | Random seed. Use `-1` for random each run |
| `num_frames` | `49` | Frames per segment. 49≈2s, 93≈3.7s (may OOM) |
| `num_inference_steps` | `8` | Denoising steps. Must be 8 for distill mode |
| `negative_prompt` | *(long string)* | Describes artifacts to avoid |

See [`config.json`](config.json) for the full list — every setting has a `__comment_*` key explaining what it does and valid values.

### 5. Run inference

```bash
cd /kaggle/tmp/LongCat-Video
python -u run_kaggle_avatar.py --config /kaggle/tmp/config.json
```

Output will be saved to `/kaggle/working/{output_name}.mp4`.

Monitor progress: `tail -f /kaggle/working/inference.log`

---

## How It Works

```
Image + Audio + Prompt
  → Text Encoder (UMT5-XXL, GPU0, then unloaded)
  → Audio Encoder (Whisper-Large-v3, GPU0, then unloaded)
  → VAE Encode (image → latent, GPU0, then unloaded)
  → DiT Denoising (INT8, split: blocks 0-23 on GPU0, blocks 24-47 on GPU1)
  → VAE Decode (latent → video frames, GPU0)
  → FFmpeg mux (video + audio → MP4)
```

- Each large model is **loaded, used, then unloaded** to stay within T4 memory
- The DiT model is split across both GPUs with **async prefetch** to overlap transfers
- For audio longer than ~2s, the video is generated in **multiple segments** automatically
- Phase 1-3 outputs are **cached** to `/kaggle/working/phase123_cache.pt` — if the session crashes during DiT generation, restart skips the 30GB model loading

## Branches

| Branch | Purpose |
|--------|---------|
| `main` | Last stable checkpoint (Run 17) |
| `experiment` | Latest code with configurable settings + GPU optimizations |

## If the Kaggle Session Crashes

See [`RESTART.md`](RESTART.md) for step-by-step recovery instructions.

**Short version:** The phase 1-3 cache (`phase123_cache.pt`) persists in `/kaggle/working/`. After reinstalling deps + re-downloading weights, just re-run the script — it picks up where it left off.

## Requirements

- **Hardware**: 2× Tesla T4 GPUs (16GB each), 30GB+ RAM
- **Storage**: ~42GB for model weights in `/kaggle/tmp/`
- **Software**: PyTorch 2.x, transformers, diffusers, safetensors, einops, librosa

## Persistence advice
Kaggle sessions are ephemeral. Keeping the patched files in GitHub is the main backup. For extra safety, also export the repo as a Kaggle Dataset or zip archive in `/kaggle/working` after each successful run.

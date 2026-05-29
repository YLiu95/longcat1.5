# Restart Instructions — LongCat-Video-Avatar-1.5 on Kaggle 2×T4

If the Kaggle session crashes/restarts, follow these steps to restore everything.

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
git clone https://${GITHUB_TOKEN}@github.com/YLiu95/longcat1.5.git backup-restore

# Copy modified inference script
cp backup-restore/run_kaggle_avatar.py /kaggle/tmp/LongCat-Video/

# Copy patched attention modules (SDPA + fp16 fixes)
cp backup-restore/attention_patched.py /kaggle/tmp/LongCat-Video/longcat_video/modules/attention.py
cp backup-restore/avatar_attention_patched.py /kaggle/tmp/LongCat-Video/longcat_video/modules/avatar/attention.py

# Copy input config
cp backup-restore/input.json /kaggle/tmp/
```

## Step 5: Run Inference

```bash
cd /kaggle/tmp/LongCat-Video
python -u run_kaggle_avatar.py > /kaggle/working/run.log 2>&1
```

Monitor progress:
```bash
tail -f /kaggle/working/run.log
```

## Expected Output

- Video: `/kaggle/working/output.mp4`
- Log: `/kaggle/working/run.log`
- Settings: 480×832, 25fps, 8 denoising steps (distill mode)
- Multi-segment: 49 frames per segment, segments auto-calculated from audio length
- Runtime: ~44 min per segment for DiT denoising

## Memory Safety Notes

- DiT uses pipeline parallelism: blocks 0-23 on GPU0, blocks 24-47 on GPU1
- Each GPU uses ~8GB for model weights + ~3-5GB for activations
- Max 49 frames per segment to stay within T4 memory limits
- PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True is set in the script
- Text encoder, audio encoder, and VAE use load-run-unload pattern

# Plan: LongCat-Video-Avatar-1.5 on Kaggle 2xT4

## TL;DR
Generate a video from image+audio using LongCat-Video-Avatar-1.5's AI2V (Audio-Image-to-Video) mode on Kaggle's free 2xT4 GPUs (16GB each). The 13.6B-param INT8-quantized DiT model (~16GB) exceeds a single T4's capacity, so we **split the model layers across both GPUs** (pipeline parallelism: blocks 0-23 on GPU0, blocks 24-47 on GPU1). Smaller models (text encoder, audio encoder, VAE) use **load-run-unload** on a single GPU. No CPU offloading.

## Workflow
1. Clone repo → install deps → download models
2. Write Python scripts with modifications  
3. Run, debug, iterate until output video produced
4. THEN convert working scripts into Kaggle notebook

## Kaggle Environment (verified)
- **GPUs**: 2x Tesla T4, 15360 MiB each, sm_75, CUDA 12.8
- **PyTorch**: 2.10.0+cu128 (NEWER than required 2.6.0 — use as-is)
- **Pre-installed**: transformers 5.0.0 ✓, diffusers 0.37.1 ✓, librosa 0.11.0 ✓, safetensors 0.7.0 ✓, einops 0.8.2 ✓
- **Missing**: flash_attn (2.8.3 available via pip), audio-separator, soundfile, soxr, pyloudnorm
- **Disk**: /kaggle/tmp → 1.1TB free (overlay), /kaggle/working → 20GB
- **RAM**: 31GB, 4 CPU cores
- **Dataset**: `/kaggle/input/datasets/yliu95/trump-xi-voice/` contains both image and audio files ✓
- **UMT5EncoderModel import**: verified working with transformers 5.0.0

---

## Research Findings

### Model Architecture
- **DiT (Diffusion Transformer)**: 13.6B params, 48 blocks, hidden_size=4096, 32 heads
- **INT8 Quantized DiT**: ~15.89GB on disk (4 shards), too large for single T4 (16GB)
- **Components needed for AI2V**:
  - Tokenizer (from LongCat-Video repo) — tiny
  - Text encoder / UMT5-XXL (from LongCat-Video) — ~20GB disk / ~10GB GPU bf16
  - VAE / AutoencoderKLWan (from LongCat-Video) — ~800MB disk / ~400MB GPU
  - Scheduler (from Avatar-1.5) — tiny config
  - DiT INT8 (from Avatar-1.5 `base_model_int8/`) — ~16GB
  - LoRA / `dmd_lora.safetensors` (from Avatar-1.5 `lora/`) — ~1-2GB
  - Whisper-Large-v3 (from Avatar-1.5) — ~3GB
  - Vocal separator ONNX (from Avatar-1.5) — ~100MB

### Input/Output Specs
- **Image**: Any resolution, auto-resized to target (480x832 for 480p). User's image is already 832x480 ✓
- **Audio**: WAV, loaded at 16kHz via librosa. Duration auto-padded to match video length
- **Video output**: 480x832 @ 25fps, 93 frames per segment (~3.72s), MP4 H.264
- **Segments**: `ceil(audio_duration_sec / 3.72)` — for 3s audio: 1 segment
- **Input JSON format**:
  ```json
  {
    "prompt": "description...",
    "cond_image": "path/to/image.png",
    "cond_audio": { "person1": "path/to/audio.wav" }
  }
  ```

### VRAM Budget (per T4 = 16GB, ~15.5GB usable)
| Phase | GPU0 | GPU1 | Notes |
|-------|------|------|-------|
| Text Encoding | ~10GB UMT5 | idle | Load-run-unload |
| Audio Encoding | ~3GB Whisper | idle | Load-run-unload |
| Image VAE Encode | ~400MB VAE | idle | Load-run-unload |
| DiT Generation | ~8GB (blocks 0-23) | ~8GB (blocks 24-47) | Pipeline parallel |
| VAE Decode | ~400MB VAE | idle | Load-run-unload |

Peak: ~10GB single-GPU phases, ~8GB+activations during DiT = ~11-12GB per GPU. Fits comfortably.

### Why NOT Context Parallelism
Context parallelism (official approach) **replicates** the full model on each GPU and splits the spatial computation. INT8 DiT is ~15.89GB → exceeds T4's 15.5GB usable VRAM. Issue #116 confirms even 24GB is tight for INT8.

### Why Pipeline Parallelism
Split DiT's 48 blocks across 2 GPUs: ~8GB model weights each, leaving ~7.5GB for activations. Runs as single process — no `torchrun`, no NCCL complications on Kaggle.

---

## Steps

### Phase A: Environment Setup

1. **Install missing dependencies** (flash_attn, audio-separator, soundfile, soxr, etc.)
   - `pip install flash_attn==2.8.3 --no-build-isolation` (pre-built wheel for CUDA 12.8)
   - `pip install audio-separator soundfile soxr pyloudnorm onnxruntime`
   - Keep existing PyTorch 2.10.0, transformers 5.0.0, diffusers 0.37.1

2. **Clone LongCat-Video repo to /kaggle/tmp**
   ```
   git clone --single-branch --branch main https://github.com/meituan-longcat/LongCat-Video /kaggle/tmp/LongCat-Video
   ```

### Phase B: Model Download

3. **Selective download from HuggingFace**
   - Authenticate with `HF_TOKEN` from Kaggle secrets
   - Download from `meituan-longcat/LongCat-Video`:
     - `--include "tokenizer/*" "text_encoder/*" "vae/*"` → `/kaggle/tmp/weights/LongCat-Video/`
   - Download from `meituan-longcat/LongCat-Video-Avatar-1.5`:
     - `--include "base_model_int8/*" "whisper-large-v3/*" "vocal_separator/*" "lora/*" "scheduler/*"` → `/kaggle/tmp/weights/LongCat-Video-Avatar-1.5/`
   - Total: ~42GB, ~7-14 min on Kaggle

### Phase C: Code Modifications (Cell 4)

4. **Create custom inference script** `/kaggle/tmp/LongCat-Video/run_kaggle_avatar.py`
   - Single-process, multi-GPU inference with load-run-unload + pipeline parallelism
   - Key functions:
     - `encode_text(prompt, neg_prompt, tokenizer_path, text_encoder_path, device)` → prompt_embeds
     - `encode_audio(audio_path, whisper_path, vocal_sep_path, device, fps)` → audio_emb
     - `encode_image(image_path, vae_path, device, resolution)` → image_latent
     - `run_dit_generation(dit_path, lora_path, scheduler_path, ...)` → output_latent
     - `decode_and_save(vae_path, output_latent, audio_path, output_dir, fps)` → video file

5. **Modify DiT model for pipeline parallelism** — edit `longcat_video/modules/avatar/longcat_video_dit_avatar.py`
   - Add `split_across_devices(device_0, device_1, split_point=24)` method
   - Moves `patch_embedding`, `timestep_embed`, text/audio projections, blocks[0:split_point] to device_0
   - Moves blocks[split_point:], `final_layer` to device_1
   - Modify `forward()` to transfer hidden_states between devices at the split point
   - Modify output transfer back to device_0

6. **Modify quantization loader** — edit `longcat_video/modules/quantization.py`
   - `load_quantized_dit()`: Add optional `device_map` parameter
   - After loading to CPU, call `model.split_across_devices(gpu0, gpu1)` instead of `.to(single_device)`

7. **Extract pipeline denoising logic** — read `longcat_video/pipeline_longcat_video_avatar.py`
   - Extract `generate_ai2v` logic into standalone function in the custom script
   - Key operations: noise init, CFG setup, denoising loop with scheduler, audio guidance
   - Must handle: `use_distill=True` (8 steps, guidance_scale=1.0), audio CFG

8. **OOM fallback logic** in the custom script
   - Try 1: 480p, 93 frames (full)
   - Try 2: 480p, 61 frames (~2.44s)
   - Try 3: 384x672, 93 frames (reduced resolution)
   - Never CPU offload — only reduce resolution/frames

### Phase D: Create Configuration & Input (Cell 5)

9. **Create input JSON** at `/kaggle/tmp/input.json`
   ```json
   {
     "prompt": "Static camera. An important Chinese political leader giving a speech to a large group of top tier elite officials. He smiles slightly and gestures gently with his right hand while speaking.",
     "cond_image": "/kaggle/input/datasets/yliu95/trump-xi-voice/zhuxi_speech_832_480.png",
     "cond_audio": {
       "person1": "/kaggle/input/datasets/yliu95/trump-xi-voice/toushang sanchi you shenming 3s.wav"
     }
   }
   ```

10. **Configurable settings block** (bottom of cell, with comments):
    ```python
    # ===== CONFIGURABLE SETTINGS =====
    RESOLUTION = "480p"          # "480p" (480x832) or "720p" (768x1280)
    NUM_INFERENCE_STEPS = 8      # 8 for distill mode (required for v1.5)
    TEXT_GUIDANCE_SCALE = 5.0    # Text CFG: 1.0-10.0, higher = more prompt adherence
    AUDIO_GUIDANCE_SCALE = 3.0   # Audio CFG: 3-5 optimal for lip sync accuracy
    NUM_SEGMENTS = 0             # 0 = auto-calculate from audio length
    SAVE_FPS = 25                # Output FPS (25 for v1.5)
    REF_IMG_INDEX = 10           # Reference image index: 0-30 (10=consistent, 30=less repetition)
    MASK_FRAME_RANGE = 3         # Higher = fewer repeated actions, too high = artifacts
    USE_INT8 = True              # INT8 quantization for DiT (required for T4)
    USE_DISTILL = True           # Distillation mode (required for v1.5)
    SPLIT_POINT = 24             # DiT block split point: 24 = even split of 48 blocks
    SEED = 42                    # Random seed for reproducibility
    OUTPUT_DIR = "/kaggle/working"
    LOG_FILE = "/kaggle/working/inference.log"
    ```

### Phase E: Execute Inference (Cell 6)

11. **Run the custom inference script**
    ```python
    import sys
    sys.path.insert(0, '/kaggle/tmp/LongCat-Video')
    exec(open('/kaggle/tmp/LongCat-Video/run_kaggle_avatar.py').read())
    ```
    OR
    ```bash
    cd /kaggle/tmp/LongCat-Video && python run_kaggle_avatar.py --config /kaggle/tmp/config.json
    ```

### Phase F: Verify & Display (Cell 7)

12. **Display output video** in notebook
    - Show video player widget
    - Print generation stats (time, memory usage, resolution, frames)
    - Display log file summary

---

## Relevant Files

### To Download/Use
- `meituan-longcat/LongCat-Video` HF repo → `tokenizer/`, `text_encoder/`, `vae/`
- `meituan-longcat/LongCat-Video-Avatar-1.5` HF repo → `base_model_int8/`, `whisper-large-v3/`, `vocal_separator/`, `lora/`, `scheduler/`

### To Clone & Modify
- `longcat_video/modules/avatar/longcat_video_dit_avatar.py` — add `split_across_devices()` method, modify `forward()` for cross-GPU transfer
- `longcat_video/modules/quantization.py` — `load_quantized_dit()` add device_map support
- `longcat_video/pipeline_longcat_video_avatar.py` — reference for denoising loop extraction (read `generate_ai2v`, `generate_avc`)
- `longcat_video/audio_process/` — reference for `get_audio_encoder`, `get_audio_feature_extractor`

### To Create
- `/kaggle/tmp/LongCat-Video/run_kaggle_avatar.py` — custom inference script with load-run-unload + pipeline parallelism
- `/kaggle/tmp/input.json` — input configuration
- Kaggle notebook `.ipynb` — final deliverable

---

## Verification

1. **Memory check**: Add `torch.cuda.memory_summary()` after each phase to confirm VRAM is freed
2. **Model loading check**: Verify each model loads to correct device and dtype (int8 for DiT, bf16 for others)
3. **Audio processing check**: Verify vocal extraction produces valid WAV, whisper encoder produces non-NaN embeddings
4. **Output check**: Verify output video exists at `/kaggle/working/`, is playable, has correct resolution/fps/duration
5. **Log check**: Verify `/kaggle/working/inference.log` captures all phases with timing
6. **GPU utilization**: Print `nvidia-smi` during DiT phase to confirm both GPUs active

---

## Decisions
- **Pipeline parallelism over context parallelism**: INT8 DiT (~16GB) exceeds single T4 capacity, context parallel replicates full model
- **Single process over torchrun**: Simpler on Kaggle, no NCCL issues, direct control over GPU placement
- **AI2V mode (not AT2V)**: User provides conditioning image, so Audio-Image-to-Video is appropriate
- **Distill mode required**: Avatar v1.5 requires `--use_distill`, uses 8 inference steps (vs 50 standard)
- **English prompt for Chinese audio**: Official examples use English prompts regardless of audio language. UMT5-XXL handles both.
- **No CPU offloading**: User explicitly forbade CPU offloading. OOM fallback reduces resolution/frames only.
- **Settings configurable**: All generation parameters exposed as variables with comments at bottom of config cell

---

## Further Considerations
1. **flash_attn compatibility**: T4 is sm_75. `flash_attn 2.7.4.post1` supports sm_75 but may need build from source (~10 min). If build fails, fall back to `torch.nn.functional.scaled_dot_product_attention` by modifying the DiT config's `attention_type` field.
2. **Download time**: ~42GB selective download. Kaggle HF downloads can be slow (~20-30 min). Consider caching models as Kaggle datasets for repeat runs.
3. **Lip redness issue**: Issue #113 reports `--use_distill` can cause increasingly red lips in long videos. For single-segment (3s), this should be minimal. If visible, reduce audio_guidance_scale.

#!/usr/bin/env python3
"""
LongCat-Video-Avatar-1.5 inference on Kaggle 2xT4.
Single-process, multi-GPU with:
  - Load-run-unload for text encoder, audio encoder, VAE
  - Pipeline parallelism for DiT (blocks 0-23 on GPU0, blocks 24-47 on GPU1)
  - No CPU offloading
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import sys
import gc
import json
import time
import math
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from einops import rearrange

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/kaggle/working/inference.log', mode='w'),
    ]
)
log = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    "base_model_dir": "/kaggle/tmp/weights/LongCat-Video",
    "avatar_model_dir": "/kaggle/tmp/weights/LongCat-Video-Avatar-1.5",
    "input_json": "/kaggle/tmp/input.json",
    "output_dir": "/kaggle/working",

    # Generation settings
    "resolution": "480p",           # "480p" (480x832) or "720p" (768x1280)
    "num_inference_steps": 8,       # 8 for distill mode (required for v1.5)
    "text_guidance_scale": 1.0,     # 1.0 for distill mode
    "audio_guidance_scale": 1.0,    # 1.0 for distill mode
    "num_frames": 49,               # 49 frames @ 25fps = ~1.96s (reduced for T4 memory)
    "save_fps": 25,
    "seed": 42,
    "use_distill": True,
    "use_int8": True,
    "split_point": 24,              # DiT block split: 24 = even split of 48

    # Avatar settings
    "ref_img_index": 10,
    "mask_frame_range": 3,
    "model_type": "avatar-v1.5",
    "max_sequence_length": 512,
}


def log_gpu_memory(label=""):
    """Log GPU memory usage for both devices."""
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        peak = torch.cuda.max_memory_allocated(i) / 1024**3
        total = (getattr(torch.cuda.get_device_properties(i), 'total_memory', None) or 
                getattr(torch.cuda.get_device_properties(i), 'total_mem', 0)) / 1024**3
        log.info(f"  GPU{i} [{label}]: {allocated:.2f}GB alloc / {peak:.2f}GB peak / {reserved:.2f}GB reserved / {total:.2f}GB total")


def torch_gc():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# ============================================================
# Phase 1: Text Encoding (load-run-unload on GPU0)
# ============================================================
def encode_text(prompt, negative_prompt, base_model_dir, device="cuda:0"):
    """Encode text prompt using UMT5 text encoder. Load-run-unload pattern."""
    import ftfy
    import html
    import regex as re
    from transformers import AutoTokenizer, UMT5EncoderModel

    log.info("=== Phase 1: Text Encoding ===")
    start = time.time()

    def prompt_clean(text):
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    # Load tokenizer (tiny, stays in RAM)
    tokenizer = AutoTokenizer.from_pretrained(base_model_dir, subfolder="tokenizer")

    # Load text encoder - force split across both GPUs (each T4 has ~14.6GB,
    # model is ~14.4GB so we MUST split to leave room for activations)
    log.info("  Loading UMT5 text encoder (split across 2 GPUs)...")
    max_mem = {0: "8GiB", 1: "8GiB"}  # Force even split
    text_encoder = UMT5EncoderModel.from_pretrained(
        base_model_dir, subfolder="text_encoder", torch_dtype=torch.bfloat16,
        device_map="auto", max_memory=max_mem,
    ).eval()
    log_gpu_memory("text_encoder loaded")

    max_seq_len = CONFIG["max_sequence_length"]
    # Find the device where inputs should go (first layer's device)
    encoder_device = next(text_encoder.parameters()).device

    def _encode_one(text):
        cleaned = prompt_clean(text)
        inputs = tokenizer(
            [cleaned], padding="max_length", max_length=max_seq_len,
            truncation=True, add_special_tokens=True,
            return_attention_mask=True, return_tensors="pt",
        )
        with torch.no_grad():
            embeds = text_encoder(
                inputs.input_ids.to(encoder_device),
                inputs.attention_mask.to(encoder_device),
            ).last_hidden_state
        # Move to CPU immediately to free GPU memory
        embeds = embeds.to(dtype=torch.bfloat16, device="cpu").unsqueeze(1)
        mask = inputs.attention_mask.clone()
        return embeds, mask

    prompt_embeds, prompt_mask = _encode_one(prompt)
    neg_embeds, neg_mask = _encode_one(negative_prompt)
    caption_channels = text_encoder.config.d_model

    # Unload text encoder from both GPUs
    del text_encoder
    torch_gc()
    log.info(f"  Text encoding done in {time.time()-start:.1f}s")
    log_gpu_memory("text_encoder unloaded")

    return prompt_embeds, prompt_mask, neg_embeds, neg_mask, caption_channels


# ============================================================
# Phase 2: Audio Encoding (load-run-unload on GPU0)
# ============================================================
def encode_audio(audio_path, avatar_model_dir, device="cuda:0", fps=25):
    """Encode audio using Whisper. Load-run-unload pattern."""
    import librosa
    import pyloudnorm as pyln
    from longcat_video.audio_process import get_audio_encoder, get_audio_feature_extractor

    log.info("=== Phase 2: Audio Encoding ===")
    start = time.time()

    sample_rate = 16000
    speech_array, sr = librosa.load(audio_path, sr=sample_rate)
    log.info(f"  Audio loaded: {len(speech_array)/sample_rate:.2f}s, sr={sr}")

    # Vocal separation (optional - skip if audio is already clean vocal)
    try:
        from audio_separator.separator import Separator
        vocal_sep_path = os.path.join(avatar_model_dir, 'vocal_separator', 'Kim_Vocal_2.onnx')
        if os.path.exists(vocal_sep_path):
            log.info("  Running vocal separation...")
            audio_output_dir_temp = Path("/kaggle/tmp/audio_temp")
            audio_output_dir_temp.mkdir(parents=True, exist_ok=True)
            sep = Separator(
                output_dir=str(audio_output_dir_temp / "vocals"),
                output_single_stem="vocals",
                model_file_dir=os.path.dirname(vocal_sep_path),
            )
            sep.load_model(os.path.basename(vocal_sep_path))
            outputs = sep.separate(audio_path)
            if outputs:
                vocal_path = audio_output_dir_temp / "vocals" / outputs[0]
                if vocal_path.exists():
                    speech_array, sr = librosa.load(str(vocal_path), sr=sample_rate)
                    log.info(f"  Vocal extracted: {len(speech_array)/sample_rate:.2f}s")
            del sep
    except Exception as e:
        log.warning(f"  Vocal separation skipped: {e}")

    # Load whisper encoder
    whisper_path = os.path.join(avatar_model_dir, 'whisper-large-v3')
    log.info("  Loading Whisper encoder...")
    audio_encoder = get_audio_encoder(whisper_path, model_type="avatar-v1.5").to(device)
    audio_feature_extractor = get_audio_feature_extractor(whisper_path, model_type="avatar-v1.5")
    log_gpu_memory("whisper loaded")

    # Loudness normalization
    meter = pyln.Meter(sample_rate)
    loudness = meter.integrated_loudness(speech_array)
    if abs(loudness) <= 100:
        speech_array = pyln.normalize.loudness(speech_array, loudness, -23)

    # Extract audio embeddings (same logic as pipeline's get_audio_embedding_whisper)
    audio_duration = len(speech_array) / sample_rate
    video_length = int(audio_duration * fps)

    MEL_CHUNK = 750 * 640
    ENC_CHUNK = 3000
    ENC_FPS = 50

    mel_chunks = []
    for i in range(0, len(speech_array), MEL_CHUNK):
        mel = audio_feature_extractor(
            speech_array[i:i + MEL_CHUNK],
            sampling_rate=sample_rate,
            return_tensors="pt",
        ).input_features
        mel_chunks.append(mel)
    audio_features = torch.cat(mel_chunks, dim=-1)
    audio_features = audio_features.to(audio_encoder.dtype)

    enc_chunks = []
    with torch.no_grad():
        for i in range(0, audio_features.shape[-1], ENC_CHUNK):
            chunk_hs = audio_encoder.encoder(
                audio_features[:, :, i:i + ENC_CHUNK].to(device),
                output_hidden_states=True,
            ).hidden_states
            enc_chunks.append(torch.stack(chunk_hs, dim=2))
    audio_prompts = torch.cat(enc_chunks, dim=1)
    audio_prompts = audio_prompts[:, :video_length * 2]

    def linear_interpolation_fps(features, input_fps, output_fps, output_len=None):
        features = features.transpose(1, 2)
        if output_len is None:
            output_len = int(features.shape[2] / float(input_fps) * output_fps)
        output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
        return output_features.transpose(1, 2)

    feat0 = linear_interpolation_fps(audio_prompts[:, :,  0: 8].mean(dim=2), ENC_FPS, fps, video_length)
    feat1 = linear_interpolation_fps(audio_prompts[:, :,  8:16].mean(dim=2), ENC_FPS, fps, video_length)
    feat2 = linear_interpolation_fps(audio_prompts[:, :, 16:24].mean(dim=2), ENC_FPS, fps, video_length)
    feat3 = linear_interpolation_fps(audio_prompts[:, :, 24:32].mean(dim=2), ENC_FPS, fps, video_length)
    feat4 = linear_interpolation_fps(audio_prompts[:, :, 32],                ENC_FPS, fps, video_length)
    audio_emb = torch.stack([feat0, feat1, feat2, feat3, feat4], dim=2)[0]  # [T, 5, D]

    if torch.isnan(audio_emb).any():
        raise ValueError("Audio embedding contains NaN values!")

    log.info(f"  Audio embedding shape: {audio_emb.shape}")

    # Unload whisper
    del audio_encoder, audio_feature_extractor, audio_prompts
    torch_gc()
    log.info(f"  Audio encoding done in {time.time()-start:.1f}s")
    log_gpu_memory("whisper unloaded")

    # Move to CPU temporarily 
    return audio_emb.cpu(), speech_array, sample_rate


# ============================================================
# Phase 3: Image VAE Encoding (load-run-unload on GPU0)
# ============================================================
def encode_image(image_path, base_model_dir, height, width, device="cuda:0"):
    """Encode conditioning image with VAE. Load-run-unload pattern."""
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    log.info("=== Phase 3: Image VAE Encoding ===")
    start = time.time()

    # Load VAE
    log.info("  Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        base_model_dir, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device).eval()
    log_gpu_memory("VAE loaded")

    vae_scale_temporal = vae.config.scale_factor_temporal  # 4
    vae_scale_spatial = vae.config.scale_factor_spatial    # 8

    # Process image
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_spatial)
    image = Image.open(image_path).convert("RGB")
    image_tensor = video_processor.preprocess(image, height=height, width=width, resize_mode='crop')
    image_tensor = image_tensor.to(device=device, dtype=torch.bfloat16)

    # Encode image
    with torch.no_grad():
        encoded = vae.encode(image_tensor.unsqueeze(2))  # add temporal dim
        cond_latent = encoded.latent_dist.mode()  # [1, C, 1, H/8, W/8]

    # Normalize latents
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(cond_latent.device, cond_latent.dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(cond_latent.device, cond_latent.dtype)
    cond_latent = (cond_latent - latents_mean) * latents_std

    # Get VAE config for later
    vae_config = {
        "scale_factor_temporal": vae_scale_temporal,
        "scale_factor_spatial": vae_scale_spatial,
        "latents_mean": vae.config.latents_mean,
        "latents_std": vae.config.latents_std,
        "z_dim": vae.config.z_dim,
    }

    # Unload VAE
    del vae, video_processor
    torch_gc()
    log.info(f"  Image encoding done in {time.time()-start:.1f}s, latent shape: {cond_latent.shape}")
    log_gpu_memory("VAE unloaded")

    return cond_latent.cpu(), vae_config


# ============================================================
# Phase 4: DiT Generation with Pipeline Parallelism
# ============================================================
def split_dit_across_devices(model, device_0, device_1, split_point=24):
    """Split DiT model blocks across two GPUs for pipeline parallelism."""
    log.info(f"  Splitting DiT: blocks 0-{split_point-1} -> {device_0}, blocks {split_point}-{len(model.blocks)-1} -> {device_1}")

    # Move shared components to device_0
    model.x_embedder.to(device_0)
    model.t_embedder.to(device_0)
    model.y_embedder.to(device_0)
    model.audio_proj.to(device_0)

    # Split blocks
    for i, block in enumerate(model.blocks):
        if i < split_point:
            block.to(device_0)
        else:
            block.to(device_1)

    # Final layer to device_1 (where last block output lives)
    model.final_layer.to(device_1)

    return model


def dit_forward_split(model, hidden_states, timestep, encoder_hidden_states,
                      encoder_attention_mask=None, num_cond_latents=0,
                      audio_embs=None, split_point=24,
                      device_0="cuda:0", device_1="cuda:1"):
    """Forward pass with pipeline parallelism across two GPUs."""
    B, _, T, H, W = hidden_states.shape
    patch_size = model.patch_size

    N_t = T // patch_size[0]
    N_h = H // patch_size[1]
    N_w = W // patch_size[2]

    # Expand timestep
    if len(timestep.shape) == 1:
        timestep = timestep.unsqueeze(1).expand(-1, N_t)

    dtype = model.x_embedder.proj.weight.dtype

    # === All on device_0 ===
    hidden_states = hidden_states.to(device_0, dtype=dtype)
    timestep_d0 = timestep.to(device_0, dtype=dtype)
    encoder_hidden_states = encoder_hidden_states.to(device_0, dtype=dtype)

    hidden_states = model.x_embedder(hidden_states)  # [B, N, C]

    with torch.amp.autocast(device_type='cuda', dtype=torch.float32):
        t = model.t_embedder(timestep_d0.float().flatten(), dtype=torch.float32).reshape(B, N_t, -1)

    encoder_hidden_states = model.y_embedder(encoder_hidden_states)

    # Audio processing (on device_0)
    audio_cond = audio_embs.to(device=device_0, dtype=dtype)
    first_frame_audio_emb_s = audio_cond[:, :1, ...]
    
    latter_frame_audio_emb = audio_cond[:, 1:, ...]
    vae_scale = model.vae_scale
    audio_window = model.audio_window
    latter_frame_audio_emb = rearrange(latter_frame_audio_emb, "b (n_t n) w s c -> b n_t n w s c", n=vae_scale)
    middle_index = audio_window // 2
    latter_first_frame_audio_emb = latter_frame_audio_emb[:, :, :1, :middle_index+1, ...]
    latter_first_frame_audio_emb = rearrange(latter_first_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_last_frame_audio_emb = latter_frame_audio_emb[:, :, -1:, middle_index:, ...]
    latter_last_frame_audio_emb = rearrange(latter_last_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_middle_frame_audio_emb = latter_frame_audio_emb[:, :, 1:-1, middle_index:middle_index+1, ...]
    latter_middle_frame_audio_emb = rearrange(latter_middle_frame_audio_emb, "b n_t n w s c -> b n_t (n w) s c")
    latter_frame_audio_emb_s = torch.concat([latter_first_frame_audio_emb, latter_middle_frame_audio_emb, latter_last_frame_audio_emb], dim=2)
    audio_hidden_states = model.audio_proj(first_frame_audio_emb_s, latter_frame_audio_emb_s)

    audio_hidden_states = audio_hidden_states[:, -N_t:]
    audio_hidden_states = rearrange(audio_hidden_states, "b t n c -> (b t) n c")

    # Text processing
    if encoder_attention_mask is not None:
        encoder_attention_mask = encoder_attention_mask.to(device_0)
        encoder_attention_mask_sq = encoder_attention_mask.squeeze(1).squeeze(1) if encoder_attention_mask.dim() > 2 else encoder_attention_mask
        encoder_hidden_states = encoder_hidden_states.squeeze(1).masked_select(
            encoder_attention_mask_sq.unsqueeze(-1) != 0
        ).view(1, -1, hidden_states.shape[-1])
        y_seqlens = encoder_attention_mask_sq.sum(dim=1).tolist()
    else:
        y_seqlens = [encoder_hidden_states.shape[2]] * encoder_hidden_states.shape[0]
        encoder_hidden_states = encoder_hidden_states.squeeze(1).view(1, -1, hidden_states.shape[-1])

    # Process blocks 0 to split_point-1 on device_0
    for i in range(split_point):
        hidden_states = model.blocks[i](
            hidden_states, encoder_hidden_states, t, y_seqlens,
            (N_t, N_h, N_w), num_cond_latents,
            audio_hidden_states=audio_hidden_states,
        )

    # === Transfer to device_1 ===
    hidden_states = hidden_states.to(device_1)
    t = t.to(device_1)
    encoder_hidden_states = encoder_hidden_states.to(device_1)
    audio_hidden_states = audio_hidden_states.to(device_1)

    # Process blocks split_point to end on device_1
    for i in range(split_point, len(model.blocks)):
        hidden_states = model.blocks[i](
            hidden_states, encoder_hidden_states, t, y_seqlens,
            (N_t, N_h, N_w), num_cond_latents,
            audio_hidden_states=audio_hidden_states,
        )

    # Final layer on device_1
    hidden_states = model.final_layer(hidden_states, t, (N_t, N_h, N_w))

    # Unpatchify
    hidden_states = model.unpatchify(hidden_states, N_t, N_h, N_w)
    hidden_states = hidden_states.to(torch.float32)

    # Transfer back to device_0
    return hidden_states.to(device_0)


def run_dit_generation(
    base_model_dir, avatar_model_dir, vae_config,
    prompt_embeds, prompt_mask, neg_embeds, neg_mask,
    cond_latent, audio_emb_cpu,
    height, width, total_video_frames, max_segment_frames, caption_channels,
    device_0="cuda:0", device_1="cuda:1",
):
    """Run DiT denoising loop with pipeline parallelism (multi-segment)."""
    from longcat_video.modules.quantization import load_quantized_dit
    from longcat_video.modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

    log.info("=== Phase 4: DiT Generation ===")
    start = time.time()

    # Free GPU reserved memory before loading DiT (frees ~6GB on GPU0)
    torch_gc()

    split_point = CONFIG["split_point"]
    num_inference_steps = CONFIG["num_inference_steps"]
    text_guidance_scale = CONFIG["text_guidance_scale"]
    audio_guidance_scale = CONFIG["audio_guidance_scale"]
    use_distill = CONFIG["use_distill"]
    seed = CONFIG["seed"]

    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        avatar_model_dir, subfolder="scheduler"
    )

    # Load DiT (INT8)
    log.info("  Loading INT8 DiT model...")
    dit = load_quantized_dit(avatar_model_dir, subfolder="base_model_int8", cp_split_hw=[1, 1])
    log_gpu_memory("DiT on CPU")

    # Load distillation LoRA (load weights only, enable after split)
    if use_distill:
        lora_path = os.path.join(avatar_model_dir, 'lora', 'dmd_lora.safetensors')
        if os.path.exists(lora_path):
            log.info("  Loading distillation LoRA...")
            dit.load_lora(lora_path, "dmd", multiplier=1.0, lora_network_dim=128, lora_network_alpha=64)

    # Split across GPUs
    dit = split_dit_across_devices(dit, device_0, device_1, split_point)
    dit.eval()

    # Force SDPA fallback: disable flash_attn/xformers flags on all attention modules
    for module in dit.modules():
        if hasattr(module, 'enable_flashattn2'):
            module.enable_flashattn2 = False
        if hasattr(module, 'enable_flashattn3'):
            module.enable_flashattn3 = False
        if hasattr(module, 'enable_xformers'):
            module.enable_xformers = False
        if hasattr(module, 'enable_bsa'):
            module.enable_bsa = False

    log_gpu_memory("DiT split across GPUs")

    # Enable LoRA after split - move each LoRA to the device of its target module
    if use_distill and "dmd" in dit.lora_dict:
        dit.disable_all_loras()
        module_loras = {}
        for lora in dit.lora_dict["dmd"].loras:
            module_name = lora.lora_name.replace("lora___lorahyphen___", "").replace("___lorahyphen___", ".")
            if module_name not in module_loras:
                module_loras[module_name] = []
            module_loras[module_name].append(lora)
        for module_name, loras in module_loras.items():
            module = dit._get_module_by_name(module_name)
            # Get device of the target module (check both parameters and buffers for QuantizedLinear)
            import itertools
            try:
                target_device = next(itertools.chain(module.parameters(), module.buffers())).device
            except StopIteration:
                target_device = device_0
            target_dtype = torch.bfloat16
            for lora in loras:
                lora.to(target_device, dtype=target_dtype, non_blocking=True)
            if not hasattr(module, 'org_forward'):
                module.org_forward = module.forward
            module.forward = dit._create_multi_lora_forward(module, loras)
        dit.active_loras.append("dmd")

    # Prepare timesteps/sigmas
    _num_timesteps = 1000
    _num_distill_sample_steps = 8
    if use_distill:
        distill_indices = torch.arange(1, _num_distill_sample_steps + 1, dtype=torch.float32)
        distill_indices = (distill_indices * (_num_timesteps // _num_distill_sample_steps)).round().long()
        distill_indices = _num_timesteps - distill_indices
        sigmas = torch.flip(torch.linspace(0, 1, _num_timesteps), [0])
        sigmas = torch.flip(sigmas[distill_indices], [0]).float()
    else:
        sigmas = torch.linspace(1, 0.001, num_inference_steps)
    sigmas = sigmas.to(torch.float32)

    # Latent dimensions
    vae_scale_temporal = vae_config["scale_factor_temporal"]
    vae_scale_spatial = vae_config["scale_factor_spatial"]
    num_channels_latents = 16

    num_latent_frames = (max_segment_frames - 1) // vae_scale_temporal + 1
    latent_h = height // vae_scale_spatial
    latent_w = width // vae_scale_spatial
    shape = (1, num_channels_latents, num_latent_frames, latent_h, latent_w)

    generator = torch.Generator(device=device_0).manual_seed(seed)

    do_cfg = text_guidance_scale > 1.0 or audio_guidance_scale > 1.0

    # Prepare text embeddings (same for all segments)
    prompt_embeds = prompt_embeds.to(device_0)
    prompt_mask = prompt_mask.to(device_0)
    neg_embeds = neg_embeds.to(device_0)
    neg_mask = neg_mask.to(device_0)

    # Calculate segments
    new_per_seg = max_segment_frames - 1  # new frames per segment (excl. conditioning frame)
    if total_video_frames <= max_segment_frames:
        num_segments = 1
    else:
        num_segments = 1 + math.ceil((total_video_frames - max_segment_frames) / new_per_seg)

    log.info(f"  Multi-segment: {num_segments} segment(s) of {max_segment_frames} frames, total target: {total_video_frames} frames")

    # Audio window parameters
    audio_window = 5
    audio_stride = 1
    aw_indices = torch.arange(audio_window) - audio_window // 2  # [-2, -1, 0, 1, 2]

    all_segment_latents = []
    current_cond_latent = cond_latent.to(device_0)

    for seg_idx in range(num_segments):
        seg_start = time.time()
        log.info(f"  --- Segment {seg_idx+1}/{num_segments} ---")

        # Build audio embeddings for this segment
        frame_offset = seg_idx * new_per_seg
        audio_emb = audio_emb_cpu.to(device_0)
        center_indices = torch.arange(frame_offset, frame_offset + max_segment_frames, audio_stride).unsqueeze(1) + aw_indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, 0, audio_emb.shape[0] - 1)
        audio_embs = audio_emb[center_indices][None, ...]  # [1, T, W, S, C]
        del audio_emb

        if do_cfg:
            audio_uncond_embs = torch.zeros_like(audio_embs)
            combined_audio = torch.cat([audio_embs, audio_embs], dim=0)
            combined_embeds = torch.cat([neg_embeds, prompt_embeds], dim=0)
            combined_mask = torch.cat([neg_mask, prompt_mask], dim=0)

        # Prepare latents for this segment
        latents = torch.randn(shape, generator=generator, device=device_0, dtype=torch.float32)
        latents[:, :, :1] = current_cond_latent

        # Reset scheduler for this segment
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas, device=device_0)
        timesteps = scheduler.timesteps

        log.info(f"  Denoising: {len(timesteps)} steps, {'CFG' if do_cfg else 'no CFG'}")
        log.info(f"  Latent shape: {latents.shape}, Audio embs: {audio_embs.shape}")

        torch_gc()
        log_gpu_memory(f"Before denoising seg {seg_idx+1}")

        # Denoising loop for this segment
        with torch.no_grad():
            for i, t_val in enumerate(tqdm(timesteps, desc=f"Denoising seg {seg_idx+1}")):
                step_start = time.time()

                if do_cfg:
                    latent_model_input = torch.cat([latents] * 2)
                    timestep = t_val.expand(2).to(dtype=torch.bfloat16)
                    timestep = timestep.unsqueeze(-1).repeat(1, latent_model_input.shape[2])
                    timestep[:, :1] = 0

                    noise_pred = dit_forward_split(
                        dit, latent_model_input, timestep,
                        combined_embeds, combined_mask,
                        num_cond_latents=1,
                        audio_embs=combined_audio,
                        split_point=split_point,
                        device_0=device_0, device_1=device_1,
                    )

                    timestep_uncond = t_val.expand(1).to(dtype=torch.bfloat16)
                    timestep_uncond = timestep_uncond.unsqueeze(-1).repeat(1, latents.shape[2])
                    timestep_uncond[:, :1] = 0

                    noise_pred_uncond = dit_forward_split(
                        dit, latents, timestep_uncond,
                        neg_embeds, neg_mask,
                        num_cond_latents=1,
                        audio_embs=audio_uncond_embs,
                        split_point=split_point,
                        device_0=device_0, device_1=device_1,
                    )

                    noise_pred_uncond_text, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred_final = (noise_pred_uncond +
                        text_guidance_scale * (noise_pred_cond - noise_pred_uncond_text) +
                        audio_guidance_scale * (noise_pred_uncond_text - noise_pred_uncond))
                else:
                    # No CFG - single forward pass
                    timestep = t_val.expand(1).to(dtype=torch.bfloat16)
                    timestep = timestep.unsqueeze(-1).repeat(1, latents.shape[2])
                    timestep[:, :1] = 0  # condition frame has t=0

                    noise_pred_final = dit_forward_split(
                        dit, latents, timestep,
                        prompt_embeds, prompt_mask,
                        num_cond_latents=1,
                        audio_embs=audio_embs,
                        split_point=split_point,
                        device_0=device_0, device_1=device_1,
                    )

                # Negate for scheduler compatibility
                noise_pred_final = -noise_pred_final

                # Update latents (only noise frames, not condition frame)
                latents[:, :, 1:] = scheduler.step(
                    noise_pred_final[:, :, 1:], t_val, latents[:, :, 1:], return_dict=False
                )[0]

                step_time = time.time() - step_start
                if i == 0 or (i + 1) % 4 == 0:
                    log.info(f"  Step {i+1}/{len(timesteps)}: {step_time:.1f}s")

        # Save segment latents to CPU
        all_segment_latents.append(latents.cpu())

        # Set conditioning for next segment: last latent frame
        current_cond_latent = latents[:, :, -1:].clone()

        del latents, audio_embs
        if do_cfg:
            del audio_uncond_embs, combined_audio
        torch_gc()

        seg_time = time.time() - seg_start
        log.info(f"  Segment {seg_idx+1} done in {seg_time:.1f}s ({seg_time/60:.1f}min)")
        log_gpu_memory(f"After segment {seg_idx+1}")

    log.info(f"  DiT generation done in {time.time()-start:.1f}s")

    # Unload DiT
    del dit
    torch_gc()
    log_gpu_memory("DiT unloaded")

    return all_segment_latents


# ============================================================
# Phase 5: VAE Decode & Save (load-run-unload on GPU0)
# ============================================================
def decode_and_save(all_segment_latents, base_model_dir, vae_config, audio_path,
                    output_dir, fps, total_video_frames):
    """Decode multi-segment latents with VAE and save video with audio."""
    from longcat_video.modules.autoencoder_kl_wan import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    log.info("=== Phase 5: VAE Decode & Save ===")
    start = time.time()
    device = "cuda:0"

    # Load VAE
    log.info("  Loading VAE for decoding...")
    vae = AutoencoderKLWan.from_pretrained(
        base_model_dir, subfolder="vae", torch_dtype=torch.bfloat16
    ).to(device).eval()
    log_gpu_memory("VAE loaded for decode")

    video_processor = VideoProcessor(vae_scale_factor=vae_config["scale_factor_spatial"])
    all_frames = []

    for seg_idx, seg_latents in enumerate(all_segment_latents):
        log.info(f"  Decoding segment {seg_idx+1}/{len(all_segment_latents)}...")

        latents = seg_latents.to(device, dtype=torch.bfloat16)
        latents_mean = torch.tensor(vae_config["latents_mean"]).view(1, -1, 1, 1, 1).to(device, torch.bfloat16)
        latents_std = 1.0 / torch.tensor(vae_config["latents_std"]).view(1, -1, 1, 1, 1).to(device, torch.bfloat16)
        latents = latents / latents_std + latents_mean

        with torch.no_grad():
            output_video = vae.decode(latents, return_dict=False)[0]

        segment_frames = video_processor.postprocess_video(output_video)[0]  # [T, H, W, C]

        if seg_idx == 0:
            all_frames.append(segment_frames)
        else:
            # Skip first frame (conditioning frame = last frame of previous segment)
            all_frames.append(segment_frames[1:])

        log.info(f"  Segment {seg_idx+1} decoded: {segment_frames.shape}")
        del latents, output_video
        torch_gc()

    # Unload VAE
    del vae
    torch_gc()

    # Concatenate all segments
    video_frames = np.concatenate(all_frames, axis=0)

    # Trim to total_video_frames
    if video_frames.shape[0] > total_video_frames:
        video_frames = video_frames[:total_video_frames]

    log.info(f"  Video decoded: {video_frames.shape} ({video_frames.shape[0]} frames, target was {total_video_frames})")

    os.makedirs(output_dir, exist_ok=True)

    # Save using imageio + ffmpeg
    import imageio
    import subprocess

    save_path = os.path.join(output_dir, "output")
    save_path_tmp = save_path + "-temp.mp4"

    video_np = (np.clip(video_frames, 0, 1) * 255).astype(np.uint8)
    writer = imageio.get_writer(save_path_tmp, fps=fps, quality=5)
    for frame in video_np:
        writer.append_data(frame)
    writer.close()

    # Mux with FULL original audio (video duration matches audio via multi-segment)
    final_path = save_path + ".mp4"
    subprocess.run([
        "ffmpeg", "-y",
        "-i", save_path_tmp,
        "-i", audio_path,
        "-c:v", "libx264", "-crf", "18",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest", final_path
    ], check=True, capture_output=True)

    # Cleanup temp file
    if os.path.exists(save_path_tmp):
        os.remove(save_path_tmp)

    log.info(f"  Video saved to {final_path}")
    log.info(f"  Duration: {video_frames.shape[0]/fps:.2f}s video, audio from original file")
    log.info(f"  VAE decode & save done in {time.time()-start:.1f}s")
    log_gpu_memory("after decode & save")

    return final_path


# ============================================================
# Main
# ============================================================
def main():
    total_start = time.time()
    log.info("=" * 60)
    log.info("LongCat-Video-Avatar-1.5 Inference on Kaggle 2xT4")
    log.info("=" * 60)

    # GPU check
    num_gpus = torch.cuda.device_count()
    log.info(f"GPUs available: {num_gpus}")
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
        log.info(f"  GPU{i}: {props.name}, {total/1024**3:.1f}GB")

    if num_gpus < 2:
        log.warning("Only 1 GPU available - will try to fit on single GPU")

    device_0 = "cuda:0"
    device_1 = "cuda:1" if num_gpus >= 2 else "cuda:0"

    # Load input config
    base_model_dir = CONFIG["base_model_dir"]
    avatar_model_dir = CONFIG["avatar_model_dir"]
    input_json = CONFIG["input_json"]

    with open(input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    prompt = input_data['prompt']
    image_path = input_data['cond_image']
    audio_path = input_data['cond_audio']['person1']

    negative_prompt = ("Close-up, bright tones, overexposed, static, blurred details, "
                      "subtitles, style, works, paintings, images, static, overall gray, "
                      "worst quality, low quality, JPEG compression residue, ugly, "
                      "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
                      "deformed, disfigured, misshapen limbs, fused fingers, still picture, "
                      "messy background, three legs, many people in the background, walking backwards")

    # Determine resolution
    resolution = CONFIG["resolution"]
    if resolution == "480p":
        height, width = 480, 832
    elif resolution == "720p":
        height, width = 768, 1280
    else:
        height, width = 480, 832

    max_segment_frames = CONFIG["num_frames"]  # max frames per segment
    fps = CONFIG["save_fps"]

    log.info(f"Input: image={image_path}, audio={audio_path}")
    log.info(f"Settings: {height}x{width}, max {max_segment_frames} frames/segment, {fps}fps, {CONFIG['num_inference_steps']} steps")

    # ---- Phase 1: Text Encoding ----
    prompt_embeds, prompt_mask, neg_embeds, neg_mask, caption_channels = encode_text(
        prompt, negative_prompt, base_model_dir, device=device_0
    )

    # ---- Phase 2: Audio Encoding ----
    audio_emb_cpu, speech_array, sample_rate = encode_audio(
        audio_path, avatar_model_dir, device=device_0, fps=fps
    )

    # ---- Phase 3: Image VAE Encoding ----
    cond_latent, vae_config = encode_image(
        image_path, base_model_dir, height, width, device=device_0
    )

    # Calculate total video frames from audio duration
    audio_duration = len(speech_array) / sample_rate
    total_video_frames = max(int(audio_duration * fps), max_segment_frames)
    log.info(f"Audio: {audio_duration:.2f}s -> {total_video_frames} total video frames")

    # ---- Phase 4: DiT Generation (multi-segment) ----
    all_segment_latents = run_dit_generation(
        base_model_dir, avatar_model_dir, vae_config,
        prompt_embeds, prompt_mask, neg_embeds, neg_mask,
        cond_latent, audio_emb_cpu,
        height, width, total_video_frames, max_segment_frames, caption_channels,
        device_0=device_0, device_1=device_1,
    )

    # ---- Phase 5: VAE Decode & Save ----
    output_path = decode_and_save(
        all_segment_latents, base_model_dir, vae_config,
        audio_path, CONFIG["output_dir"], fps, total_video_frames
    )

    total_time = time.time() - total_start
    log.info("=" * 60)
    log.info(f"DONE! Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    log.info(f"Output: {output_path}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

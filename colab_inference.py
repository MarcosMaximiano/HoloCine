import torch
from diffsynth.pipelines.wan_video_holocine import WanVideoHoloCinePipeline, ModelConfig
from diffsynth import save_video


def enforce_4t_plus_1(n: int) -> int:
    t = round((n - 1) / 4)
    return 4 * t + 1


def prepare_multishot_inputs(global_caption: str, shot_captions: list[str], total_frames: int, custom_shot_cut_frames: list[int] | None = None) -> dict:
    num_shots = len(shot_captions)
    if "This scene contains" not in global_caption:
        global_caption = global_caption.strip() + f" This scene contains {num_shots} shots."
    per_shot_string = " [shot cut] ".join(shot_captions)
    prompt = f"[global caption] {global_caption} [per shot caption] {per_shot_string}"

    processed_total_frames = enforce_4t_plus_1(total_frames)

    num_cuts = num_shots - 1
    processed_shot_cuts: list[int] = []
    if custom_shot_cut_frames:
        for frame in custom_shot_cut_frames:
            processed_shot_cuts.append(enforce_4t_plus_1(frame))
    else:
        if num_cuts > 0:
            ideal_step = processed_total_frames / num_shots
            for i in range(1, num_shots):
                approx_cut_frame = i * ideal_step
                processed_shot_cuts.append(enforce_4t_plus_1(round(approx_cut_frame)))

    processed_shot_cuts = sorted(set(processed_shot_cuts))
    processed_shot_cuts = [f for f in processed_shot_cuts if 0 < f < processed_total_frames]

    return {"prompt": prompt, "shot_cut_frames": processed_shot_cuts, "num_frames": processed_total_frames}


def build_pipeline(model_root: str = "./checkpoints", device: str = "cuda", torch_dtype: torch.dtype = torch.bfloat16, use_low_noise: bool = False) -> WanVideoHoloCinePipeline:
    dit_path = f"{model_root}/HoloCine_dit/full/full_high_noise.safetensors"
    if use_low_noise:
        dit_path = f"{model_root}/HoloCine_dit/full/full_low_noise.safetensors"

    pipe = WanVideoHoloCinePipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=[
            ModelConfig(path=f"{model_root}/Wan2.2-T2V-A14B/models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
            ModelConfig(path=dit_path, offload_device="cpu"),
            ModelConfig(path=f"{model_root}/Wan2.2-T2V-A14B/Wan2.1_VAE.pth", offload_device="cpu"),
        ],
    )
    pipe.enable_vram_management()
    pipe.to(device)
    return pipe


def run_inference(
    pipe: WanVideoHoloCinePipeline,
    output_path: str,
    global_caption: str | None = None,
    shot_captions: list[str] | None = None,
    prompt: str | None = None,
    negative_prompt: str | None = None,
    num_frames: int | None = None,
    shot_cut_frames: list[int] | None = None,
    seed: int = 0,
    tiled: bool = True,
    height: int = 480,
    width: int = 832,
    num_inference_steps: int = 50,
    fps: int = 15,
    quality: int = 5,
):
    pipe_kwargs: dict = {
        "negative_prompt": negative_prompt,
        "seed": seed,
        "tiled": tiled,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
    }

    if global_caption and shot_captions:
        if num_frames is None:
            raise ValueError("Must provide 'num_frames' for structured input (Mode 1).")
        inputs = prepare_multishot_inputs(global_caption, shot_captions, num_frames, shot_cut_frames)
        pipe_kwargs.update(inputs)
    elif prompt:
        pipe_kwargs["prompt"] = prompt
        if num_frames is not None:
            processed_frames = enforce_4t_plus_1(num_frames)
            pipe_kwargs["num_frames"] = processed_frames
        if shot_cut_frames is not None:
            processed_cuts = [enforce_4t_plus_1(f) for f in shot_cut_frames]
            pipe_kwargs["shot_cut_frames"] = processed_cuts
    else:
        raise ValueError("Invalid inputs. Provide either (global_caption, shot_captions, num_frames) OR (prompt).")

    final_pipe_kwargs = {k: v for k, v in pipe_kwargs.items() if v is not None}
    if "prompt" not in final_pipe_kwargs:
        raise ValueError("A 'prompt' or ('global_caption' + 'shot_captions') is required.")

    video = pipe(**final_pipe_kwargs)
    save_video(video, output_path, fps=fps, quality=quality)

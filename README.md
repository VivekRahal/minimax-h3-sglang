# MiniMax H3 Ref2VA with SGLang

A Google Colab workflow for running **MiniMaxAI/MiniMax-H3** reference-to-video-with-audio (`ref2va`) generation through **SGLang Diffusion**.

The notebook starts a local SGLang server, accepts ordered image/video/audio references, submits a MiniMax H3 video-generation job, polls its status, and downloads the final MP4 with synchronized audio.

## What this notebook includes

- MiniMax H3 `ref2va` inference with the official `MiniMaxAI/MiniMax-H3` checkpoint
- SGLang Diffusion local server setup
- Single-GPU and four-GPU execution profiles
- `kitchen_int8` quantization for single-GPU execution
- SageAttention backend configuration
- CUDA/GPU, RAM, and disk checks
- Ordered image, video, and audio reference uploads
- Automatic media probing with `ffprobe`
- SGLang `/v1/videos` API submission
- Job-status polling and server-health checks
- MP4 preview and download in Google Colab
- Troubleshooting guidance for OOMs, server startup, and model loading

## Notebook

### View the code on GitHub

Open:

[`MiniMaxH3_using_SGLang.ipynb`](./MiniMaxH3_using_SGLang.ipynb)

The repository notebook is committed **without saved execution outputs**, so GitHub can render the code reliably instead of embedding large Colab upload widgets/logs.

### Open directly in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/VivekRahal/minimax-h3-sglang/blob/main/MiniMaxH3_using_SGLang.ipynb)

Run the notebook from top to bottom in **Google Colab with a CUDA GPU**.

## Requirements

The notebook installs or uses:

- Python
- PyTorch with CUDA
- `sglang[diffusion]`
- `uv`
- `requests`
- `comfy-kitchen`
- `ffmpeg` / `ffprobe`
- Google Colab

The first MiniMax H3 startup can require a large checkpoint download and substantial temporary disk and host RAM.

## Model

The workflow uses:

```text
MiniMaxAI/MiniMax-H3
```

Task:

```text
ref2va
```

The notebook is designed for the official MiniMax H3 checkpoint used by SGLang. A Diffusers-specific quantized checkpoint should not be treated as an interchangeable SGLang model path.

## GPU profiles

The notebook selects a profile based on the visible GPUs:

### `single_large`

For a single GPU with roughly 80 GiB or more VRAM.

- INT8 transformer
- text encoder offloading
- SageAttention
- transformer kept resident where possible

### `single_safe`

For smaller single-GPU environments.

- INT8 inference
- more aggressive CPU/host-memory streaming
- `DIT_RESIDENT_LAYERS = 0` as the conservative baseline

### `four_gpu`

For four large GPUs.

- 4 GPUs
- tensor parallel size 2
- Ulysses degree 2
- resident weights and parallel execution

## Basic workflow

1. Install SGLang Diffusion and dependencies.
2. Verify CUDA, GPU memory, system RAM, and free disk space.
3. Start the local SGLang MiniMax H3 server.
4. Check the server health endpoint.
5. Upload reference images, videos, and optional audio.
6. Build ordered MiniMax H3 reference conditions.
7. Define the generation prompt.
8. Configure duration, aspect ratio, inference steps, resolution, and seed.
9. Submit the request to:

```text
POST /v1/videos
```

10. Poll the returned video job ID until generation completes.
11. Preview and download the generated MP4.

## Reference ordering

References are uploaded in the exact order required by the prompt.

The notebook maps them to tags such as:

```text
<Picture 1>
<Picture 2>
<Video 1>
<Audio 1>
```

Make sure the prompt tags match the uploaded reference order.

## Main generation settings

Example values in the notebook include:

```python
SECONDS = 5
SHORT_EDGE = 768
ASPECT_RATIO = "9:16"
NUM_INFERENCE_STEPS = 20
SEED = 42
```

You can adjust these based on available VRAM, desired quality, and generation time.

## Troubleshooting

Inspect recent server logs with:

```bash
tail -n 80 /content/sglang_h3_ref2va.log
```

If `single_large` runs out of memory:

- switch to `single_safe`
- keep `DIT_RESIDENT_LAYERS = 0`
- restart the SGLang server
- reduce `SHORT_EDGE`
- reduce video duration

INT8 inference may produce numerically different outputs from full-precision execution.

## Notes

- Keep the Colab runtime connected while generation is running.
- The first model startup can involve a very large download.
- Four large resident GPUs can avoid repeated CPU-to-GPU weight transfers.
- Test a short clip before starting a longer generation.
- The notebook itself notes that its configuration has not been GPU-tested in the workspace in which it was prepared.

## References

- [SGLang](https://github.com/sgl-project/sglang)
- [MiniMax H3 SGLang guide](https://github.com/sgl-project/sglang/blob/main/docs/cookbook/diffusion/MiniMax/MiniMax-H3.mdx)

## License

No license has been added yet. Add an appropriate license before redistributing or accepting external contributions.

# mlx-vlm-muse-glimmer

MLX / [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) model support for **Muse-Glimmer**,
a multimodal (vision + language) model. This is a pure architecture port — it loads and
runs **any** Muse-Glimmer checkpoint on Apple Silicon, including Meta's official release
and any fine-tuned or derived weights.

It ships the model class mlx-vlm needs (`muse_glimmer`): the Qwen2.5-VL-style vision
tower, the language model, the projector, and the image processor.

## What's here

```
muse_glimmer/
  config.py                    # VisionConfig / TextConfig / ModelConfig
  vision.py                    # 50-layer ViT: 2D-RoPE, window/full attention,
                               #   interpolated learned pos-emb, 2x2 pixel-shuffle merge
  language.py                  # gated attention, qk-norm, centered RMSNorm,
                               #   logit soft-cap, sliding/full attention
  muse_glimmer.py              # top-level: encode image -> adapter -> projection ->
                               #   splice at the image token
  processing_muse_glimmer.py   # image processor + chat processor (smart-resize,
                               #   temporal patchify, <|patch|> expansion)
  __init__.py
```

## Install

Drop the package into your mlx-vlm install so `mlx_vlm.load` can find it:

```bash
pip install mlx-vlm
python - <<'PY'
import os, shutil, mlx_vlm
dst = os.path.join(os.path.dirname(mlx_vlm.__file__), "models", "muse_glimmer")
shutil.rmtree(dst, ignore_errors=True)
shutil.copytree("muse_glimmer", dst)
print("installed ->", dst)
PY
```

## Usage

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template

model, processor = load("mlx-community/Muse-Glimmer-30B-bf16")   # or any local dir
prompt = apply_chat_template(processor, model.config,
                             "What is in this image?", num_images=1)
print(generate(model, processor, prompt, image=["photo.jpg"], max_tokens=128))
```

Text-only prompts work too (`num_images=0`, no `image=` arg).

## Architecture notes

- **Vision:** patch embed (3·14·14·2 = 1176 → 1536) + bilinearly-interpolated 32×32
  learned position table, 50 pre-norm ViT blocks with 2D rotary embeddings and a
  window/full attention schedule, `ln_post`, then a 2×2 spatial (pixel-shuffle) merge
  feeding a 2-layer GELU adapter and a linear projection into the text hidden size.
  Image features are RMSNorm'd and scattered into the token stream at the image token.
- **Language:** grouped-query attention with a per-head RMSNorm and a learned output
  gate, centered RMSNorm (effective scale `1 + w`), per-layer RoPE θ with a
  sliding/full attention schedule, `output_multiplier`, and `tanh` logit soft-capping.

## Compatibility

- Loads Meta's official Muse-Glimmer weights and any derived checkpoint unchanged.
- Requires the full multimodal checkpoint (vision + language tensors). A language-only
  checkpoint loads but has no vision tower to run.

## License

MIT. Model weights are covered by their own licenses; this repo is code only.

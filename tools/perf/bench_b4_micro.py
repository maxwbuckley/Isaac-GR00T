"""Microbenchmark for B4's inference-time text caching (fix/b4-processing-cache).

Isolates what the cache actually saves per control step, without the ~8 ms of
E2E noise that swamps the effect in benchmark_inference.py.

B4's cache-hit path (from _tokenize_vlm_inputs) is:
    image_processor(images)            <- always runs, content changes
    + clone(cached input_ids/attn)     <- cheap
versus the uncached path:
    vlm_processor(text=..., images=...)   <- image features AND tokenization

So the per-step saving is:  full_call - (image_only + clone)

Also times the two smaller caches (language normalization, chat template).
Run under the repo venv with HF_TOKEN set.
"""

import statistics as stats
import time

import numpy as np
import torch
from transformers import AutoProcessor

import gr00t.model  # noqa: F401  (registers Gr00tN1d7Processor)


CKPT = "checkpoints/GR00T-N1.7-LIBERO/libero_10"
N_IMAGES = 2  # libero_sim: image + wrist_image
H = W = 256
ITERS = 300
WARMUP = 30


def timeit(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    samples.sort()
    return {
        "median": stats.median(samples),
        "mean": stats.mean(samples),
        "sd": stats.stdev(samples),
        "p90": samples[int(0.9 * len(samples))],
    }


def fmt(name, r):
    print(f"  {name:<44} {r['median']:7.3f} ms   mean {r['mean']:7.3f} +/- {r['sd']:5.3f}")


def main():
    proc = AutoProcessor.from_pretrained(CKPT)
    proc.eval()
    vlm = proc.collator.processor
    tok = vlm.tokenizer

    rng = np.random.default_rng(0)
    images = [rng.integers(0, 255, (H, W, 3), dtype=np.uint8) for _ in range(N_IMAGES)]
    frames = [torch.as_tensor(im) for im in images]
    instruction = "pick up the black bowl between the plate and the ramekin and place it on the plate"

    # --- chat template, as the processor builds it -------------------------
    conversation = [
        {
            "role": "user",
            "content": [
                *[{"type": "image", "image": f} for f in frames],
                {"type": "text", "text": instruction},
            ],
        }
    ]
    texts = [vlm.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)]

    print(f"\nB4 microbenchmark - {N_IMAGES} x {H}x{W} images, {ITERS} iters after {WARMUP} warmup")
    print(f"checkpoint: {CKPT}\n")

    # --- the three cached artifacts ---------------------------------------
    print("Cache 1: language normalization (cached on raw instruction)")
    r_lang = timeit(lambda: proc._formalize_language(instruction))
    fmt("_formalize_language()", r_lang)

    print("\nCache 2: chat-template render (cached on instruction + image geometry)")
    r_tmpl = timeit(
        lambda: vlm.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    )
    fmt("apply_chat_template()", r_tmpl)

    print("\nCache 3: VLM tokenization - the dominant term")
    r_full = timeit(lambda: vlm(text=texts, images=images, return_tensors="pt", padding=True))
    fmt("UNCACHED: vlm_processor(text, images)", r_full)

    r_img = timeit(lambda: vlm.image_processor(images=images, return_tensors="pt"))
    fmt("CACHE HIT part a: image_processor(images)", r_img)

    cached = vlm(text=texts, images=images, return_tensors="pt", padding=True)
    img_keys = set(vlm.image_processor(images=images, return_tensors="pt").keys())
    text_side = {k: v.clone() for k, v in cached.items() if k not in img_keys}
    r_clone = timeit(lambda: {k: v.clone() for k, v in text_side.items()})
    fmt("CACHE HIT part b: clone(text tensors)", r_clone)

    hit_cost = r_img["median"] + r_clone["median"]
    saving_tok = r_full["median"] - hit_cost
    total_saving = saving_tok + r_lang["median"] + r_tmpl["median"]

    print("\n" + "=" * 74)
    print("PER-STEP SAVING (medians)")
    print("=" * 74)
    print(f"  tokenization: {r_full['median']:.3f} (uncached) - {hit_cost:.3f} (hit) "
          f"= {saving_tok:+.3f} ms")
    print(f"  + language normalization                        = {r_lang['median']:+.3f} ms")
    print(f"  + chat-template render                          = {r_tmpl['median']:+.3f} ms")
    print(f"  {'TOTAL per control step':<47} = {total_saving:+.3f} ms")
    print(f"\n  text-side share of the uncached processor call   = "
          f"{saving_tok / r_full['median'] * 100:.1f}%")
    print(f"  cached text tensors: {sorted(text_side)}")
    print(f"  image tensors (always recomputed): {sorted(img_keys)}")


if __name__ == "__main__":
    main()

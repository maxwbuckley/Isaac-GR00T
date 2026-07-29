"""In-situ A/B of B4's text cache: real captured inputs, cache=None vs cache.

Rather than guessing what the pipeline feeds the VLM processor, this hooks
_tokenize_vlm_inputs during a real Gr00tN1d7Processor call on a real dataset
step, captures the exact (texts, images) arguments, then benchmarks the very
function B4 introduces both ways.

No GPU work and no model weights involved, so variance is far lower than the
E2E harness. Reports percentiles because earlier runs looked bimodal.
"""

import statistics as stats
import time

import numpy as np
import torch
from transformers import AutoProcessor

import gr00t.model  # noqa: F401
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7 as P


CKPT = "checkpoints/GR00T-N1.7-LIBERO/libero_10"
DATASET = "demo_data/libero_demo"
ITERS = 500
WARMUP = 50


def percentiles(samples):
    s = sorted(samples)
    n = len(s)
    return {
        "min": s[0],
        "p10": s[int(0.10 * n)],
        "median": stats.median(s),
        "p90": s[int(0.90 * n)],
        "max": s[-1],
        "mean": stats.mean(s),
        "sd": stats.stdev(s),
    }


def timeit(fn, iters=ITERS, warmup=WARMUP):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return percentiles(out)


def show(name, r):
    print(
        f"  {name:<26} median {r['median']:6.3f} | p10 {r['p10']:6.3f} p90 {r['p90']:6.3f} "
        f"| mean {r['mean']:6.3f} +/- {r['sd']:5.3f} | min {r['min']:6.3f} max {r['max']:7.3f}"
    )


def main():
    proc = AutoProcessor.from_pretrained(CKPT)
    proc.eval()

    # ---- capture the real arguments B4's function receives ----------------
    captured = {}
    original = P._tokenize_vlm_inputs

    def spy(vlm_processor, texts, images, cache):
        if not captured:
            captured["vlm"] = vlm_processor
            captured["texts"] = texts
            captured["images"] = images
        return original(vlm_processor, texts, images, cache)

    P._tokenize_vlm_inputs = spy
    try:
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.data.types import MessageType, VLAStepData

        tag = EmbodimentTag.LIBERO_PANDA
        modality = {
            k: v for k, v in proc.get_modality_configs()[tag.value].items() if k != "rl_info"
        }
        loader = LeRobotEpisodeLoader(dataset_path=DATASET, modality_configs=modality)
        step_data = extract_step_data(loader[0], 0, modality_configs=modality, embodiment_tag=tag)
        lang_key = modality["language"].modality_keys[0]
        vla = VLAStepData(
            images={k: np.stack(step_data.images[k]) for k in step_data.images},
            states={k: np.asarray(v) for k, v in step_data.states.items()},
            actions={},
            text=step_data.text,
            embodiment=tag,
        )
        # Tokenization happens in the COLLATOR, not in processor.__call__:
        # in eval mode _get_vlm_inputs returns early via _build_vlm_content,
        # and the collator is where vlm_tokenize_cache lives.
        processed = proc([{"type": MessageType.EPISODE_STEP.value, "content": vla}])
        proc.collator([processed])
    except Exception as exc:  # capture may succeed even if the full call errors later
        if not captured:
            raise
        print(f"(note: processor call raised after capture: {type(exc).__name__}: {exc})")
    finally:
        P._tokenize_vlm_inputs = original

    vlm = captured["vlm"]
    texts = captured["texts"]
    images = captured["images"]

    print(f"\nB4 in-situ A/B - {ITERS} iters after {WARMUP} warmup")
    print(f"captured images: {type(images).__name__} of {len(images)} x "
          f"{type(images[0]).__name__}"
          f"{tuple(images[0].shape) if hasattr(images[0], 'shape') else ''}"
          f" dtype={getattr(images[0], 'dtype', '?')}")
    print(f"captured texts : {len(texts)} string(s), {len(texts[0])} chars\n")

    # ---- the actual A/B --------------------------------------------------
    r_uncached = timeit(lambda: P._tokenize_vlm_inputs(vlm, texts, images, None))

    cache = P._LRUCache()
    P._tokenize_vlm_inputs(vlm, texts, images, cache)  # prime
    hits0 = cache.hits
    r_cached = timeit(lambda: P._tokenize_vlm_inputs(vlm, texts, images, cache))
    print(f"cache hits during timed run: {cache.hits - hits0} (misses {cache.misses})")

    print("\nTOKENIZATION PATH")
    show("cache=None (main)", r_uncached)
    show("cache=LRU (B4, all hits)", r_cached)

    d_med = r_uncached["median"] - r_cached["median"]
    d_p90 = r_uncached["p90"] - r_cached["p90"]
    print(f"\n  saving at median: {d_med:+.3f} ms  ({d_med / r_uncached['median'] * 100:+.1f}%)")
    print(f"  saving at p90   : {d_p90:+.3f} ms  ({d_p90 / r_uncached['p90'] * 100:+.1f}%)")

    # ---- bitwise parity check -------------------------------------------
    a = P._tokenize_vlm_inputs(vlm, texts, images, None)
    b = P._tokenize_vlm_inputs(vlm, texts, images, cache)
    same = sorted(a.keys()) == sorted(b.keys()) and all(
        torch.equal(a[k], b[k]) for k in a if isinstance(a[k], torch.Tensor)
    )
    print(f"\n  bitwise identical cached vs uncached outputs: {same} "
          f"(keys: {sorted(a.keys())})")


if __name__ == "__main__":
    main()

"""Generate an anime head reference sheet with the local ComfyUI (Anima). Settings follow ComfyUI's
"Text to Image (Anima Base v1.0)" template: qwen_3_06b_base text encoder, qwen_image_vae, 30 steps, cfg 4, euler/simple.

usage: python gen.py <out dir> [seed ...]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

URL = "http://127.0.0.1:8188"
W, H = 1536, 768
PROMPT = ("anime girl head character reference sheet, two views side by side, left: front view, right: exact side profile view "
          "facing right, same character, adult woman, neutral expression, mouth closed, looking straight, hair tied back in a "
          "low ponytail, forehead visible, ears visible, head and neck only, orthographic, clean lineart, flat cel shading, "
          "plain white background, model sheet, turnaround, symmetrical front view")
NEGATIVE = ("worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts, sepia, three-quarter view, "
            "tilted head, smile, open mouth, hands, body, text, watermark, multiple characters, bangs covering eyes")


def graph(seed):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "anima_aestheticV11.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_06b_base.safetensors", "type": "stable_diffusion", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "qwen_image_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyLatentImage", "inputs": {"width": W, "height": H, "batch_size": 1}},
        "7": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
              "seed": seed, "steps": 30, "cfg": 4.0, "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "formcode/anime_ref"}},
    }


def call(path, data=None):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return r.read()


out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for seed in [int(s) for s in sys.argv[2:]] or [1, 2, 3, 4]:
    pid = json.loads(call("/prompt", {"prompt": graph(seed)}))["prompt_id"]
    t0 = time.time()
    while True:
        h = json.loads(call(f"/history/{pid}"))
        if pid in h:
            st = h[pid].get("status", {})
            if st.get("status_str") == "error":
                sys.exit(f"seed {seed}: ComfyUI error {json.dumps(st)[:800]}")
            im = h[pid]["outputs"]["9"]["images"][0]
            break
        time.sleep(1)
    (out / f"ref_seed{seed}.png").write_bytes(call(f"/view?filename={im['filename']}&subfolder={im['subfolder']}&type={im['type']}"))
    print(f"seed {seed}: {time.time() - t0:.1f}s", flush=True)

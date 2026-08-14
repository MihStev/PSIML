#!/usr/bin/env python
"""Interactive web demo: press a direction, the model generates the next frames.

Flask + a single page. The model is loaded ONCE and stays resident, so a press
costs only sampling -- with the 4-step schedule that is about a second for a
16-frame block, which is faster than the 1.6 s of video it produces.

Two modes, and the distinction is the honest part:

  ANCHORED (default) -- every press starts again from the episode's REAL context.
    This is the same regime as every number we report, and it does not degrade.

  CHAIN -- the generated block becomes the context for the next press, i.e. a
    true free rollout. We measured that control collapses after ONE self-generated
    block (96.9% -> 53.1%), so this mode is expected to fall apart within a few
    presses. It is exposed on purpose: the demo should be able to show the
    limitation, not hide it.

Run:
    python web_demo.py --port 8080 [--n_steps 4 --dmd_schedule]
then forward the port and open http://localhost:8080
"""
import argparse
import base64
import io
import os
import sys
import threading
import time

os.environ.setdefault("USER", "mls10")
os.environ.setdefault("LOGNAME", "mls10")
os.environ.setdefault("HOME", "/home/mls10")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, "/home/mls10/minWM-dawidzard/Wan21")
sys.path.insert(0, "/home/mls10/minWM-dawidzard/shared")
sys.path.insert(0, "/home/mls10/minWM-dawidzard/lora_action")
os.chdir("/home/mls10/minWM-dawidzard")

import lmdb                                     # noqa: E402
import numpy as np                              # noqa: E402
import torch                                    # noqa: E402
from flask import Flask, jsonify, request       # noqa: E402
from omegaconf import OmegaConf                 # noqa: E402
from PIL import Image                           # noqa: E402

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb   # noqa: E402
from train_lora_action import ActionEncoderV2                                   # noqa: E402

D = 0.07
DIRS = {"up": (0.0, -D), "down": (0.0, +D), "right": (-D, 0.0), "left": (+D, 0.0), "still": (0.0, 0.0)}

app = Flask(__name__)
STATE = {}
LOCK = threading.Lock()

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Action-Conditioned World Model</title>
<style>
:root{--bg:#0f1115;--fg:#e8e8ea;--dim:#8b8f9a;--acc:#4c8dff;--warn:#ff9f43}
*{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--fg);
  font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  display:flex;flex-direction:column;align-items:center;padding:24px}
h1{font-size:19px;font-weight:600;margin:0 0 2px}
.sub{color:var(--dim);font-size:13px;margin-bottom:18px}
canvas{image-rendering:pixelated;border-radius:10px;background:#000;
  width:min(70vw,512px);height:min(70vw,512px)}
.pad{display:grid;grid-template-columns:repeat(3,64px);grid-template-rows:repeat(3,64px);
  gap:8px;margin:20px 0 8px}
button{background:#1b1f28;color:var(--fg);border:1px solid #2c313d;border-radius:8px;
  font-size:20px;cursor:pointer;transition:.12s}
button:hover:not(:disabled){background:var(--acc);border-color:var(--acc)}
button:disabled{opacity:.35;cursor:wait}
.row{display:flex;gap:10px;align-items:center;margin-top:6px}
.row button{font-size:13px;padding:8px 14px;width:auto;height:auto}
#status{color:var(--dim);font-size:13px;height:20px;margin-top:10px;font-variant-numeric:tabular-nums}
#mode{color:var(--acc);font-weight:600}
.warn{color:var(--warn)}
.log{color:var(--dim);font-size:12px;margin-top:14px;max-width:520px;text-align:center}
</style></head><body>
<h1>Action-Conditioned Video World Model</h1>
<div class="sub">BAIR robot pushing &middot; Wan2.1-T2V-1.3B + LoRA &middot; 64&times;64</div>
<canvas id="c" width="64" height="64"></canvas>
<div class="pad">
  <div></div><button data-a="up">&#9650;</button><div></div>
  <button data-a="left">&#9664;</button><button data-a="still">&#9679;</button><button data-a="right">&#9654;</button>
  <div></div><button data-a="down">&#9660;</button><div></div>
</div>
<div class="row">
  <button id="mode-btn">mode: <span id="mode">ANCHORED</span></button>
  <button id="reset">new scene</button>
</div>
<div id="status">ready &middot; arrow keys work too</div>
<div class="log" id="note">ANCHORED: every press restarts from the episode's real context &mdash; the regime all
our reported numbers use. Switch to CHAIN to feed the model its own output: control is measured to
collapse after one self-generated block.</div>
<script>
const c=document.getElementById('c'),x=c.getContext('2d');
let busy=false,chain=false;
function paint(frames,i){ if(i>=frames.length)return;
  const im=new Image(); im.onload=()=>{x.drawImage(im,0,0);setTimeout(()=>paint(frames,i+1),90)};
  im.src='data:image/png;base64,'+frames[i]; }
async function go(a){ if(busy)return; busy=true;
  document.querySelectorAll('button').forEach(b=>b.disabled=true);
  const t0=performance.now();
  document.getElementById('status').textContent='generating…';
  try{
    const r=await fetch('/step',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({action:a,chain:chain})});
    const d=await r.json(); paint(d.frames,0);
    const ms=Math.round(performance.now()-t0);
    document.getElementById('status').innerHTML=
      `${a} &middot; ${d.frames.length} frames &middot; ${ms} ms &middot; block ${d.block}`
      + (chain&&d.block>1?' <span class="warn">(degrading)</span>':'');
  }catch(e){ document.getElementById('status').textContent='error: '+e; }
  document.querySelectorAll('button').forEach(b=>b.disabled=false); busy=false; }
document.querySelectorAll('.pad button').forEach(b=>b.onclick=()=>go(b.dataset.a));
document.getElementById('mode-btn').onclick=async()=>{ chain=!chain;
  document.getElementById('mode').textContent=chain?'CHAIN':'ANCHORED';
  await fetch('/reset',{method:'POST'});
  document.getElementById('status').textContent='mode switched, context reset'; };
document.getElementById('reset').onclick=async()=>{ await fetch('/reset',{method:'POST',
    headers:{'Content-Type':'application/json'},body:JSON.stringify({random:true})});
  document.getElementById('status').textContent='new scene loaded'; go('still'); };
addEventListener('keydown',e=>{const m={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',
  ArrowRight:'right',' ':'still'}; if(m[e.key]){e.preventDefault();go(m[e.key]);}});
go('still');
</script></body></html>"""


@app.route("/")
def index():
    return PAGE


def decode_png_list(frames_uint8):
    out = []
    for f in frames_uint8:
        buf = io.BytesIO()
        Image.fromarray(f).save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode())
    return out


@app.route("/reset", methods=["POST"])
def reset():
    body = request.get_json(silent=True) or {}
    with LOCK:
        if body.get("random"):
            STATE["idx"] = int(np.random.randint(0, STATE["n_scenes"]))
        STATE["latent"] = load_latent(STATE["idx"])
        STATE["block"] = 0
    return jsonify(ok=True, idx=STATE["idx"])


@app.route("/step", methods=["POST"])
def step():
    if not STATE:
        # preview mode: a checkerboard so the plumbing is visibly alive
        g = (np.indices((64, 64)).sum(0) % 16 < 8).astype(np.uint8) * 90 + 40
        f = np.stack([g, g, g], -1)
        return jsonify(frames=decode_png_list([f] * 16), block=0, ms=0)
    body = request.get_json(force=True)
    name = body.get("action", "still")
    chain = bool(body.get("chain", False))
    dx, dy = DIRS.get(name, (0.0, 0.0))
    t0 = time.time()
    with LOCK:
        ctx = STATE["latent"]
        gen = generate_block(ctx, dx, dy)
        frames = decode_frames(gen)
        if chain:
            STATE["latent"] = gen                 # its own output becomes the next context
            STATE["block"] += 1
        else:
            STATE["block"] = 1
        blk = STATE["block"]
    n_ctx_px = 1 + 4 * (STATE["n_ctx"] - 1)
    return jsonify(frames=decode_png_list(frames[n_ctx_px:]), block=blk,
                   ms=int((time.time() - t0) * 1000))


# ---------------------------------------------------------------- model side
def load_latent(idx):
    lat = retrieve_row_from_lmdb(STATE["env"], "latents", np.float16, idx,
                                 shape=STATE["lat_shape"][1:]).astype(np.float32)
    return torch.from_numpy(lat).to(device=STATE["device"], dtype=torch.bfloat16).unsqueeze(0)


def generate_block(real_latent, dx, dy):
    m, dev = STATE["model"], STATE["device"]
    F, n_ctx = STATE["NUM_FRAMES"], STATE["n_ctx"]
    apl = np.zeros((1, F, 16), dtype=np.float32)
    for i in range(1, F):
        apl[0, i] = np.tile([dx, dy, 0.5, 0.25], 4)
    a = torch.tensor(apl, device=dev)
    an = (a - STATE["a_mean"]) / STATE["a_std"]
    an[:, 0, :] = 0.0
    emb = STATE["action_encoder"](an.to(torch.bfloat16))

    s = real_latent.clone()
    noise = torch.randn_like(real_latent)
    s[:, n_ctx:] = noise[:, n_ctx:]
    for i, t_val in enumerate(STATE["schedule"]):
        ts = torch.zeros((1, F), device=dev, dtype=torch.bfloat16)
        ts[:, n_ctx:] = t_val.item()
        s[:, :n_ctx] = real_latent[:, :n_ctx]
        _, x0 = m.generator(noisy_image_or_video=s, conditional_dict=STATE["cond"], timestep=ts,
                            clean_x=real_latent, aug_t=None, viewmats=STATE["vm"], Ks=STATE["ks"],
                            action_embed=emb)
        x0 = x0.float().clamp(-6, 6)
        if i == len(STATE["schedule"]) - 1:
            s[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
        else:
            sn = float(m.scheduler.sigmas[i + 1])
            s[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
    s[:, :n_ctx] = real_latent[:, :n_ctx]
    return s


def decode_frames(lat):
    x = STATE["model"].vae.decode_to_pixel(lat.to(STATE["device"]))
    return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()


def boot(args):
    device = torch.device("cuda")
    torch.set_grad_enabled(False)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt["args"]["rank"]
    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    config = OmegaConf.merge(OmegaConf.load("Wan21/configs/default_config.yaml"), config)

    from model import CameraCausalDiffusion
    print("=== loading model (once) ===", flush=True)
    model = CameraCausalDiffusion(config, device=device)
    base = torch.load(args.base_checkpoint, map_location="cpu")
    gen_sd = base.get("generator_ema", base.get("generator"))
    try:
        model.generator.load_state_dict(gen_sd)
    except RuntimeError:
        model.generator.load_state_dict(
            {k.replace("model._fsdp_wrapped_module.", "model.", 1): v for k, v in gen_sd.items()},
            strict=False)
    model.generator.to(device=device, dtype=torch.bfloat16)
    model.text_encoder.to(device=device, dtype=torch.bfloat16)
    model.vae.to(device=device, dtype=torch.bfloat16)

    from peft import LoraConfig, inject_adapter_in_model
    inject_adapter_in_model(
        LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q", "k", "v", "ffn.0", "ffn.2"]),
        model.generator.model)
    model.generator.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    enc = ActionEncoderV2(out_dim=1536).to(device=device, dtype=torch.bfloat16)
    enc.load_state_dict(ckpt["action_encoder_state_dict"])

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    F = lat_shape[1]

    if args.dmd_schedule:
        model.scheduler.set_timesteps(1000)
        full = torch.cat((model.scheduler.timesteps.cpu(), torch.tensor([0.0])))
        sch = full[[1000 - i for i in (1000, 750, 500, 250)]].to(device)
        model.scheduler.sigmas = (sch.cpu() / model.scheduler.num_train_timesteps)
        model.scheduler.timesteps = sch.cpu()
    else:
        model.scheduler.set_timesteps(args.n_steps)
        sch = model.scheduler.timesteps.to(device)

    STATE.update(
        model=model, action_encoder=enc, device=device, env=env, lat_shape=lat_shape,
        NUM_FRAMES=F, n_ctx=config.num_frame_per_block, schedule=sch,
        a_mean=ckpt["action_mean"].to(device), a_std=ckpt["action_std"].to(device),
        cond=model.text_encoder(text_prompts=["a robot arm pushing objects on a table"]),
        vm=torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(1, F, 1, 1),
        ks=torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device,
                        dtype=torch.bfloat16).view(1, 1, 3, 3).repeat(1, F, 1, 1),
        n_scenes=lat_shape[0], idx=args.context_idx, block=0,
    )
    STATE["latent"] = load_latent(args.context_idx)
    print(f"=== ready: {len(sch)} denoising steps, {lat_shape[0]} scenes ===", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--base_checkpoint",
                    default="/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test")
    p.add_argument("--context_idx", type=int, default=3)
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--dmd_schedule", action="store_true")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--no_model", action="store_true",
                    help="serve the page WITHOUT loading the model -- for looking at the UI while\n                          the GPU is busy. Presses return a placeholder, not a generation.")
    args = p.parse_args()
    if args.no_model:
        print("=== PREVIEW MODE: no model, presses return a placeholder ===", flush=True)
    else:
        boot(args)
    app.run(host="0.0.0.0", port=args.port, threaded=False)

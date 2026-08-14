#!/usr/bin/env python
"""Pre-render the interactive demo into ONE self-contained HTML file.

Why static instead of a live server: the container's Jupyter is PID 1 on port 6006,
so no other port is reachable from outside and the server-proxy extension cannot be
loaded without restarting the container's entrypoint. A page with the frames baked in
needs no port, no tunnel and no GPU -- it opens anywhere, including from a phone during
the talk, and it cannot fail live. Danilo's "prerender good demos!!!" points the same way.

What is pre-rendered, per scene:
  - ANCHORED: one block per action (up/down/left/right/still), each from the SAME real
    context and the SAME noise, so the difference between buttons is the action alone.
  - CHAIN: a free rollout where each generated block becomes the next context. Included
    on purpose -- control is measured to collapse after one self-generated block, and the
    demo should be able to show that, not hide it.

Frames are embedded as base64 PNG. At 64x64 a frame is ~1.5 KB, so a few hundred frames
stay well inside any page budget.
"""
import argparse
import base64
import io
import json
import os
import sys

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
from omegaconf import OmegaConf                 # noqa: E402
from PIL import Image                           # noqa: E402

from wan_utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb   # noqa: E402
from train_lora_action import ActionEncoderV2                                   # noqa: E402

D = 0.07
DIRS = {"up": (0.0, -D), "down": (0.0, +D), "right": (-D, 0.0), "left": (+D, 0.0),
        "still": (0.0, 0.0)}
ORDER = ["up", "down", "left", "right", "still"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/home/mls10/checkpoints/bair_lora_big/step_8000.pt")
    p.add_argument("--base_checkpoint",
                    default="/tmp/local_ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt")
    p.add_argument("--lmdb_path", default="/tmp/bair_lmdb/test")
    p.add_argument("--scenes", type=int, nargs="+", default=[3, 51, 108, 204])
    p.add_argument("--chain_len", type=int, default=4)
    p.add_argument("--n_steps", type=int, default=4)
    p.add_argument("--dmd_schedule", action="store_true", default=True)
    p.add_argument("--out", default="/home/mls10/logs/demo/index.html")
    return p.parse_args()


def png_b64(frame):
    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device("cuda")
    torch.set_grad_enabled(False)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    rank = ckpt["args"]["rank"]
    config = OmegaConf.load("Wan21/configs/ar_camera_tf.yaml")
    config = OmegaConf.merge(OmegaConf.load("Wan21/configs/default_config.yaml"), config)

    from model import CameraCausalDiffusion                     # noqa: E402
    print("=== loading model ===", flush=True)
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

    from peft import LoraConfig, inject_adapter_in_model        # noqa: E402
    inject_adapter_in_model(
        LoraConfig(r=rank, lora_alpha=rank * 2, target_modules=["q", "k", "v", "ffn.0", "ffn.2"]),
        model.generator.model)
    model.generator.model.load_state_dict(ckpt["lora_state_dict"], strict=False)
    enc = ActionEncoderV2(out_dim=1536).to(device=device, dtype=torch.bfloat16)
    enc.load_state_dict(ckpt["action_encoder_state_dict"])
    a_mean, a_std = ckpt["action_mean"].to(device), ckpt["action_std"].to(device)
    cond = model.text_encoder(text_prompts=["a robot arm pushing objects on a table"])

    env = lmdb.open(args.lmdb_path, readonly=True, lock=False)
    lat_shape = get_array_shape_from_lmdb(env, "latents")
    F, n_ctx = lat_shape[1], config.num_frame_per_block
    vm = torch.eye(4, device=device, dtype=torch.bfloat16).view(1, 1, 4, 4).repeat(1, F, 1, 1)
    ks = torch.tensor([[0.5, 0, 0.5], [0, 0.5, 0.5], [0, 0, 1]], device=device,
                      dtype=torch.bfloat16).view(1, 1, 3, 3).repeat(1, F, 1, 1)

    if args.dmd_schedule:
        model.scheduler.set_timesteps(1000)
        full = torch.cat((model.scheduler.timesteps.cpu(), torch.tensor([0.0])))
        sch = full[[1000 - i for i in (1000, 750, 500, 250)]].to(device)
        model.scheduler.sigmas = (sch.cpu() / model.scheduler.num_train_timesteps)
        model.scheduler.timesteps = sch.cpu()
    else:
        model.scheduler.set_timesteps(args.n_steps)
        sch = model.scheduler.timesteps.to(device)
    print(f"=== {len(sch)} denoising steps ===", flush=True)

    def gen(ctx, dx, dy, noise):
        apl = np.zeros((1, F, 16), dtype=np.float32)
        for i in range(1, F):
            apl[0, i] = np.tile([dx, dy, 0.5, 0.25], 4)
        an = (torch.tensor(apl, device=device) - a_mean) / a_std
        an[:, 0, :] = 0.0
        emb = enc(an.to(torch.bfloat16))
        s = ctx.clone()
        s[:, n_ctx:] = noise[:, n_ctx:]
        for i, t_val in enumerate(sch):
            ts = torch.zeros((1, F), device=device, dtype=torch.bfloat16)
            ts[:, n_ctx:] = t_val.item()
            s[:, :n_ctx] = ctx[:, :n_ctx]
            _, x0 = model.generator(noisy_image_or_video=s, conditional_dict=cond, timestep=ts,
                                    clean_x=ctx, aug_t=None, viewmats=vm, Ks=ks, action_embed=emb)
            x0 = x0.float().clamp(-6, 6)
            if i == len(sch) - 1:
                s[:, n_ctx:] = x0[:, n_ctx:].to(torch.bfloat16)
            else:
                sn = float(model.scheduler.sigmas[i + 1])
                s[:, n_ctx:] = ((1 - sn) * x0 + sn * torch.randn_like(x0))[:, n_ctx:].to(torch.bfloat16)
        s[:, :n_ctx] = ctx[:, :n_ctx]
        return s

    def dec(lat):
        x = model.vae.decode_to_pixel(lat.to(device))
        return ((x.float().clamp(-1, 1) + 1) / 2 * 255).byte()[0].permute(0, 2, 3, 1).cpu().numpy()

    n_ctx_px = 1 + 4 * (n_ctx - 1)
    data = {}
    for idx in args.scenes:
        lat = torch.from_numpy(
            retrieve_row_from_lmdb(env, "latents", np.float16, idx, shape=lat_shape[1:])
            .astype(np.float32)).to(device=device, dtype=torch.bfloat16).unsqueeze(0)
        noise = torch.randn_like(lat)                     # SAME noise for every action
        entry = {"context": [png_b64(f) for f in dec(lat)[:n_ctx_px]], "anchored": {}, "chain": []}
        for name in ORDER:
            dx, dy = DIRS[name]
            entry["anchored"][name] = [png_b64(f) for f in dec(gen(lat, dx, dy, noise))[n_ctx_px:]]
            print(f"  scene {idx}: {name}", flush=True)
        cur = lat
        for k in range(args.chain_len):
            name = ORDER[k % 4]
            dx, dy = DIRS[name]
            cur = gen(cur, dx, dy, torch.randn_like(cur))
            entry["chain"].append({"action": name,
                                   "frames": [png_b64(f) for f in dec(cur)[n_ctx_px:]]})
            print(f"  scene {idx}: chain {k+1}/{args.chain_len} ({name})", flush=True)
        data[str(idx)] = entry

    html = PAGE.replace("__DATA__", json.dumps(data)).replace("__STEPS__", str(len(sch)))
    with open(args.out, "w") as f:
        f.write(html)
    print(f"  -> {args.out}  ({os.path.getsize(args.out)/1048576:.2f} MB)", flush=True)


PAGE = """<title>Action-Conditioned World Model</title>
<style>
:root{--bg:#fbfbfd;--fg:#16181d;--dim:#6b7280;--line:#e3e5ea;--acc:#2563eb;--warn:#c2410c;--card:#fff}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --bg:#0f1115;--fg:#e8e8ea;--dim:#9096a1;--line:#252932;--acc:#5b9dff;--warn:#fb923c;--card:#161922}}
:root[data-theme="dark"]{--bg:#0f1115;--fg:#e8e8ea;--dim:#9096a1;--line:#252932;--acc:#5b9dff;--warn:#fb923c;--card:#161922}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);padding:32px 20px 56px;
 font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,sans-serif;
 display:flex;flex-direction:column;align-items:center}
h1{font-size:22px;font-weight:650;margin:0 0 4px;letter-spacing:-.01em;text-align:center}
.sub{color:var(--dim);font-size:13.5px;margin-bottom:26px;text-align:center;max-width:560px}
.stage{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px;
 display:flex;flex-direction:column;align-items:center;box-shadow:0 1px 3px rgba(0,0,0,.05)}
canvas{image-rendering:pixelated;border-radius:10px;background:#000;
 width:min(78vw,384px);height:min(78vw,384px);display:block}
.pad{display:grid;grid-template-columns:repeat(3,58px);grid-template-rows:repeat(3,58px);
 gap:8px;margin:20px 0 4px}
button{background:transparent;color:var(--fg);border:1px solid var(--line);border-radius:10px;
 font-size:17px;cursor:pointer;transition:.13s;font-family:inherit}
button:hover{background:var(--acc);border-color:var(--acc);color:#fff}
button.on{background:var(--acc);border-color:var(--acc);color:#fff}
.row{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:12px}
.row button{font-size:12.5px;padding:7px 13px}
#status{color:var(--dim);font-size:12.5px;margin-top:14px;height:18px;font-variant-numeric:tabular-nums}
.warn{color:var(--warn);font-weight:600}
.note{color:var(--dim);font-size:12.5px;margin-top:22px;max-width:560px;text-align:center;
 border-top:1px solid var(--line);padding-top:16px}
kbd{border:1px solid var(--line);border-radius:4px;padding:1px 5px;font-size:11px;font-family:inherit}
</style>
<h1>Action-Conditioned Video World Model</h1>
<div class="sub">BAIR robot pushing &middot; Wan2.1-T2V-1.3B + LoRA &middot; 64&times;64 &middot; __STEPS__ denoising steps<br>
Press a direction: the arm moves the way it was told. Frames are model output, pre-rendered.</div>
<div class="stage">
  <canvas id="c" width="64" height="64"></canvas>
  <div class="pad">
    <div></div><button data-a="up">&#9650;</button><div></div>
    <button data-a="left">&#9664;</button><button data-a="still">&#9679;</button><button data-a="right">&#9654;</button>
    <div></div><button data-a="down">&#9660;</button><div></div>
  </div>
  <div class="row" id="scenes"></div>
  <div class="row"><button id="chain">play free rollout &rarr;</button></div>
  <div id="status">ready &middot; arrow keys work too</div>
</div>
<div class="note">Every button starts from the same real context with the same noise, so the only thing
that changes is the commanded action. <b>Free rollout</b> feeds each generated block back in as the next
context &mdash; control is measured to collapse after one self-generated block, and that is shown rather
than hidden. <kbd>&larr;</kbd><kbd>&rarr;</kbd><kbd>&uarr;</kbd><kbd>&darr;</kbd> also work.</div>
<script>
const DATA=__DATA__, ids=Object.keys(DATA);
let cur=ids[0], busy=false;
const c=document.getElementById('c'),x=c.getContext('2d'),st=document.getElementById('status');
const cache={};
function img(b64){ if(cache[b64])return cache[b64];
  const i=new Image(); i.src='data:image/png;base64,'+b64; cache[b64]=i; return i; }
Object.values(DATA).forEach(e=>{e.context.forEach(img);
  Object.values(e.anchored).forEach(fr=>fr.forEach(img)); e.chain.forEach(b=>b.frames.forEach(img));});
function draw(b64){ const i=img(b64); if(i.complete)x.drawImage(i,0,0); else i.onload=()=>x.drawImage(i,0,0); }
function play(frames,done){ let k=0; busy=true;
  (function tick(){ if(k>=frames.length){busy=false; if(done)done(); return;}
    draw(frames[k++]); setTimeout(tick,90); })(); }
function showCtx(){ play(DATA[cur].context); }
function act(a){ if(busy)return;
  document.querySelectorAll('.pad button').forEach(b=>b.classList.toggle('on',b.dataset.a===a));
  st.textContent='action: '+a+' \\u00b7 16 frames \\u00b7 scene '+cur;
  play(DATA[cur].context.concat(DATA[cur].anchored[a])); }
document.querySelectorAll('.pad button').forEach(b=>b.onclick=()=>act(b.dataset.a));
const sc=document.getElementById('scenes');
ids.forEach(id=>{const b=document.createElement('button'); b.textContent='scene '+id;
  b.onclick=()=>{cur=id; st.textContent='scene '+id; showCtx();}; sc.appendChild(b);});
document.getElementById('chain').onclick=()=>{ if(busy)return;
  const ch=DATA[cur].chain; let i=0;
  (function nxt(){ if(i>=ch.length){ st.innerHTML='free rollout finished \\u00b7 <span class="warn">quality and control degrade</span>'; return; }
    const b=ch[i]; st.innerHTML='free rollout \\u00b7 block '+(i+1)+'/'+ch.length+' \\u00b7 '+b.action
      +(i>0?' \\u00b7 <span class="warn">self-generated context</span>':'');
    i++; play(b.frames,nxt); })(); };
addEventListener('keydown',e=>{const m={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',ArrowRight:'right',' ':'still'};
  if(m[e.key]){e.preventDefault();act(m[e.key]);}});
showCtx();
</script>"""


if __name__ == "__main__":
    main()

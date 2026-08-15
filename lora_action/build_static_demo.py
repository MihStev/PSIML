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
    p.add_argument("--sr_weights", default=None,
                    help="path to Real-ESRGAN weights; if given, every frame is upscaled 4x\n                          before embedding. ESRGAN and not the diffusion upscaler on purpose:\n                          the diffusion one invents texture independently per frame, so its\n                          fabricated patterns shimmer between frames -- far more visible in\n                          motion than in a still.")
    p.add_argument("--jpeg_quality", type=int, default=0,
                    help="0 = PNG (right for raw 64x64). At 4x, PNG blows the page size up to\n                          ~100 MB, so use JPEG ~85 -- visually indistinguishable on this\n                          content and 5-8x smaller.")
    p.add_argument("--out", default="/home/mls10/logs/demo/index.html")
    return p.parse_args()


SR = {"model": None, "quality": 0}


def png_b64(frame):
    if SR["model"] is not None:
        import torch as _t
        with _t.no_grad():
            x = _t.from_numpy(frame).float().div(255).permute(2, 0, 1).unsqueeze(0).cuda()
            y = SR["model"](x).clamp(0, 1)
        frame = (y[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    buf = io.BytesIO()
    if SR["quality"]:
        Image.fromarray(frame).save(buf, format="JPEG", quality=SR["quality"], optimize=True)
        mime = "jpeg"
    else:
        Image.fromarray(frame).save(buf, format="PNG", optimize=True)
        mime = "png"
    return mime[0] + base64.b64encode(buf.getvalue()).decode()


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

    if args.sr_weights:
        from spandrel import ModelLoader
        SR["model"] = ModelLoader().load_from_file(args.sr_weights).to("cuda").eval()
        print(f"=== Real-ESRGAN x{SR['model'].scale} on every frame ===", flush=True)
    SR["quality"] = args.jpeg_quality

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


PAGE = """<title>Arm Control Panel</title>
<style>
:root{
  --ground:#f6f7f4; --panel:#fdfdfb; --ink:#1b1e21; --dim:#6f7671;
  --line:#dbdfd6; --line2:#eceee8; --accent:#8a5a12; --oxide:#a3401a; --bezel:#c9cdc3;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#131512; --panel:#1a1d19; --ink:#e7e8e3; --dim:#8d938a;
  --line:#2b2f28; --line2:#22261f; --accent:#c8942f; --oxide:#d4703f; --bezel:#2f342c;
}}
:root[data-theme="dark"]{
  --ground:#131512; --panel:#1a1d19; --ink:#e7e8e3; --dim:#8d938a;
  --line:#2b2f28; --line2:#22261f; --accent:#c8942f; --oxide:#d4703f; --bezel:#2f342c;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--ink);
  padding:44px 20px 64px;
  font:15px/1.62 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  display:flex;flex-direction:column;align-items:center;gap:0}
.mono{font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace}
header{max-width:60ch;text-align:center;display:flex;flex-direction:column;gap:9px;margin-bottom:30px}
h1{font-size:26px;font-weight:640;letter-spacing:-.021em;margin:0;text-wrap:balance;line-height:1.2}
.spec{font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11px;letter-spacing:.10em;text-transform:uppercase;color:var(--dim)}
.lede{font-size:14.5px;color:var(--dim);margin:0;text-wrap:balance}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:4px;
  padding:26px 26px 20px;display:flex;flex-direction:column;align-items:center;gap:0}
.screen{padding:9px;background:var(--bezel);border-radius:3px;line-height:0}
canvas{image-rendering:pixelated;display:block;background:#0a0b0a;border-radius:1px;
  width:min(74vw,352px);height:min(74vw,352px)}
.pad{display:grid;grid-template-columns:repeat(3,54px);grid-template-rows:repeat(3,54px);
  gap:6px;margin:22px 0 0}
button{font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  background:transparent;color:var(--ink);border:1px solid var(--line);border-radius:3px;
  font-size:15px;cursor:pointer;transition:background .12s,border-color .12s,color .12s}
button:hover{background:var(--accent);border-color:var(--accent);color:var(--panel)}
button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
button.on{background:var(--accent);border-color:var(--accent);color:var(--panel)}
.strip{display:flex;gap:6px;flex-wrap:wrap;justify-content:center;margin-top:18px;
  padding-top:16px;border-top:1px solid var(--line2);width:100%}
.strip button{font-size:11.5px;letter-spacing:.05em;padding:7px 12px;text-transform:uppercase}
#status{font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  font-size:11.5px;letter-spacing:.045em;color:var(--dim);margin-top:18px;min-height:17px;
  text-align:center;font-variant-numeric:tabular-nums}
.warn{color:var(--oxide)}
.branch{color:var(--accent)}
footer{max-width:60ch;margin-top:30px;padding-top:18px;border-top:1px solid var(--line);
  color:var(--dim);font-size:13px;text-align:center;text-wrap:pretty}
footer b{color:var(--ink);font-weight:600}
kbd{font-family:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  border:1px solid var(--line);border-radius:3px;padding:1px 5px;font-size:11px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
<header>
  <div class="spec">Wan2.1-T2V-1.3B + LoRA &middot; BAIR &middot; 64&times;64 &middot; __STEPS__ steps</div>
  <h1>The arm moves the way it is told</h1>
  <p class="lede">Press a direction. Every button starts from the same real context with the same
  noise, so the only thing that changes is the commanded action.</p>
</header>
<div class="panel">
  <div class="screen"><canvas id="c" width="64" height="64"></canvas></div>
  <div class="pad">
    <div></div><button data-a="up" aria-label="up">&#9650;</button><div></div>
    <button data-a="left" aria-label="left">&#9664;</button>
    <button data-a="still" aria-label="hold">&#9679;</button>
    <button data-a="right" aria-label="right">&#9654;</button>
    <div></div><button data-a="down" aria-label="down">&#9660;</button><div></div>
  </div>
  <div class="strip" id="scenes"></div>
  <div class="strip"><button id="chain">play free rollout</button></div>
  <div id="status">ready</div>
</div>
<footer>Each arrow <b>branches from the same real context</b> &mdash; it is not a continuation, which is
exactly why the differences between them can only come from the action. <b>Free rollout</b> is the one
that continues: each generated block becomes the next context, and control is measured to collapse
after one self-generated block, so it is shown here rather than hidden.
Arrow keys work too: <kbd>&larr;</kbd> <kbd>&uarr;</kbd> <kbd>&darr;</kbd> <kbd>&rarr;</kbd></footer>
<script>
const DATA=__DATA__, ids=Object.keys(DATA);
let cur=ids[0], busy=false;
const c=document.getElementById('c'),x=c.getContext('2d'),st=document.getElementById('status');
const cache={};
function img(b){ if(cache[b])return cache[b];
  const m=b[0]==='j'?'jpeg':'png';
  const i=new Image(); i.src='data:image/'+m+';base64,'+b.slice(1); cache[b]=i; return i; }
Object.values(DATA).forEach(e=>{e.context.forEach(img);
  Object.values(e.anchored).forEach(f=>f.forEach(img)); e.chain.forEach(b=>b.frames.forEach(img));});
function draw(b){ const i=img(b); if(i.complete)x.drawImage(i,0,0); else i.onload=()=>x.drawImage(i,0,0); }
function play(frames,done){ let k=0; busy=true;
  (function tick(){ if(k>=frames.length){busy=false; if(done)done(); return;}
    draw(frames[k++]); setTimeout(tick,90); })(); }
function showCtx(){ play(DATA[cur].context); }
function act(a){ if(busy)return;
  document.querySelectorAll('.pad button').forEach(b=>b.classList.toggle('on',b.dataset.a===a));
  st.innerHTML='action '+a.toUpperCase()+'  \u00b7  16 generated frames  \u00b7  '
    +'<span class="branch">branch from the same context</span>';
  play(DATA[cur].anchored[a]); }
document.querySelectorAll('.pad button').forEach(b=>b.onclick=()=>act(b.dataset.a));
const sc=document.getElementById('scenes');
ids.forEach(id=>{const b=document.createElement('button'); b.textContent='scene '+id;
  b.onclick=()=>{cur=id; document.querySelectorAll('#scenes button').forEach(o=>o.classList.remove('on'));
    b.classList.add('on'); st.textContent='scene '+id+'  \u00b7  real context'; showCtx();};
  sc.appendChild(b);});
sc.firstChild.classList.add('on');
document.getElementById('chain').onclick=()=>{ if(busy)return;
  const ch=DATA[cur].chain; let i=0;
  (function nxt(){ if(i>=ch.length){
      st.innerHTML='free rollout ended  \u00b7  <span class="warn">quality and control degrade</span>'; return; }
    const b=ch[i];
    st.innerHTML='free rollout  \u00b7  block '+(i+1)+'/'+ch.length+'  \u00b7  '+b.action.toUpperCase()
      +(i>0?'  \u00b7  <span class="warn">self-generated context</span>':'');
    i++; play(b.frames,nxt); })(); };
addEventListener('keydown',e=>{const m={ArrowUp:'up',ArrowDown:'down',ArrowLeft:'left',
  ArrowRight:'right',' ':'still'};
  if(m[e.key]){e.preventDefault();act(m[e.key]);}});
showCtx();
</script>"""


if __name__ == "__main__":
    main()

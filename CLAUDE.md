# Projekat

Action-conditioned video world model, baziran na [minWM](https://github.com/shengshu-ai/minWM)
(ovaj repo, kloniran u `/home/mls10/minWM`).

- **Backbone:** Wan2.1-T2V-1.3B (najlakši baseline u repou; alternativa je HunyuanVideo-1.5/8B, ne koristimo je)
- **Dataset:** BAIR robot pushing small (TFDS `bair_robot_pushing_small`, 64x64)
- **Pristup:** uzeti teacher-forcing checkpoint (PRE distilacije, tj. izlaz Phase 2 / Stage 1 pipeline-a
  ispod) i raditi LoRA adaptaciju na BAIR. LoRA sloj posle svakog bloka = kontrola akcijom.
  Referenca za LoRA na WAN: [aigc-apps/VideoX-Fun](https://github.com/aigc-apps/VideoX-Fun)
- **Metrike:** PSNR, SSIM, FID preko `torchmetrics`
- **Rok:** 5-dnevni demo, kratki rollout (8-16 frejmova). Dug rollout nije cilj.

## Pitfalls iz repo skilla (`onboarding-world-model`) — primenjivo na naš LoRA trening

- **bs < 8 nije dovoljno** za kontrolabilnost (razlika od običnog T2V treninga, kontrolabilnost je
  osetljivija na batch size).
- **Kontrolabilnost se ne vidi prvih ~1k koraka** kod 1.3B modela (kod 8B čak >5k) — ne zaključivati
  da LoRA/pristup ne radi na osnovu ranih koraka treninga.
- GT kondicije moraju biti precizne/nedvosmislene — za nas znači da BAIR `action` vektori moraju ući
  u trening bez šuma iz preprocesiranja (TFRecord → bilo koji format za trener).

## Trenutni korak (ažurirano nakon storage sage — pročitaj OVO pre nastavka)

**BAIR dataset: skinut, ekstrakcija OTKAZANA namerno, sad živi na `/lustre/data/mls10/`, NE u
ovom kontejneru.** Detalji:
- TFDS download je prvo pao pri 4% (proxy prekid), pa uspešno resume-ovan preko `wget -c` +
  ručna registracija u TFDS keš (video `finalize_bair_download.py` u `/home/mls10/scripts/` ako
  treba slična tehnika opet). Kompletiran, 100% (32,274,964,480 bajtova).
- **Otkriven kritičan problem usred ekstrakcije:** `/home/mls10` kvota je **40GB** (ne 44TB — to je
  bila veličina CELOG deljenog NFS-a preko `df -h`, ne naša kvota; `quota`/`repquota` komande ne
  postoje ovde da se to programski potvrdi, ali vidljivo je u praksi). Sa BAIR tar-om (30GB) +
  ekstrakcijom (dodatnih ~31GB privremeno) + venv-ovima (11GB) prešli smo na ~100GB, daleko preko
  limita. Ekstrakcija je ubijena, sav BAIR-vezan sadržaj obrisan iz `/home/mls10`.
- **Rešenje:** BAIR `.tar` (i samo `.tar`, neekstrahovan) prebačen preko login node-a
  (`ssh mls10@10.68.6.58`, koji vidi i `/home/mls10` NFS I `/lustre/data/mls10`) preko `rsync` na
  **`/lustre/data/mls10/bair_robot_pushing_small/`**. Ovaj Claude Code kontejner **NE VIDI** tu
  putanju (proverio: `/lustre/data/mls10` ne postoji odavde, ni kao `/data`). Login node je jedini
  most između ova dva prostora za sada.
- `/home/mls10` je sad čist: **16GB** (venv-ovi 11GB + `.local` 4.8GB + repo/logs ~sitno), duboko
  ispod 40GB limita.
- **GOTOVO — BAIR ekstrakcija (tar → TFRecord) završena i trajno smeštena.** Urađeno: (1) `.tar`
  kopiran sa read-only `/data` na `/tmp` (privatan overlay, brz, dovoljno prostora), registrovan u
  TFDS keš istim `finalize_bair_download.py` obrascem (skip re-download); (2) `download_and_prepare()`
  pokrenut sa `data_dir=/tmp/bair_extract`, ~30min, generisao **20.80 GiB TFRecord** (train: 43,264
  primera/256 shard-ova, test: 256 primera/1 shard) — mnogo manje nego rana ekstrapolacija (~36GB)
  sugerisala, rani tempo rasta nije bio reprezentativan; (3) **VAŽNO: `/tmp` NIJE vidljiv sa login
  node-a** (za razliku od `/home/mls10`) — moralo se prvo kopirati `/tmp`→`/home/mls10` (privremeno,
  tesno, 38/40GB na trenutak) PA TEK ONDA `rsync` sa login node-a na `/lustre/data/mls10/`, isti
  obrazac kao dataset/checkpoint-i ranije. Staging kopija u `/home/mls10` odmah obrisana posle
  potvrđenog transfera (vraćeno na ~17GB). Sad vidljivo kao **`/data/bair_robot_pushing_small_tfrecord/`**
  (read-only), format potvrđen: `action` (4,) float32, `endeffector_pos` (3,) float32, `image_main`/
  `image_aux1` (64,64,3) uint8, **30 frejmova po sekvenci** — spremno za dataloader kad se piše
  LoRA trening skripta.
- **GOTCHA:** kopiranje/rsync je izgubio jedan nivo foldera koji TFDS očekuje (`<data_dir>/
  bair_robot_pushing_small/2.0.0/...`, mi imamo `.../2.0.0/...` bez tog srednjeg foldera, `/data`
  je read-only pa se ne može ispraviti tamo). Ne koristiti `tfds.builder(name, data_dir=...)` —
  koristiti **`tfds.builder_from_directory("/data/bair_robot_pushing_small_tfrecord/2.0.0")`**
  umesto toga, radi identično ali zaobilazi tu konvenciju.
- Za notebook čitanje/vizuelizaciju: registrovan i drugi Jupyter kernel, "Python (TF - BAIR
  dataset)" (`tf-bair` venv + `ipykernel` + `matplotlib`, oba naknadno instalirana).
- **Sledeći korak (nezavršeno):** konverzija ovog TFRecord-a u format koji Wan21 trener stvarno
  očekuje (LMDB stil sa VAE-enkodiranim latentima, po uzoru na `build_worldplaygen_lmdb.py`), ili
  pisanje sopstvenog light PyTorch `Dataset`/`DataLoader` koji čita TFRecord direktno — odluka
  za kasnije, zavisi koliko je stvarni `Trainer` (`wan_trainer/camera_ar_diffusion.py`) vezan za
  LMDB specifično.

## Za sledeću/novu sesiju (ako se kontejner restartuje)

- Ovaj CLAUDE.md (na trajnom `/home/mls10`) učitava se automatski kad je `cwd` unutar
  `/home/mls10/minWM` — nova sesija odmah ima sav ovaj kontekst.
- **`/data` mount NIJE automatski** — mount-ovi datasetova se fiksiraju u trenutku kad se
  kontejner/sesija pravi. Ako se dataset ("psiml-big-data" → `/lustre/data/mls10`) doda POSLE što je
  kontejner već pokrenut (kao što se desilo ovog puta — kontejner pokrenut 12:09, dataset kreiran
  12:32), `/data` se neće pojaviti dok se sesija ne restartuje SA tim datasetom eksplicitno
  attach-ovanim (preko FMLE portala, u trenutku kreiranja nove sesije — ne postoji poznat
  hot-attach). **Pre nego što se osloniš na `/data`, proveri `ls /data` prvo, ne pretpostavljaj.**
- SSH ka login node-u (10.68.6.58) iz OVOG kontejnera je slomljen za sve alate koji zovu
  `getpwuid()` (pravi `ssh`/`rsync`/`ssh-keygen` binarni fajlovi) zbog "I have no name!" UID
  problema — `paramiko` (Python) radi kao zaobilazak za autentifikaciju/komande, ali za veće
  transfere pokazalo se lakše da **korisnik sam otvori terminal na svom laptopu** (nema taj UID
  problem) i uradi `ssh`/`rsync` odatle. Login node vidi i `/home/mls10` (NFS) i `/lustre/data/mls10`.

## PyTorch okruženje

- Venv: `/home/mls10/venvs/pytorch-minwm` (bazirano na `/opt/conda/bin/python3`, 3.13.9 — ovo je
  "sistemski python" iz Pravila, odvojeno od `venvs/tf-bair`). GPU proveren: A100-SXM4-40GB, slobodan
  (0 MiB u upotrebi), CUDA 12.8 + nvcc + gcc/g++ prisutni na hostu.
- Setup skripta: `/home/mls10/scripts/setup_pytorch_env.sh` (pip install -r requirements.txt),
  `nohup` u pozadini, log `/home/mls10/logs/setup_pytorch_env.log`. flash-attn korak preskočen
  (vidi Trenutni korak — ne treba nam).

## Demo (smoke-test posle PyTorch env-a, pre restarta BAIR downloada)

- Repo quickstart (README): treba prvo `hf download MIN-Lab/minWM --local-dir ./ckpts --include "..."`
  za 3 DMD (distilovana, Stage 3) checkpointa — **poseban download, nezavisan od BAIR-a**.
- Pokreće se `Wan21/scripts/inference/run_infer_causal_camera.sh` sa `TRAJECTORY_PATH` (npr.
  `Wan21/prompts/trajectories.txt`, keyboard-stil sekvence tipa `"a*4,w*8,s*7"`).
- Ovo NIJE naš teacher-forcing checkpoint (koristi finalni distilovan model) — svrha je samo da
  potvrdi da ceo pipeline radi (GPU, hf download kroz proxy, camera-conditioned inference), pre nego
  što ulažemo vreme u LoRA kod.
- Izlaz je video fajl (mp4) na disku klastera (`OUTPUT_FOLDER=./outputs/quickstart_wan_action2v`) —
  vizuelan, ne samo tekst/metrike.
- **Kako ga korisnik gleda** (radi sa laptopa, promptuje preko ove cluster-sesije):
  - ako ima odvojen SSH/SFTP pristup klasteru → sam `scp`-uje fajl i pušta lokalno;
  - inače → upakovati kao Artifact (self-contained HTML, hostovan na claude.ai, link se otvara u
    browseru na laptopu). Limit 16MB po stranici — ako je mp4 veći, smanjiti rezoluciju/dužinu ili
    izvući par reprezentativnih frejmova/GIF pre ugradnje.

## LoRA implementacija — referenca (VideoX-Fun, `scripts/wan2.1/train_lora.py`)

Konkretan, proveren primer LoRA treninga na Wan2.1-T2V-1.3B (isti backbone kao naš):

```bash
accelerate launch --use_deepspeed --deepspeed_config_file config/zero_stage2_config.json \
  scripts/wan2.1/train_lora.py \
  --pretrained_model_name_or_path=<wan_ckpt> \
  --rank=64 --network_alpha=32 \
  --target_name="q,k,v,ffn.0,ffn.2" \
  --use_peft_lora
```

- Injektuje se preko `peft`: `LoraConfig(r=64, lora_alpha=32, target_modules=[...])` +
  `inject_adapter_in_model(lora_config, transformer3d)` — `peft==0.17.0` je već u minWM
  `requirements.txt`, ne treba dodatna zavisnost.
- Target moduli: `q, k, v` (attention projekcije) + `ffn.0, ffn.2` (prva/treća FFN sloja) u svakom
  DiT bloku — ovo je naš polazni `target_modules` za LoRA na Wan21 blokovima.
- Rank 64 / alpha 32 kao startna tačka (alpha ~ rank/2).
- Postoji i custom (ne-peft) LoRA klasa (`create_network` + `network.apply_to(...)`) u istom repou,
  ali `use_peft_lora` put je jednostavniji za nas jer se peft već oslanja na postojeći stack.
- **Ne rešava condition injection za `action`** — ovo je čisto tekstualni LoRA (menja q/k/v/ffn težine
  na osnovu teksta/podataka), ne dodaje novi ulazni signal u model. Za `action` vektor i dalje treba
  sopstveni mehanizam (npr. mali MLP koji projektuje `action` u isti embedding prostor pa se dodaje/
  concatenate-uje pre LoRA-adaptiranih slojeva, ili cross-attention na `action` embedding) — otvoreno,
  sledi posle infrastrukture.

## Odluke sa mentorom (Nedko Savov, INSAIT)

Kontekst iz mail prepiske (avg 2026), da se ne izgubi rezonovanje iza izbora gore:

- Mentor je predložio dva dataseta: [RoboNet](https://www.robonet.wiki/) ili stariji/prostiji
  Berkeley [robotic-interaction-datasets](https://sites.google.com/berkeley.edu/robotic-interaction-datasets/home)
  (odatle je BAIR robot pushing — izabran je ovaj, prostiji, zbog akcija i 5-dnevnog roka).
- Mentor je potvrdio da minWM **nema** gotovu LoRA adaptaciju za WAN ("planning to provide this but
  not given yet") — poklapa se sa nalazom iz repoa gore (LoRA infra postoji za HunyuanVideo, ne za Wan21).
  Ovo je poznat gap, ne naš propust; treba ga sami popuniti po uzoru na VideoX-Fun.
- Dve opcije za bazni checkpoint, mentor eksplicitno rangirao rizik:
  - **Teacher-forcing checkpoint (pre distilacije) + LoRA — "safer", izabrano kao primarni pristup.**
    Neće moći dugi rollout, ali kratkoročni (8-16 frejmova) hoće — što je i cilj demoa.
  - Finalni **distilovani** model + LoRA — brže/lakše (model je već optimizovan), ali rizično: prolazi
    kroz manje diffusion koraka i mentor nije siguran da će trening biti stabilan (otvoreno istraživačko
    pitanje). **Sekundarni eksperiment ako ostane vremena, nije prioritet.**
  - Referenca za teacher-forcing (Stage 1) papire: Self Forcing, Causal Forcing.
- Metrike (PSNR, SSIM, FID preko torchmetrics) — mentor predložio, potvrđeno u prepisci.

## Bitne činjenice o repou (minWM), relevantne za naš pristup

- Wan pipeline je u `Wan21/`, treniranje u `Wan21/scripts/training/`, 4 faze:
  `run_stage0_bidirectional_camera.sh` → `run_stage1_ar_camera.sh` (**Teacher Forcing AR Diffusion
  — ovo je checkpoint koji nama treba**) → `run_stage2_causal_{ode,cd}_camera.sh` →
  `run_stage3_causal_dmd_camera.sh` (distilacija, PRESKAČEMO).
- Stage 1 (teacher-forcing) izlaz: podrazumevano `logs/ar_camera_tf/checkpoint_model_<step>/model.pt`,
  posle validacije premešten u `./ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt`
  (vidi `training_wan.md`, §1.2 Stage 1). Predtrenirani prior-stage ckpt (ako se stage 0 preskoči) skida se sa
  `hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/bidirectional/**"`.
- **VAŽAN GAP:** repo-ova podrazumevana kondiciona injekcija je kamera poza preko PRoPE
  (`use_camera: true`, `CausalWanModel`), trenirana na 480×832, 77-frame video → 20 latent frejmova
  (vidi `Wan21/scripts/data_preprocessing/build_worldplaygen_lmdb.py`). To NIJE isto što i BAIR
  gripper-akcija (dx, dy, ...) na 64×64. Repo README eksplicitno kaže da su "human pose, latent
  concatenation, cross-attention" tek **planirani** condition-injection metodi, ne postojeći.
  → LoRA-adaptacija za BAIR akcije treba sopstveni condition-injection put (verovatno cross-attention
  ili latent-concat na svaki blok, kako je opisano u VideoX-Fun referenci), a ne reuse PRoPE koda 1:1.
  Ovo je otvoreno inženjersko pitanje, ne rešeno u repou.
- LoRA infrastruktura (peft) postoji uglavnom u `HY15/trainer/...` (HunyuanVideo granama) i delimično
  u `shared/configs/dits_base.py` (`lora_param_names_mapping`, `exclude_lora_layers`). Za Wan21 postoji
  samo `Wan21/demo_utils/utils.py::separate_lora_AB` (helper za razdvajanje LoRA A/B težina) — nema
  gotovog LoRA training loop-a za Wan, treba ga dodati/portovati (odatle i referenca na VideoX-Fun).
- Repo podaci se pripremaju kao LMDB (custom builder skripte, npr. `build_worldplaygen_lmdb.py`), ne
  TFDS. Naš BAIR TFDS dataset će verovatno trebati konverziju u LMDB istog stila (latente iz Wan VAE,
  prompti/uslovi) da bi se uklopio u postojeći trainer, ili pisanje sopstvenog light dataloader-a mimo
  LMDB puta — odluka za kasnije, kad vidimo koliko trainer zavisi od LMDB formata.

## Skillovi iz repoa (Claude Code project skills, `.claude/skills/`)

Repo već nosi 3 gotova skilla — auto-detektuju se kad je `cwd` unutar `/home/mls10/minWM` (ili
podfolder); ako radiš iz `/home/mls10` direktno, `cd` prvo u ovaj repo da bi se učitali:

- **`debug-world-model`** — dijagnostika za NaN loss, jitter, drift, distillation collapse i sl.
- **`integrate-new-backbone`** — recept za dodavanje novog DiT backbone-a (nama nije prioritet, backbone već imamo).
- **`onboarding-world-model`** — teorijska podloga za dvofazni (bidirectional → causal forcing) pipeline i tipične zamke.

## Okruženje

- Docker kontejner na DGX klasteru. A100-SXM4-40GB, ekskluzivno naše.
- UID nije u `/etc/passwd` → "I have no name!" i groups greške pri svakoj komandi. Normalno, ignoriši.
- NEMA `tmux`, NEMA `screen`, NEMA `module`. Za duge poslove koristi `nohup ... &` + log fajl.
- `/dev/shm` je samo 64MB → PyTorch `DataLoader` sa `num_workers>0` puca (Bus error).
  Koristi `num_workers=0` ili `torch.multiprocessing.set_sharing_strategy('file_system')`.
- Internet SAMO kroz `http_proxy`/`https_proxy` env varijable. Bez njih ništa ne radi
  (već su setovane u shell-u za ovaj klaster).
- Sistemski python: 3.13.9. Za TensorFlow (samo download BAIR-a preko TFDS) koristi `/usr/bin/python3.12`.
- Kontejner ima 256 logičkih jezgara po `nproc`/`lscpu`, ALI cgroup CPU kvota je **8 jezgara**
  (`cpu.cfs_quota_us=800000` / `cpu.cfs_period_us=100000`, potvrđeno throttling-om u `cpu.stat`).
  Za bilo šta CPU-paralelno (npr. `MAX_JOBS` pri kompajliranju) koristi ~8, ne `nproc`-ov broj —
  veći broj samo pravi thrashing (procesi se otimaju o 8 jezgara, ništa ne odmiče).
- **`/data` iz Jupyter notebook instance NIJE isto što i ovaj kontejner** — `/data` tamo je
  read-only mount od `/lustre/data/<username>`, upisuje se preko odvojenog login node-a
  (`ssh username@10.68.6.58`, bez GPU-a, samo za file transfer). Ovaj Claude Code kontejner
  nema `/data` uopšte (proverio: ne postoji). **ALI**: `jupyter-lab` proces radi u OVOM istom
  kontejneru (PID 1) — nije odvojen pod, samo `/data` mount ovde nije prikačen (verovatno vezano
  za FMLE dataset "psiml-big-data" → `/lustre/data/mls10`, kreiran 11.08.2026, još nije attach-ovan
  na ovu sesiju).
- **Diskord pravilo instituta (Stefan Stepanović, seminar kanal):** login node (10.68.6.58) je
  **SAMO za file manipulaciju** (scp/rsync/download). Environment setup i bilo kakav compute ide
  isključivo sa jupyter/compute instanci (ovog kontejnera). Ne mešati — ne pokretati trening/pip/itd
  preko login node-a.
- SSH ključ za login node: generisan lokalno (`~/.ssh/id_ed25519_login`, `ssh-keygen` ne radi zbog
  "I have no name!" UID problema — generisan preko Python `cryptography` biblioteke umesto toga).
  Čeka se da korisnik doda javni ključ na nalog pre nego što `rsync` na `/lustre/data/mls10` može da
  krene. Plan: BAIR dataset (`/home/mls10/data/bair_robot_pushing_small/`) → `/lustre/data/mls10`
  preko rsync-a sa login node-a, kad ključ bude dodat.

## Putanje

- `/home/mls10` — NFS, **TRAJNO**, ali stvarna kvota je **~40GB** (NE 44TB koje `df -h` prikazuje —
  to je veličina celog deljenog volumena; `quota`/`repquota` komande ne postoje da se tačan broj
  programski potvrdi, drži se pod ~35GB da ostane margina). Ovde kod (repo), venv-ovi. **NE veliki
  dataseti** — ti idu na `/lustre/data/mls10` preko login node-a (vidi "Za sledeću sesiju" gore).
- `/outputs` — Lustre (`exafs`), 24TB slobodno, ALI **READ-ONLY u ovom kontejneru** (potvrđeno:
  mount flag `ro` i stvaran test pisanja oba failuju). Originalna instrukcija da checkpointi idu
  ovde je **pogrešna za ovaj kontejner** — ne koristiti za pisanje.
- **`/fairing` — Lustre (isti `exafs` storage), 24TB slobodno, WRITABLE, ALI VEZAN ZA INSTANCU
  KONTEJNERA, NE ZA NALOG** — potvrđeno: folder napravljen u starom kontejneru
  (`/fairing/minWM_checkpoints/`) NE postoji u novom kontejneru (`MinWM`), iako oba pokazuju isti
  `exafs` server/kapacitet preko `df`. Znači **NIJE pouzdano za bilo šta što mora da preživi
  restart/novi kontejner** — koristi ga samo kao scratch unutar JEDNE sesije. Suprotno od `/home/mls10`
  koje je dva puta potvrđeno deljeno (stari kontejner → login node, stari kontejner → novi kontejner).
- **REVIDIRAN plan za checkpointe**, dat ovo otkriće:
  - **Bazni model (VAE + text-encoder + teacher-forcing, ~17.8GB, read-only upotreba tokom treninga)**
    → skinuti preko login node-a na `/lustre/data/mls10/`, isti obrazac kao BAIR — pojavljuje se kao
    `/data` (read-only), trajno, nezavisno od kontejnera.
  - **Naši trening izlazi (LoRA delta checkpointi, verovatno desetine-stotine MB po checkpointu, mnogo
    manji od punog modela)** → `/home/mls10` direktno (writable, trajno, i dalje ima ~24GB slobodno
    od 40GB kvote posle venv-ova — dovoljno za mnogo LoRA checkpointa).
  - `/fairing` → samo za privremene/scratch fajlove unutar jedne sesije, ne za bilo šta bitno.
- `/tmp/python` — isti `exafs` storage, mount kaže `rw` ali fajl-permisije (vlasnik drugi UID)
  odbijaju pisanje — NE koristiti, iako mount flag deluje obećavajuće.
- `/` — overlay, 1.7TB, brz ali **NESTAJE** pri restartu kontejnera. Samo scratch.

## Pravila

- Ne instaliraj ništa globalno u kontejner — nestaje pri restartu. Sve u venv pod `/home/mls10`.
- Odvojeni env-ovi: TensorFlow (samo za download dataseta preko TFDS) i PyTorch (trening, LoRA, inference).
  Ne mešaj ih u isti venv (repo-ov `requirements.txt` je čist PyTorch/diffusers/peft stack, bez TF-a).
- Pre svakog velikog downloada proveri stvarnu potrošnju sa `du -sh /home/mls10` (NE `df -h` — to
  pokazuje ceo deljeni volumen, ne našu ~40GB kvotu). `/fairing` (24T) i `/lustre/data/mls10` (preko
  login node-a) nemaju taj problem, veliki dataseti/checkpointi idu tamo.
- Checkpointi → `/fairing`, NE `/outputs` (vidi Putanje gore).

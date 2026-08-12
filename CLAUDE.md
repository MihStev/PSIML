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

## Action-conditioning — POC USPEŠAN (12.08, jutro), plan iz sinoć koriguje se u detalju

**Ispravka sinoćnjeg nacrta:** `context` NIJE promenljive dužine u praksi — `WanModel.forward()`
radi sa **fiksnom** dužinom `self.text_len=512` (umt5-xxl, `text_dim=4096`), padding logika za
promenljivu dužinu je defanzivan no-op kod. Znači ne "produžujemo" sekvencu (kako je sinoć
planirano), nego **upisujemo action embedding u već postojeći prazan (zero-padded) prostor**
unutar tih 512 pozicija — BAIR placeholder prompt tokenizuje se na svega **9 tokena**, ima
ogroman prostor. `context_lens` je svuda već `None` (ne koristi se za maskiranje) — ne treba ga
dirati.

**Test (`/home/mls10/minWM/lora_action/poc_action_injection.py`, NE throwaway ovaj put, ostaje kao
osnova za pravi kod):**
1. `ActionEncoder` MLP (4→256→4096, SiLU), inicijalizovan nasumično
2. `prompt_embeds[b, seq_lens[b], :] = action_embed[b]` — upis na prvu praznu poziciju posle
   pravog teksta
3. LoRA rank=8 (današnji cilj, ne 64) na `q,k,v,ffn.0,ffn.2`: **9.46M trenable parametara**
4. Jedan pravi trening korak (forward+backward+`optimizer.step()`), i dalje sintetički video
   latent (BAIR podaci dolaze sledeći korak), ali **prava** text encoder izlaz

**Rezultat — potvrđeno da radi:**
- `ActionEncoder` gradient norm = **0.0148 (nenula!)** — gradijent stvarno prolazi kroz ceo model
  nazad do nove MLP mreže, arhitektura je potvrđena, ne samo teoretski ispravna
- Peak memorija: **14.78GB/15.05GB — praktično identično kao sinoćnji rank-64 test (15.38GB).**
  Rank LoRA-e skoro ne utiče na memoriju (dominiraju fiksne težine modela: VAE+text-encoder+
  transformer ~15GB), očekivano.
- Vreme: 5.2s (brže od sinoćnjih 18.4s — verovatno manje LoRA parametara + topliji keš)

**Sledeći korak — DELIMIČNO ZAVRŠENO (12.08, prepodne):** zameniti sintetički video latent
PRAVIM BAIR podacima — dataset pipeline plan iz sinoć (TF venv seče prozore → PyTorch venv
VAE-enkoduje → lagan `Dataset`) ostaje na snazi. Prvi korak (TF venv seče prozore) je urađen:

- `lora_action/extract_bair_windows.py` (ostaje u kodu, NIJE throwaway) čita
  `/data/bair_robot_pushing_small_tfrecord/2.0.0` preko `tfds.builder_from_directory` i ispisuje
  sirove (neenkodirane) `(image_main, action)` parove po sekvenci u sharded `.npz` fajlove.
- Pokrenuto za oba splita: **train (43,264 sekvence → 22 sharda) i test (256 sekvenci → 1 shard)**,
  izlaz na `/tmp/bair_raw/{train,test}/`, ukupno ~15GB.
- **VAŽNO — `/tmp` je efemeran** (isti overlay rizik kao `/tmp/local_ckpts`, vidi "Putanje" ispod):
  nestaje pri restartu kontejnera. Skripta je jeftina za ponovno pokretanje (par minuta) — ne
  vredi truda da se 15GB sirovih (neenkodiranih) npz-ova trajno čuva; radije re-run skripte na
  početku nove sesije ako `/tmp/bair_raw` ne postoji.
- **Preostaje (drugi i treći korak, nezavršeno):** PyTorch venv VAE-enkodiranje ovih npz shard-ova
  (analogno `build_worldplaygen_lmdb.py`) u LMDB ili lagan `Dataset`/`DataLoader` koji čita
  direktno — ista otvorena odluka kao gore ("konverzija u format koji Wan21 trener očekuje").

**Flash-attn: ISPRAVKA — IPAK NAM TREBA, ranije pogrešan zaključak.** `Wan21/wan/modules/attention.py`
ima `attention()` dispatcher sa SDPA fallback-om (to je tačno), ALI `Wan21/wan/modules/model.py`
(cross-attention u `CausalWanModel`, koja se stvarno koristi za AR/camera trening i inferencu) poziva
**direktno** `flash_attention()` funkciju iz `attention.py` (ne kroz dispatcher), koja ima tvrd
`assert FLASH_ATTN_2_AVAILABLE` bez ikakvog fallback-a. Otkriveno tek kad je `run_infer_ar_camera.sh`
pukao na tom assert-u tokom stvarnog inference pokušaja — teoretska analiza koda unapred nije bila
dovoljna, trebalo je pokrenuti da se vidi.
- Build ponovo pokrenut, ovog puta ispravno: `MAX_JOBS=8` (realna cgroup kvota) + ninja/PATH fix
  (video ranije) + **`FLASH_ATTN_CUDA_ARCHS=80`** (samo A100/SM80, ne i 90/100/120 — seče broj `.cu`
  fajlova za kompajliranje ~4x, sa 1118 na ~280). Skripta: `/home/mls10/scripts/setup_flash_attn_v2.sh`,
  log `/home/mls10/logs/setup_flash_attn_v2.log`.
- **Pravi razlog prethodnog "nema.pristupa" problema nije bio flash-attn sam po sebi nego
  "I have no name!" UID bag** koji je pucao na drugom mestu (`getpass.getuser()` unutar
  `torch.compile` cache dir logike, pozvano iz `causal_model.py` na import-u) — rešeno postavljanjem
  `USER=mls10 LOGNAME=mls10` env varijabli pre pokretanja bilo kog Wan21 training/inference skripta.
  **Ovo je opšte pravilo za SVE Wan21 skripte, ne samo flash-attn** — dodaj `USER`/`LOGNAME`/`HOME`
  env vars uvek.

## Gotchas otkriveni pri pokretanju `run_infer_ar_camera.sh` (primenjivo na sve Wan21 inference/training skripte)

1. **`Wan21/wan_models/Wan2.1-T2V-1.3B` simlink obavezan** — kod hardkodira taj put (ne config).
   `mkdir -p Wan21/wan_models && ln -s /data/ckpts/Wan2.1-T2V-1.3B Wan21/wan_models/Wan2.1-T2V-1.3B`
   (koristi `/data/ckpts/...` kao target, ne `./ckpts/...` — naši fajlovi su na `/data`).
2. **`diffusion_pytorch_model.safetensors` (5.68GB, osnovni T2V transformer) MORA biti fizički
   prisutan** u `wan_models/Wan2.1-T2V-1.3B/`, čak i kad se posle prepisuje teacher-forcing
   checkpoint-om preko `--checkpoint_path` — `CausalWanModel.from_pretrained()` ga učitava prvo.
   Već skinut i na `/data/ckpts/Wan2.1-T2V-1.3B/` (dodat naknadno istim rsync obrascem).
3. **`USER`/`LOGNAME`/`HOME` env vars obavezni** (vidi gore, "I have no name!" bag pogađa
   `torch.compile`/`getpass.getuser()`).
4. **`trajectory_path` fajl mora imati >= onoliko linija koliko `data_path` (tekstualni promptovi)
   ima** — `assert len(trajectory_list) >= num_prompts`. Za brz test sa 2 videa, napravi i skraćeni
   `data_path` fajl (2 linije) da se poklopi sa skraćenim `trajectory_path` (2 linije).
5. **Cold-start učitavanje je sporo** (~7 min za VAE 0.5GB + text-encoder 11.36GB + checkpoint 5.96GB
   + transformer 5.68GB sa `/data` Lustre mounta) — normalno, ne prekidati misleći da je zaglavljeno.
   Proveri `ps aux` za CPU%/RSS rast, ne samo GPU memoriju (GPU ostaje ~0% dok je sve na CPU strani).
6. **`OUTPUT_FOLDER` treba da ide na `/tmp` ili sličan non-kvotisan prostor** za test-pokretanja
   (npr. `/tmp/eval_ar_wan`), ne u repo (`/home/mls10` kvota).

**PyTorch env (`requirements.txt` bez flash-attn): uspešno instaliran**, venv na
`/home/mls10/venvs/pytorch-minwm`. Spreman za korišćenje.

**Model checkpoint-i: GOTOVO.** Bazni Wan2.1-T2V-1.3B (VAE + text-encoder + config, BEZ osnovnog
transformera — ne treba nam, koristimo teacher-forcing checkpoint) + `ar_diffusion_tf/model.pt`
(teacher-forcing, 5.96GB) skinuti ovde preko `hf download`, pa preko login node `rsync`-a (isti
obrazac kao BAIR) na `/lustre/data/mls10/ckpts/`. Sad vidljivo kao **`/data/ckpts/`** (read-only) u
svakom kontejneru koji ima dataset attach-ovan. Struktura:
```
/data/ckpts/Wan2.1-T2V-1.3B/{Wan2.1_VAE.pth, config.json, models_t5_umt5-xxl-enc-bf16.pth, google/umt5-xxl/*}
/data/ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt
```
Napomena: `diffusion_pytorch_model.safetensors` (osnovni T2V transformer, 5.68GB) je ISPOČETKA bio
preskočen, ali se ispostavilo da je obavezan (`CausalWanModel.from_pretrained()` ga učitava pre nego
što se prepiše teacher-forcing checkpoint-om) — naknadno skinut istim obrascem, sad je deo
`/data/ckpts/Wan2.1-T2V-1.3B/`.

**Lokalni keš checkpoint-a: GOTOVO za ovu sesiju** — `/tmp/local_ckpts/` (22GB, `cp -r` sa `/data/ckpts/`,
NE `rsync` — taj binarni fajl ne postoji u ovom kontejneru, samo na login node-u). Koristiti
`CHECKPOINT_PATH`/simlink ka `/tmp/local_ckpts/...` umesto `/data/ckpts/...` za brže učitavanje ubuduće
u OVOJ sesiji. **Nestaje pri restartu kontejnera** (overlay `/`) — ponoviti
`cp -r /data/ckpts/. /tmp/local_ckpts/` na početku svake nove sesije, ODMAH, pre prvog test-pokretanja.

## Jupyter kernel — TODO na početku svake nove sesije

Jupyter Lab (isti kontejner, PID 1) podrazumevano ima samo JEDAN kernel: bazni conda Python
(`/opt/conda/bin/python`), NE naš venv. Dva odvojena problema koja to pravi:
1. Bazni Python ima svoj `torch==2.13.0+cu130` koji je **polomljen za ovaj hardver** — drajver
   ovde podržava CUDA 12.8, taj torch traži CUDA 13.0 → `torch.cuda.is_available()` vraća `False`
   sa jasnom porukom ("NVIDIA driver on your system is too old"). Ne naš bag, zatečeno stanje.
2. Naš `pytorch-minwm` venv (ispravan `torch==2.9.1+cu128`) nije registrovan kao Jupyter kernel
   uopšte, pa ga notebook ne nudi kao opciju.

**Fix (uraditi na početku nove sesije ako korisnik radi iz notebook-a, ne samo terminala):**
```bash
/home/mls10/venvs/pytorch-minwm/bin/pip install ipykernel
/home/mls10/venvs/pytorch-minwm/bin/python -m ipykernel install --user --name pytorch-minwm \
  --display-name "Python (minWM - torch 2.9.1+cu128)"
```
Posle toga korisnik u notebook-u bira Kernel → Change Kernel → "Python (minWM - torch 2.9.1+cu128)".
Registracija ide u `/home/mls10/.local/share/jupyter/kernels/` — proveriti da li preživljava
restart kontejnera (verovatno da, `.local` je pod `/home/mls10` NFS, ali nije još potvrđeno).

## Mentorov test brzine/memorije (Nedko Savov) — REZULTAT

Pokrenuto `run_infer_ar_camera.sh` sa teacher-forcing checkpoint-om, default rezolucija/dužina iz
repoa (77 raw frejmova / 20 latentnih, ne naša BAIR skala).

**Brzina** (5 blokova po 4 frejma, autoregressive, KV cache raste): 30s → 37s → 44s → 52s → 59s po
bloku (usporava sa rastom KV cache-a) = **~222s (3.7min) za ceo video od 20 latentnih frejmova**.

**Memorija — OOM.** Diffusion sampling stigne do **37.6GB / 39.67GB** (skoro cela kartica), pa puca
na VAE decode koraku (treba dodatnih 7.71GB, ima samo 2.07GB slobodno). **Na default rezoluciji/
dužini iz repoa, model NE STAJE ceo (sampling + decode) na jednom A100 (40GB).**

**VAŽNA NAPOMENA:** ovo je default konfiguracija repoa (visoka rezolucija/dužina), NE naš stvarni
cilj (BAIR 64×64, 8-16 frejmova) — gornja granica modela uopšte, ne nužno naš budući memory profil.

**REŠENO retry-em sa `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** — bila je fragmentacija
memorije, ne stvaran nedostatak prostora. Sa ovim env var-om, ceo test (2 videa, sampling + VAE
decode) prošao bez greške:
- **Peak GPU memorija: 38.6GB / 39.67GB (~97%)** — jedva stane, izuzetno tesno, ali radi.
- **Latencija do prvog denoised chunk-a (bez decode-a):** 29.7s
- **Ukupno vreme po videu** (svi blokovi + decode, pun ~19-koračni trajectory): ~242s (4 min)
- Oba mp4 fajla uspešno generisana u `/tmp/eval_ar_wan/`.
- **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` treba dodati kao standardni env var za SVE
  buduće Wan21 pokretanja** (inference i trening), ne samo za ovaj test — cena je nula, korist
  potencijalno sprečava OOM od fragmentacije.
- **Zaključak za Nedka:** model radi na 1x A100 na default (visoka rezolucija/dužina) konfiguraciji,
  ali sa vrlo malo margine (~3%). Naš stvarni cilj (BAIR 64×64, 8-16 frejmova) je mnogo manji obim —
  memorijski otisak bi trebalo da bude drastično niži, verovatno dosta prostora za LoRA trening.
  Ipak vredi potvrditi na stvarnoj BAIR skali kad dodjemo do toga, ne pretpostavljati.

**ISPRAVKA — `torch.no_grad()` teorija je bila pogrešna, ne samo "nepotrebna".** Prvobitno
primećeno da `WanDiffusionWrapper.model` (glavni transformer, `wan_utils/wan_wrapper.py:132`) ima
samo `.eval()`, ne `requires_grad_(False)` — tehnički tačno, ali **irelevantno u praksi**: u
`wan_inference.py` liniji 76 postoji `torch.set_grad_enabled(False)`, pozvano GLOBALNO na samom
početku skripte, pre konstrukcije pipeline-a. Gradijenti su bili isključeni za CEO proces od
starta — moj dodati `with torch.no_grad():` (linija ~259) bio je čist no-op naslagan na već
globalno isključen autograd, ne "fix" nečega što je bilo pokvareno. **Uklonjen iz koda** (vraćeno
na originalno stanje) — repo-ov originalni inference kod je ispravan po ovom pitanju, moja
prvobitna dijagnoza (da fali `no_grad`) bila je netačna, samo se slučajno poklopila sa tim da
peak memorija zaista ostaje ista (jer NIJE ni bilo šta da se popravi).
- Umesto toga dodata prava instrumentacija u `wan_inference.py` (minimalna, ne menja ponašanje):
  baseline VRAM snapshot posle učitavanja modela (`torch.cuda.memory_allocated/reserved`, posle
  linije ~131, pre `reset_peak_memory_stats`), i `torch.cuda.max_memory_allocated/reserved` na
  kraju skripte (posle linije ~305) — peak preko cele generacije (oba videa), ispisano kao
  `[MEM] baseline after model load: ...` i `[MEM] peak during generation: ...`.
- Sledeći korak je i dalje test na stvarnoj BAIR skali (manja rezolucija → mnogo manji VAE decode
  tenzori automatski, verovatno bez OOM rizika uopšte).

**Demo (quickstart Wan Action2V inferenca): ODRAĐENO.** DMD checkpoint (`hf download MIN-Lab/minWM
--include "Wan21/Action2V/dmd/**"`, 5.96GB) skinut direktno na `/tmp/local_ckpts_dmd/` (lokalno,
bez login-node dance-a — jednokratna upotreba, ne treba trajno na `/data`). Rezultat: chunk0
latencija 1.22s, peak memorija 27.5GB — potvrđuje "real-time" tvrdnju iz README-a. 2 mp4 fajla
generisana, upakovana u Artifact (video ugrađen kao base64, samostalan link, ne zavisi od
kontejnera): https://claude.ai/code/artifact/094e191a-27dc-43ac-bbe9-3d8df7af746d

## Mock trening test na BAIR skali — REZULTAT (ključan nalaz, konačan odgovor Nedku)

Namerno **privremen, throwaway skript** (napisan van `Wan21/`, obrisan posle merenja — ne ostaje
u kodu). Koristio je **pravu** `CameraCausalDiffusion.generator_loss()` iz `model/camera_diffusion.py`
(ista funkcija koju stvarni `Trainer.train_one_step` zove), sa `peft` LoRA injektovanom preko
teacher-forcing checkpoint-a, na **sintetičkim (nasumičnim) tenzorima BAIR oblika** — memorija/
brzina zavise od oblika tenzora, ne od sadržaja, pa nije trebalo čekati BAIR ekstrakciju.

Konfiguracija testa: batch=1, **16 frejmova** (gornja granica 8-16 cilja, deljivo sa
`num_frame_per_block=4`), latent **8×8×16kanala** (64×64 piksela / 8x VAE downsample), rank 64
LoRA na `q,k,v,ffn.0,ffn.2`. Camera (viewmats/Ks) = identity placeholder (BAIR nema kameru, naš
`action`-conditioning mehanizam još nije dizajniran — ovo samo vežba postojeću arhitekturu dok se
to ne uradi).

**Rezultati:**
- LoRA trenable parametri: **75.69M** (rank 64, 5 target modula po bloku)
- Baseline posle load-a modela + LoRA: 14.76GB allocated / 15.10GB reserved
- **Peak tokom treninga (forward+backward+optimizer.step()): 15.38GB allocated / 15.43GB reserved**
  — od 40GB dostupno, **~62% kartice slobodno**
- Vreme za 1 korak: 18.4s — **napomena: samo JEDAN korak ikad pokrenut, verovatno uključuje
  jednokratni CUDA "cold start" (kernel odabir/kompajliranje); steady-state vreme po koraku je
  verovatno niže. Ako treba precizna procena ukupnog trajanja treninga, ponoviti sa 3-5 uzastopnih
  koraka i uzeti prosek koraka 2+ (izbeći prvi).**

**Zaključak: NE treba manji model.** Na BAIR skali imamo ogromnu memorijsku marginu (15.4GB od
40GB), potpuno suprotno od tesne situacije na default (832×480) rezoluciji koju smo ranije merili.
Ovo je konačan odgovor na Nedkovo pitanje o "computationally limited" scenariju.

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

## Discord status report (dan 1, uveče) i Nedkov fidbek (dan 2) — akcione stavke

**Poslato Discord grupi (kraj dana 1):** rezime infrastrukture (repo/venv-ovi/flash-attn setup ~26min,
BAIR 31GB + checkpoint-i 22GB na `/lustre`), stress-test rezultati (38.36GB peak / 29.6s po chunk-u /
~242s po videu na **default 832×480/77fr rezoluciji repoa, NE na BAIR skali**), DMD smoke-test
(27.5GB peak / 1.22s latencija, potvrđuje "real-time" tvrdnju), i eksplicitna napomena da mock BAIR
trening test (throwaway skript, vidi sekciju gore) **NIJE tretiran kao pouzdan benčmark** — samo dodatni
nalaz, brojevi namerno neobjavljeni dok se ne razume tačno šta skript radi. Generisani demo video-i:
[Google Drive link](https://drive.google.com/drive/folders/1128EaGWiBHfUyeqCa3Aw-E29Qm4VYYRR?usp=sharing).
Podela rada dan 1: Mihajlo teorijska strana, Dawidzard infrastruktura/tehnički detalji.

**Nedko Savov (mentor) — odgovor, dan 2:**
1. **Memorijska zabrinutost:** 38GB peak je **samo inference** (stress-test), sa gradijentima (pravi
   trening) treba znatno više od "still 2GB slobodno" — na default rezoluciji vjerovatno ne staje.
   **VAŽNO — ovo se odnosi na default 832×480 rezoluciju, NE na naš stvarni BAIR cilj (64×64, 8-16
   frejmova)**, gdje mock test već pokazuje ogromnu marginu (15.4GB od 40GB, vidi sekciju gore) — ali
   to treba potvrditi na **pravom** treningu (real data + real LoRA loop), ne pretpostaviti.
2. **Predlog — keširanje VAE feature-a:** raditi jedan forward pass kroz dataset unapred i čuvati
   VAE-enkodirane latente umesto sirovih slika kao ulaz. **Ovo se poklapa 1:1 sa već otvorenim
   "sledećim korakom"** u našem planu (VAE-enkodiranje BAIR npz shard-ova → LMDB/`Dataset`, vidi
   sekciju "Action-conditioning" gore) — Nedko potvrđuje pravac, nije novi rad.
3. **Predlog — gradient checkpointing / activation recompute:** eksplicitno rekao da NE treba
   implementirati ako već nije deo minWM-a (previše komplikovano), "možda je samo switch". **Provjereno
   u kodu (12.08): VEĆ POSTOJI I UKLJUČENO PO DEFAULTU.** `Wan21/wan/modules/model.py` i
   `causal_model.py` imaju `_supports_gradient_checkpointing = True`, stvarno pozivaju
   `torch.utils.checkpoint.checkpoint(...)` u forward-u; **svi** stage config YAML-ovi
   (`ar_camera_tf.yaml` itd.) imaju `gradient_checkpointing: true` po defaultu. Akciona stavka za nas:
   samo osigurati da naš (budući) custom LoRA training loop poziva
   `wan_wrapper.enable_gradient_checkpointing()` / poštuje taj flag — ne treba ništa novo graditi.
4. **Procena vremena treninga:** na osnovu naše forward-pass tajminga, **~20-30h wall-clock za 2.5k
   iteracija**, što Nedko smatra minimumom za vidljivu kontrolabilnost — ali napominje da naš
   (jednostavniji) dataset možda daje rezultate ranije. Otvoreno, potvrditi empirijski.

**Novi zadaci (dan 2, iz ove diskusije):**
- [ZAVRŠENO, ispred plana iz reporta] BAIR tar → TFRecord → window-ekstrakcija (`extract_bair_windows.py`)
- [ZAVRŠENO] VAE-enkodiranje window-a → LMDB (`build_bair_lmdb.py`), 43,264 trening sekvenci,
  latenti `(8,16,8,8)` + sirove akcije `(30,4)` po uzorku. Kompatibilan sa postojećim
  `CameraLatentLMDBDataset` (dummy identity kamera polja) + novo `actions` polje.
- [ZAVRŠENO] **Pravi (ne mock) memory/speed benchmark** (`real_training_benchmark.py`) — spojio
  LoRA rank=64 (75.69M trenable) + action-injection POC + gradient checkpointing + PRAVE latente
  iz LMDB-a u jedan pravi training korak (forward+backward+optimizer.step()). **REZULTAT (BATCH=1):**
  steady-state **0.68s/korak**, peak memorija **15.40GB/40GB** (skoro identično mock testu, 15.38GB
  — dobra unakrsna potvrda memorije). Projekcija za 2.5k iteracija: **~28 minuta** — DRASTIČNO manje
  od Nedkove procjene (20-30h), jer je njegova procjena bila iz native-rezolucijskog (832×480)
  tajminga, ne BAIR skale. **VAŽNA OGRADA: ovo je BATCH=1.** Repo-ov onboarding skill izričito kaže
  "bs < 8 nije dovoljno za kontrolabilnost" — pravi trening mora ići sa batch≥8, vrijeme/memorija
  će se promijeniti (vjerovatno ne linearno, GPU se bolje iskoristi pri batch-ovanju, vidi isti
  efekat kod VAE-enkodiranja). **Sledeći korak prije javljanja mentoru: ponoviti benchmark sa
  batch≥8 za pouzdan konačan broj.**
- [U TOKU] LoRA fine-tuning implementacija (condition-injection POC gotov, VideoX-Fun referenca kao
  osnova za training loop, vidi sekciju gore) — glavni tehnički cilj
- [NEODLUČENO] Kako se akcija ubacuje po AR bloku (BAIR daje 30 akcija/epizodu, injection mehanizam
  trenutno uzima JEDNU — benchmark gore koristi mean-pool kao privremeni placeholder, NE rešava
  ovo pitanje). Otvoreno za diskusiju sa mentorom/timom.
- [PLANIRANO] Čitanje originalnog minWM paper-a detaljnije (obostrano, teorijska podloga)

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

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
  efekat kod VAE-enkodiranja).
- [ZAVRŠENO] **Rank sweep na batch=8 (KONAČAN broj za Nedka).** Ponovljen isti benchmark za
  rank ∈ {8, 16, 64}, batch=8, 5 koraka svaki (`real_training_benchmark.py --rank R --batch_size 8`):

  | rank | trenable params | s/korak (steady) | peak mem | 2.5k iter |
  |------|-----------------|-------------------|----------|-----------|
  | 8    | 9.46M           | 1.28s             | 15.35GB  | 0.89h     |
  | 16   | 18.92M          | 1.29s             | 15.40GB  | 0.90h     |
  | 64   | 75.69M          | 1.30s             | 15.75GB  | 0.91h     |

  **Nalaz: rank ne utiče praktično ni na brzinu ni na memoriju na BAIR skali** (razlika 8→64 je
  0.3GB / 0.02s) — LoRA matrice su premale naspram ostatka modela da bi se osjetilo. Znači izbor
  ranka (Mihajlo/mentori pominjali 8 ili 16, ranije korišćeni 64 je bio nasleđen iz VideoX-Fun
  primjera, ne stvarna odluka) treba da bude o kvalitetu/overfitting riziku, NE o trošku — trošak
  je identičan. **Svi rankovi projektuju ~1h za 2.5k iteracija — konačan, pouzdan odgovor na
  Nedkovu procjenu od 20-30h**, potvrđen na pravim podacima i realnom batch size-u (uslov "bs≥8"
  iz onboarding skilla ispoštovan).
- [U TOKU] LoRA fine-tuning implementacija (condition-injection POC gotov, VideoX-Fun referenca kao
  osnova za training loop, vidi sekciju gore) — glavni tehnički cilj
- [PLANIRANO] Čitanje originalnog minWM paper-a detaljnije (obostrano, teorijska podloga)

## ARHITEKTONSKA ODLUKA (12.08, nakon review-a jačeg modela) — action injection PREKO timestep/AdaLN, ne text-slot

**Stari mehanizam (text-embedding slot upis, iz POC-a) se NAPUŠTA za pravi trening.** Razlog:
mean-pool 30 sirovih akcija u JEDNU (placeholder korišćen za benchmark gore) je **diskvalifikujuć,
ne samo suboptimalan** — BAIR akcije su delte (pomjeraj po prelazu), prosjek npr. "lijevo pa nazad"
≈ nula, model nema iskoristiv signal, nauči da ignoriše akciju, generiše "nešto što liči na BAIR"
sa pristojnim PSNR-om ali BEZ kontrolabilnosti — demo bi bio mrtav.

**Novi mehanizam — potvrđeno izvodljiv u kodu (ne samo predlog):**
`causal_model.py:1006-1008` — `e0 = self.time_projection(e).unflatten(dim=0, sizes=t.shape)`
— **postoji već gotova PO-FREJMU timestep/AdaLN magistrala**, oblika `[B, F, 6, dim]` (svaki
latentni frejm ima svoj noise-level embedding, jer je to suština teacher-forcing/diffusion-forcing
treninga). Akcija se dodaje na ovu magistralu, ne na text-embedding.

**Poravnanje akcija ↔ latentni frejmovi** (VAE je kauzalan, frejm 0 poseban — potvrđeno našim
ranijim decode testom: 8 latenata → 29 sirovih frejmova = 1 + 4×7):
```
sirovi frejmovi:  0 | 1  2  3  4 | 5  6  7  8 | ...
latentni frejm:    0 |     1      |     2      | ...
akcije (a_t = prelaz t→t+1):
  latent 0 -> nema prošle akcije (nula ili a_0)
  latent 1 -> a_0,a_1,a_2,a_3 (16 brojeva, FLATTEN, ne prosjek!)
  latent 2 -> a_4,a_5,a_6,a_7
  ...
```

**ActionEncoder v2:** `Linear(16,256) -> SiLU -> Linear(256,256) -> SiLU -> Linear(256,dim)`,
**poslednji Linear zero-init** (weight i bias) — model na koraku 0 ponaša se identično pretreniranom,
ništa se ne uništava startom. Dodaje se na per-frame timestep embedding PRIJE `time_projection`
(ili na projektovani modulation vektor — provjeriti koje se oblici čistije poklapaju).

**Dodatno u planu:** normalizacija akcija (mean/std preko trening seta, sačuvati u checkpoint),
action dropout p=0.1 (naučeni null embedding, omogućava CFG na inferenci), fiksan tekst (jedan
caption, T5-enkodiran JEDNOM, keširan na disk — tekst više NE nosi akciju).

## NOVI RIZICI (identifikovani review-om, nismo ih ranije vidjeli)

1. **[RIJEŠENO, 12.08] 64×64 NIJE prenisko za ovaj backbone — potvrđeno na pravim podacima.**
   Dva testa (`lora_action/resolution_diagnostic.py`, `resolution_compare.py`), NO LoRA (čist
   pretreniran backbone):
   - **Test A (jedan korak, automatski rastući šum po frejmu iz training rasporeda):** rani
     frejmovi zamućeni ali prepoznatljivi, kasniji frejmovi degradiraju u boju/šum bez strukture.
     Prvo je izgledalo zabrinjavajuće, ALI:
   - **Test B (fiksiran umjeren šum na SVIM frejmovima + 3 iterativna koraka refinement-a, čist
     protokol):** **konzistentno dobra rekonstrukcija kroz CIJELU sekvencu**, gornji (pravo) i
     donji (predikcija) red se skoro poklapaju, boje/oblici/pozicije tačni, oba testirana uzorka.
     → **Test A "kaša" je bila posljedica visokog šuma po frejmu (training raspored), NE
     rezolucije.** DiT jasno može da predstavi/denoise-uje BAIR scenu na 64×64.
   - **Pokušaj testa na 128×128 (bicubic upsample) je PUKAO** — ne kvalitetom, nego strukturno:
     `flex_attention` block_mask je precomputed fiksne veličine (vezan za native 8×8 latent/64×64
     konfiguraciju), ne rekonstruiše se automatski za drugu rezoluciju/broj tokena. Popravka bi
     tražila dodatnu izmjenu koda (parametrizacija block mask konstrukcije po rezoluciji).
   - **ODLUKA: ostajemo na native 64×64.** Rizik riješen, upsampling na 128/256 se NE radi (nema
     potrebe s obzirom da je 64×64 potvrđeno OK, a fix za block_mask je netrivijalan dodatni posao
     koji ne donosi korist s obzirom na 5-dnevni rok).
2. **PSNR/SSIM/FID ne dokazuju kontrolabilnost** — BAIR ima fiksnu kameru/statičnu pozadinu, model
   koji IGNORIŠE akciju i samo kopira kontekst dobija pristojan PSNR. **Prava metrika: action-swap
   divergence** — isti početni frejm, dvije različite akcione sekvence (prava vs. shuffle/negacija),
   rollout oba, izmjeriti L2/PSNR razliku IZMEĐU njih. Ako je razlika ~0, model ignoriše akciju bez
   obzira na apsolutni PSNR/SSIM/FID. Dodati ovo, javiti mentoru da se dodaje.
3. **`num_frame_per_block=4`** — provjereno u kodu (`ar_camera_tf.yaml`, `causal_model.py`),
   konfigurabilan parametar, ALI vjerovatno "zapečen" u to na čemu je Stage-1 checkpoint treniran
   (block-causal maska + KV-cache raspored). NE dirati veličinu bloka radi finije kontrole —
   granularnost rešava per-frame injection (gore), ne manji blok.
4. **Timestep sampling mora odgovarati Stage-1 režimu** (per-frame nezavisni noise levels, ne isti
   timestep za sve frejmove) — tiha greška ako se ignoriše: loss pada, trening "radi", rezultat je
   loš. **Reuse-ovati postojeći trainer-ov timestep-sampling kod, ne pisati nov.**
5. Rank 64 vjerovatno overkill — novo znanje ("akcija→dinamika") je u ActionEncoder-u, ne u LoRA
   rangu. Razmotriti rank 32 + odvojen, viši LR za ActionEncoder (npr. 3e-4 vs 1e-4 za LoRA).
6. **Prije dugog treninga, obavezno:** (a) grad-check na LoRA+ActionEncoder zajedno (imamo sličnu
   provjeru već, ponoviti sa novim mehanizmom), (b) overfit-one-batch (~300 koraka na JEDAN batch,
   loss mora ići skoro na nulu, decode mora vizuelno odgovarati) — jeftina provjera PRIJE trošenja
   GPU sati na dugi trening.

**Sledeći korak (redoslijed, dogovoren):** (1) rezoluciona dijagnostika [ZAVRŠENO] → (2) pisanje
training loop-a sa per-frame action injection [ZAVRŠENO] → (3) grad-check + overfit-one-batch
[ZAVRŠENO] → (4) pravi dugi trening, checkpoint na svakih 250 koraka, action-swap divergence eval
na svakom checkpointu [SLEDEĆE].

## IMPLEMENTACIJA training loop-a — ZAVRŠENA, sanity-check PROŠAO (12.08)

**Model surgery** (stvarna izmjena core repo koda, ne samo naš `lora_action/` folder — namjerno,
malo i additivno, svi novi parametri default `None` tako da se ništa ne mijenja kad se ne koriste):
- `Wan21/wan_utils/wan_wrapper.py` — `WanDiffusionWrapper.forward()` dobija `action_embed=None`,
  proslijeđuje ga dalje u `clean_x is not None` (teacher-forcing) granu.
- `Wan21/wan/modules/causal_model.py` — `_forward_train()` dobija `action_embed=None`; dodaje se na
  `e` (za noisy/predicted portion, `t`) I na `e_clean` (za context portion, `aug_t`) **PRIJE**
  `time_projection`, oba puta `action_embed.flatten(0,1)` (isti F frejmova za oba).
- `Wan21/model/camera_diffusion.py` — `CameraCausalDiffusion.generator_loss()` dobija
  `action_embed=None`, prosljeđuje ga `self.generator(...)`.
- **VAŽNO OTKRIVENO PRI TESTU (ne pretpostavljati dim iz class default-a!):** `CausalWanModel`
  class signature default je `dim=2048`, ALI stvarni checkpoint (`Wan2.1-T2V-1.3B/config.json`)
  ima `"dim": 1536` — ActionEncoder mora izlaziti u **1536**, ne 2048. Prvi pokušaj je pukao tačno
  na ovome (`RuntimeError: size 1536 vs 2048`), popravljeno odmah. Pouka: provjeriti runtime
  vrijednost (config.json), ne class default u kodu.

**Novi fajlovi:**
- `lora_action/bair_dataset.py` — `BairActionLatentDataset` (proširuje `CameraLatentLMDBDataset`
  sa `actions` poljem), `_align_to_latent_frames()` (latent 0 = nula, latent i≥1 = raw
  `actions[4(i-1):4i]` flattened, 16 dim, BEZ prosjeka), `compute_action_stats()` (mean/std,
  računa se jednom, latent-0 redovi isključeni iz statistike).
- `lora_action/train_lora_action.py` — glavni loop. `ActionEncoderV2` (16→256→256→1536,
  zero-init poslednji sloj), odvojen LR (LoRA 1e-4, ActionEncoder 3e-4), action dropout p=0.1
  (naučeni null embedding, CFG-ready), reuse `generator_loss()` bez izmjene (isti timestep-sampling
  kod kao uvijek), checkpoint sadrži LoRA+ActionEncoder+null+action stats+optimizer state.
  `--overfit_single_batch` flag za sanity-check mod (isti batch svaki korak).
- `lora_action/overfit_visual_check.py` — učita checkpoint, jedan forward na IDENTIČAN batch
  (bez dropout-a), dekoduje x0_pred i x0 real, sačuva uporedo za vizuelnu provjeru.

**REZULTAT sanity checka (rank=16, batch=8, 300 koraka, overfit na 1 fiksiran batch):**
- Loss: 0.65 → 0.10 tokom treninga (6.5x pad), eval loss (bez dropout-a) 0.062 — jasan silazan trend
- Vizuelno (`overfit_visual_check.png`): gdje god je nasumično uzorkovan šum bio nizak (ista
  napomena kao Test A ranije — evaluacija i dalje uzorkuje šum nasumično, nije fiksiran), predikcija
  se JASNO poklapa sa pravom scenom (boje, oblici, pozicije robotske ruke) — potvrđuje da model
  stvarno uči, ne samo da loss brojevi opadaju slučajno.
- **Cijeli pipeline potvrđen end-to-end:** LMDB → Dataset → action alignment → ActionEncoder →
  injekcija u AdaLN kanal → LoRA gradijent → VAE decode. Regresija provjerena prije ovoga
  (`real_training_benchmark.py` i dalje daje identične brojeve kao prije surgery-ja, kad se
  `action_embed` ne prosljeđuje).

**SLEDEĆE:** pravi dugi trening (2500 iteracija, rank TBD od tima — vidi rank sweep tabelu gore),
plus action-swap divergence eval (tačka 2 od rizika) prije/uz to.

## PRAVI TRENING POKRENUT (12.08, popodne) — W&B praćenje uživo

**Odluka tima: rank=16.** Action-swap divergence eval i PSNR/SSIM/FID **odloženi za kasnije**
(dogovoreno, ne blokira start dugog treninga).

- **W&B integracija** dodata u `train_lora_action.py` — scalar metrike (`loss`, `avg_recent_loss`)
  na svaki `--log_every` korak, **vizuelni uzorak** (pravo vs. predikcija, dekodiran) na svaki
  checkpoint. `--no_wandb` flag za lokalni debug bez W&B-a.
- **"I have no name!" UID bag riješen trajno** (ne samo prefiks po komandi) — `export
  USER/LOGNAME/HOME` dodato u `~/.bashrc`, važi za sve NOVE terminale ubuduće (Mihajlov isto).
- **Pokrenuto:** `train_lora_action.py --rank 16 --batch_size 8 --max_steps 2500
  --checkpoint_every 250`, pozadinski proces, log `/home/mls10/logs/real_training_run.log`,
  checkpoint-i u `/home/mls10/checkpoints/bair_lora/step_{250,500,...,2500}.pt`.
- **W&B run:** https://wandb.ai/sm220315d-etf-/bair-action-lora/runs/9ljziy6w — uživo praćenje
  loss krive i vizuelnih uzoraka. Očekivano trajanje ~1h (po rank=16 benchmarku, batch=8).
- Provjereno: `wandb/` folder koji `wandb.init()` pravi lokalno je već u `.gitignore` (linija 171),
  neće se slučajno komitovati.

## TRENING ZAVRŠEN (12.08, 17:20) — rezultat i status generacije

**Trening uspješno završen.** 2500/2500 koraka, 3334s (~56min), final loss **0.1327**,
avg_recent_loss **0.1263** (stabilan pad sa ~0.61 na početku, platoirao ~0.13-0.16 od pola treninga
nadalje — zdrav LoRA fine-tuning obrazac). Nula grešaka/NaN kroz cijeli log. Checkpoint
`/home/mls10/checkpoints/bair_lora/step_2500.pt`. Svih 10 checkpoint-a + puna W&B istorija
sinhronizovani na https://wandb.ai/sm220315d-etf-/bair-action-lora/runs/9ljziy6w.

## GENERACIJA VIDEA — OTVOREN PROBLEM, treba pravi AR rollout (ne brz fix)

**Cilj:** dat pravu BAIR scenu (početni frejm) + IZABRANU akciju (ne iz dataseta), generisati
nastavak. Skripta: `lora_action/generate_video.py`.

**Pokušaj #1 (ad-hoc "pogodi x0 → ponovo zašumi → pogodi opet", 5 koraka, fiksni sigma 0.9→0.15):**
NEUSPJEH. Prvi frejm (pravi kontekst) OK, ostatak degradira u teksturisanu zeleno-tamnu "kašu" bez
strukture. Latent opseg vrijednosti već sumnjivo širok (-9.7 do 11.4, naspram -3 do 4 u svim ranijim
testovima koji su denoise-ovali POSTOJEĆI sadržaj, ne generisali iz čistog šuma).

**Pokušaj #2 (prava flow-matching ODE integracija preko `scheduler.step()`, 12 koraka):**
GORE, ne bolje. Latent vrijednosti **eksplodiraju** monotono kroz korake (-9→-16→-23→-30→-36→-42→
-48→-54→-59→-64→-68→-70, opadajući ali neubijeni prirast — konvergira ka pogrešnoj/dalekoj tački,
ne ka normalnom opsegu). Dekodirano: još zasićenije zeleno, potpuno bez strukture.

**Dijagnoza (nije potvrđena, ali najvjerovatnija):** Model je treniran isključivo sa **teacher
forcing** — `generator_loss()` (i naš training loop) UVIJEK daje modelu i čist kontekst i zašumljenu
metu ISTOVREMENO, u jednom joint forward prolazu (real ground truth uvijek dostupan za loss).
Naš pokušaj generacije radi nešto suštinski drugačije: **denoise-uje cijeli blok od 7 budućih
latentnih frejmova ODJEDNOM iz čistog šuma**, uslovljeno na samo 1 poznat frejm — režim koji model
nikad nije "vidio" tokom treninga. Prava produkcijska generacija u repou
(`pipeline/causal_inference.py`, `CausalInferencePipeline`) radi **blok-po-blok, autoregresivno, sa
KV-cache-om** (`current_start` progresivno raste) — mi to nismo izgradili, samo smo pozajmili
training funkciju (`generator_loss`/direktan `model.generator()` poziv) za zadatak za koji nije
namijenjena. **VAŽNA KOMPLIKACIJA:** `CausalInferencePipeline` sam po sebi je dizajniran za
DISTILOVANE modele sa kratkim `denoising_step_list` (postoji samo u `causal_cd_camera.yaml`,
`causal_ode_camera.yaml`, `causal_forcing_dmd_camera.yaml` config-ima) — naš `ar_camera_tf.yaml`
(teacher-forcing, NEdistilovan) ga nema, pipeline se ne može direktno reuse-ovati 1:1, treba
prilagoditi za pun multi-step diffusion schedule sa KV-cache-om.

**ODLUKA:** ovo NIJE brza popravka (već pokušano dvaput, drugi put pogoršano) — treba prava
autoregresivna rollout implementacija (blok-po-blok + KV-cache), materijalno veći posao. Poslat
kontekst jačem modelu (vidi `kontekst_za_jaci_model.md`, ažuriran) za arhitektonski predlog prije
daljeg kodiranja. **Dokaz da mehanizam (LoRA+ActionEncoder) uopšte radi ostaje overfit vizuelna
provjera + zdrava loss kriva treninga — generacija je odvojen, dodatni problem, ne dovodi u pitanje
da li je trening uspio.**

## GENERACIJA RIJEŠENA + KONTROLABILNOST DOKAZANA (12.08, 18:11)

**Obje greške bile su u `generate_video.py`, NIJEDNA u treningu.** Dijagnoza vođena mjerenjem:

**BUG 1 — skala akcije (fatalna).** Izmišljene "kanonske" akcije `[3.0,0,0,0]`. Stvarni BAIR dims
0/1: std=0.0405, opseg ±0.07 → 3.0 je **74 sigma** izvan distribucije. ActionEncoder je davao
embedding norme **218 vs ~3.5 za prave akcije (62x)**. Taj vektor se dodaje na per-frame time
embedding prije `time_projection`, tj. postaje AdaLN shift/scale/gate za **svih 30 DiT blokova** —
62x predimenzionisan vektor uništi modulaciju cijele mreže. Otud zelena kaša + eksplozija latenta.
FIX: akcije se grade iz PRAVE akcione sekvence, override samo dims 0/1 unutar ±0.07; + hard guard
koji odbija pokretanje ako akcija pređe 6 sigma.

**BUG 2 — razbijena blok-struktura.** Trening uvijek daje JEDAN zajednički timestep po bloku od
`num_frame_per_block=4` frejmova (`BaseModel._get_timestep`). V1 je stavio frejm 0 na t=0 a frejmove
1-3 (isti blok!) na t=1000. FIX: generisanje na granularnosti bloka — frejmovi 0-3 pravi kontekst
(t=0), frejmovi 4-7 generisani (jedan zajednički t). Provjerena i maska
(`_prepare_teacher_forcing_mask`): `noise_context_ends = block_index * attention_block_size`, znači
noisy blok i vidi čiste frejmove SAMO prethodnih blokova → nema trivijalnog prepisivanja, trening
zadatak je legitiman.

**BUG 3 — pogrešan sampler.** Euler ODE (`scheduler.step` na `flow_pred`) akumulira grešku bez
korekcije → latenti odlutaju na ±13 (pravi su ±3-4) → zasićena ravna žuta boja. FIX: **x0-sampler**
(predvidi x0 → ponovo zašumi na sledeći sigma → ponovi), koji svaki korak vraća rješenje na manifold
podataka, + clamp na ±6. Napomena: V1 je slučajno koristio x0-pristup i ja sam pogrešno zaključio
da je sampler kriv — pao je zbog BUG 1, ne zbog samplera.

**REZULTAT (x0 sampler, 24 koraka, rank16/step_2500):** `|x0|` stabilno **0.75 kroz svih 24 koraka**,
opseg ±2.9 (tačno kao pravi latenti) — nula lutanja. Vizuelno: koherentna robotska scena kroz cijelu
sekvencu, u sva tri varijanta. Izlazi: `/home/mls10/logs/generated_v3/`.

**ACTION-SWAP DIVERGENCE — KONTROLABILNOST DOKAZANA BROJČANO:**
Ista scena (idx 100), isti šum, mijenja se SAMO akcija (`real` / `right`=dim0+0.07 / `left`=dim0-0.07):

| poređenje | kontekst L1 (treba ~0) | GENERISANI dio L1 | PSNR |
|-----------|------------------------|-------------------|------|
| real vs right | 2.38 | **33.41** | 12.78 dB |
| real vs left  | 1.08 | **20.23** | 15.68 dB |
| right vs left | 2.36 | **34.16** | 12.67 dB |

right-vs-left razlika = 34.16 L1 na skali 0-255, dok je std samog sadržaja 61.46 → **55.6% signala.**
Kontekst dio ~identičan (1-2 jedinice = mp4 kodek šum). **Akcija nedvosmisleno kontroliše izlaz.**

**Mjerenje magnitude (odgovara na "je li signal akcije preslab"):** `||e||` (time embedding) = 3.2-5.2;
`||action_embed||` = 3.24 (prava) / 5.7-5.9 (right/left); **`||emb(right)-emb(left)|| = 10.66 = 317%
od ||e||`**. Akcija je ravnopravan/dominantan doprinos AdaLN modulaciji, ne šapat — što retroaktivno
objašnjava zašto je 74σ ulaz bio toliko razoran (ogroman leveridž nad cijelom mrežom).

**"Zašto radi do pola pa pukne" (raniji simptom):** VAE dekodira latent 0 → 1 piksel-frejm, latente
1-7 → po 4. Znači latenti 0-3 (kontekst) = piksel-frejmovi 0-12 = **45% slike**, latenti 4-7
(generisano) = frejmovi 13-28. Oštra granica na 45% NIJE bila postepeno lutanje nego strukturna
granica stvarnost/generisano.

## SMJER AKCIJE VERIFIKOVAN — model je naučio TAČNU fiziku (12.08, 18:20)

**Korisnik je primijetio da video označen "right" ide LIJEVO, a "left" blago desno.** To je pokrenulo
provjeru — i ispostavilo se da su MOJE OZNAKE bile obrnute, ne model.

**Verifikacija bez modela, na sirovim BAIR pikselima** (`/tmp/bair_raw/`, 2000 epizoda): uzete
epizode sa najvećim kumulativnim `dim0` u oba smjera, izmjeren stvarni pomjeraj centroida hvataljke
(maska po boji: dominantno crveno, niska svjetlina) između frejma 0 i 29:

| akcija | stvarni pomjeraj na slici |
|--------|---------------------------|
| `dim0` pozitivno | **-8.53 px** (LIJEVO), std 8.5, n=25 |
| `dim0` negativno | **+13.18 px** (DESNO), std 7.2, n=25 |
| `dim1` pozitivno | **+7.69 px** (DOLJE), std 5.6 |
| `dim1` negativno | **-3.49 px** (GORE), std 7.2 |

→ **Robotsko `+x` je "lijevo" u slici** (koordinatni sistem robota nije poravnat sa kamerom).
`ACTION_OVERRIDES` u `generate_video.py` ispravljen (right = dim0 **-**0.07, left = **+**0.07).
`dim1` je bio tačan od početka.

**ZNAČAJ — ovo je nadogradnja rezultata, ne ispravka greške u modelu:** ranije smo imali samo
"akcija MIJENJA izlaz" (divergencija 34 L1). Sad imamo **"akcija pomjera ruku u fizički TAČNOM
smjeru"** — što je stvarni cilj projekta. Model je naučio pravu vezu akcija→dinamika, uključujući i
**asimetriju prisutnu u samim podacima** (13.2 px u jednom smjeru vs 8.5 u drugom — korisnik je to
nezavisno primijetio kao "left ide na desno blago").

## PROPUŠTENA PRILIKA U TRENINGU (za sledeći run)

- **Trenirali smo na 0.46 EPOHE**: 2500 koraka × batch 8 = 20,000 uzoraka od 43,264 → **54% dataseta
  nikad viđeno**. Nijedan uzorak nije viđen dvaput (nema overfit rizika), ali je resurs neiskorišćen.
- **GPU je bio na 34-45% iskorišćenosti, 15.4GB od 40GB.** Ključno: batch=1 → 15.40GB, batch=8 →
  15.40GB — **batch size praktično ne utiče na memoriju** (15GB su fiksne težine, aktivacije su na
  ovoj skali zanemarljive). batch 32-64 bi stalo bez problema.
- Zašto je trajalo samo 55min: 128 latentnih tokena po klipu naspram 31,200 na native rezoluciji
  (244x manje). Ubrzanje "samo" ~23x jer pri 128 tokena dominira fiksni trošak 30 blokova, ne FLOPs.
- 2500 iteracija je bio MINIMUM (Nedko), a repo skill kaže da se kontrolabilnost ne vidi prvih ~1000
  koraka → efektivno ~1500 korisnih koraka.
- **PREPORUKA za sledeći trening:** `--batch_size 32 --max_steps 8000` = 256,000 uzoraka = 5.9 epoha
  (13x više podataka), procjena ~3-4h. Imamo i memoriju i vrijeme.

**OSTAJE OTVORENO:** divergencija dokazuje da akcija MIJENJA izlaz, ali ne i da ga mijenja u
SEMANTIČKI TAČNOM smjeru (da "right" stvarno pomjera hvataljku desno). Za to treba ili vizuelna
provjera smjera ili kvantitativna mjera pozicije hvataljke. Takođe: testirano na JEDNOM uzorku
(idx 100), treba na više.

## VELIKI TRENING U TOKU (12.08, 18:22 →) — W&B run `fpppbwfj`

Pokrenut nakon što je generacija riješena i kontrolabilnost dokazana. Cilj: iskoristiti resurs koji
je prošli run ostavio neiskorišćenim (0.46 epohe, GPU na 34-45%).

**Konfiguracija:** `--rank 16 --batch_size 32 --max_steps 8000 --lr_lora 2e-4 --lr_action 6e-4
--checkpoint_every 500 --val_every 500`, checkpointi u `/home/mls10/checkpoints/bair_lora_big/`,
log `/home/mls10/logs/big_training.log`,
W&B https://wandb.ai/sm220315d-etf-/bair-action-lora/runs/fpppbwfj

- LR skaliran 2x (sqrt pravilo) jer je batch 4x veći. Reverzibilno — checkpoint svakih 500 koraka.
- 8000 × 32 = 256,000 uzoraka = **5.9 epoha** (prošli run: 0.46 epohe).

**NOVO — validacioni loss na neviđenom test splitu.** Do sada smo SVE evaluirali na trening
podacima (i `idx=100` iz demo generacije je iz train LMDB-a — model je vidio tu epizodu). Test split
(256 uzoraka, `/tmp/bair_lmdb/test`) stajao je netaknut od 13:25. Sad se evaluira svakih 500 koraka
— rješava i metodološki prigovor i detekciju prezasićenja odjednom (sa 5.9 epoha overfitting postaje
moguć, sa 0.46 nije bio).

**Izmjereno u prvim koracima:**

| | bs=8 (prošli) | bs=32 (ovaj) |
|---|---|---|
| s/korak | 1.33 | **3.45** |
| memorija | 15.4 GB | **16.7 GB** (4x batch = +1.3 GB, aktivacije zanemarljive) |
| GPU util | 34-45% | **58%** |
| propusnost | 6.0 uzoraka/s | **9.3 uzoraka/s** (+55%) |

Procjena: 8000 × 3.45s = **7.6h → kraj ~02:00**.

**PRVI VAL REZULTAT (korak 500): `val_loss=0.1483`, `train_recent=0.1409`, gap `+0.0074`** — samo 5%
lošije na podacima koje model nikad nije vidio. **Nema prezasićenja, model generalizuje.**

**Napomena:** prošli run je ZAVRŠIO na `avg_recent_loss=0.1263` posle svih 2500 koraka; ovaj je na
0.127 već na koraku 650 (8% puta). Ostatak treninga je čist dobitak. ALI: očekivati poboljšanje u
KVALITETU KONTROLE, ne u loss broju — kriva se već izravnava. Mjeriti divergencijom i tačnošću
smjera, ne loss-om.

**Kriterijum za prekid:** kad `val_loss` prestane da pada ili počne da raste dok train pada. Bolje
od bilo kog unaprijed određenog broja koraka. Checkpoint svakih 500 → zaustavljanje je besplatno.

## ISPRAVKA RANIJEG ZAKLJUČKA: 128/256 rezolucija NIJE strukturno blokirana

Ranije sam zapisao da je 128×128 "puklo strukturno" (`flex_attention block_mask` fiksne veličine) i
otpisao upsampling. **To je bio pogrešan zaključak.** Maska se kešira pri PRVOM pozivu
(`if self.block_mask is None` u `_forward_train`) — pukla je samo zato što je `resolution_compare.py`
radio 64 **pa onda** 128 u ISTOM procesu. Trening na jednoj rezoluciji nikad ne bi naišao na to.

**Ako se ide na 256×256 (kandidat za dan 3, TEK nakon evaluacije ovog treninga):**
- **BAIR je nativno 64×64** — upsampling 64→256 NE dodaje informaciju, zamućenost je upečena.
  Jedini pravi argument je REŽIM: Wan2.1 je treniran na 480p+, naših 8×8 latenta = 16 tokena/frejm
  je daleko izvan njegovog režima; 256×256 daje 32×32 latent = 256 tokena/frejm, mnogo bliže.
- Cijena: 2048 tokena po klipu umjesto 128 (**16x**), LMDB ~14.5 GB umjesto 0.9 GB, vjerovatno
  20-40 s/korak → traži manji batch. To je zaseban poludnevni eksperiment, ne "igranje".
- NE raditi prije nego što se izmjeri kontrolabilnost ovog treninga — inače se ne zna šta je čemu
  doprinijelo.

## ISTRAŽIVANJE DATASETA ZA VEĆU REZOLUCIJU (12.08, veče) — zaključeno

**Pitanje:** postoji li BAIR u većoj rezoluciji od 64×64?

**Odgovor: originalna BAIR u punoj rezoluciji (512×640) NIJE javno objavljena.** Dokazano:
- `rail.eecs.berkeley.edu/datasets/` sadrži SAMO `bair_robot_pushing_dataset_v0.tar` i njegovu
  `.tar.gz` verziju. Ništa drugo, nigdje na serveru.
- `Content-Length` tog tar-a = **32,274,964,480 bajta — bajt u bajt isto kao naš lokalni**. Link sa
  Berkeley stranice vodi na fajl koji već imamo.
- Aritmetika sadržaja tar-a potvrđuje 64×64: 189,852,160 B/fajl ÷ 256 trajektorija ÷ 60 slika
  (30 frejmova × main+aux1) = **12,360 B po slici**, a 64×64×3 = 12,288 B. Sirovi uint8, ne
  kompresovano. Da je 512×640, slika bi bila 983 KB.
- TFDS issue #2157 traži baš tu originalnu rezoluciju i stoji NERJEŠEN.

**Rješenje: RoboNet je nasljednik istih podataka u 128×128.** RoboNet sadrži baš te BAIR Sawyer
pushing podatke (isti robot, ista laboratorija; vlasnik fajlova u našem tar-u je `frederik` =
Frederik Ebert, koautor RoboNet-a). Dostupan kroz TFDS koji naša cijev već koristi.

| config | download | na disku | epizoda |
|--------|----------|----------|---------|
| `robonet_sample_128` | **120 MB** | 639 MB | 700 |
| `robonet_128` | **36.2 GB** | 144.9 GB | **162,417** |
| (naš BAIR 64×64) | 31 GB | 20.8 GB | 43,264 |

→ 3.75x više epizoda I 4x rezolucija, za isti red veličine preuzimanja kao BAIR.
→ **Sample verzija (120 MB) omogućava test cijevi za ~10 min prije obavezivanja.**
→ Izmjene koda: RoboNet akcije su **5-dim** (ne 4) → ActionEncoder ulaz 20 umjesto 16; video je
  promjenljive dužine; multi-robot (raznovrsniji, teži).

**Odbačene alternative:** Bridge `demos_8_17.zip` = **441 GB** (nemoguće kroz ovaj proxy);
Bridge `scripted_6_18.zip` = 32.5 GB, 640×480, ali novi format + drugi prostor akcija = nova cijev.
Upsampling BAIR-a 64→256 = ne dodaje informaciju (zamućenost je upečena).

**Prostor nije problem:** `/data` (Lustre) ima **22 TB slobodno** od 123 TB. Ranija bojazan
neosnovana.

## DMD ("dvije muhe" ideja) — premisa o manjoj memoriji NE STOJI

Razmatrano: raditi DMD (distilovan model) + veću rezoluciju odjednom, uz pretpostavku da DMD troši
manje memorije. Provjereno u kodu:
- `Wan21/model/dmd.py` drži **TRI kopije transformera**: `self.generator`, `self.fake_score`,
  `self.real_score` → pravi DMD trening je ~3x SKUPLJI po memoriji, ne jeftiniji.
- Ako se radi samo LoRA na DMD *checkpointu* našim postojećim flow-matching gubitkom → memorija je
  ISTA kao sad, ne manja. I tu je Nedkova zamjerka: distilovan model je učen da PRESKAČE korake, pa
  bi ga trening na pravi flow djelimično VRAĆAO unazad (otud "nisam siguran da će biti stabilan").
- **Eksperimentalna higijena:** rezolucija i DMD su nezavisne ose. Mijenjati obje odjednom znači da
  se pri lošem rezultatu ne zna šta je krivo. Razdvojiti.

**Plan:** (1) DMD LoRA na 64×64 — jeftino, čist odgovor na Nedkovo otvoreno pitanje; nestabilnost je
VALIDAN nalaz, ne neuspjeh. (2) Rezolucija odvojeno, preko `robonet_sample_128` → `robonet_128`.

## EVALUACIONA SKRIPTA (`lora_action/evaluate.py`) — napisana, smoke test PROŠAO (12.08, 21:45)

**Dizajn — dva režima, jer PSNR/SSIM imaju smisla samo u prvom:**
- **Režim A** (prava akcija epizode, ground truth POSTOJI): PSNR, SSIM, FID. FID uz eksplicitnu
  ogradu — pri ovom broju uzoraka koristi se samo kao RELATIVNA mjera između checkpointa.
- **Režim B** (isti šum, `dim0` prisiljen na ±0.07, ground truth NE postoji po konstrukciji):
  divergencija + tačnost smjera. Smjer se mjeri u dvije varijante: *relativna* (da li "desno"
  završi desnije od "lijevo" — robusno, ruka ima fizičke granice) i *apsolutna* (svaka varijanta
  mora ići u svom smjeru — strože).

**Ključne odluke:** test split (256 neviđenih epizoda, ne trening); **isti šum za sve varijante**
(inače poređenje nije pošteno); bazni model se učitava JEDNOM a checkpointi se mijenjaju u mjestu
(8 checkpointa = 5 min učitavanja umjesto 40); zadržan guard za akcije izvan distribucije.

**SMOKE TEST (step_1500, 8 scena test splita, 12 koraka):**

| metrika | vrijednost |
|---------|-----------|
| PSNR | 17.41 |
| SSIM | 0.7272 |
| divergencija (generisano) | **43.84** |
| divergencija (kontekst) | **0.00** ← ugrađena kontrola, tačno nula |
| smjer relativno | **1.000** (8/8) |
| smjer apsolutno | **0.938** (15/16) |

- `ctx = 0.00` je dokaz da mjerenje nije pokvareno (ranijih 1-2 jedinice bile su mp4 kompresija;
  ovdje se radi nad sirovim nizovima).
- Divergencija 43.84 je VIŠA od ranijih 34.16, i to na NEVIĐENIM scenama.
- **Smjer 8/8 relativno na neviđenim podacima = semantička kontrola, ne samo "akcija mijenja izlaz".**
- Ograda: 8 scena, 12 koraka — treba pun run za statističku težinu (planirano za dan 3).

## VAL LOSS — nemojte ga čitati po tački (naučeno na svojoj koži)

```
1000: 0.1194 | 1500: 0.1171 | 2000: 0.1195 | 2500: 0.1176 | 3000: 0.1079 | 3500: 0.1135
```
Val **osciluje u pojasu 0.108-0.120 od koraka ~1000**, šum ±0.006. Tokom sesije sam dvaput donio
pogrešan zaključak: nazvao plato na 2500, povukao ga na 3000 kad je val pao, pa se na 3500 vratio.
**Obje reakcije su bile na šum.** Trening loss uredno pada (0.1409 → 0.1115), što znači da sve
poslije ~1500 ide u pamćenje, ne generalizaciju.

**Zaključak: odluke se ne donose na osnovu val loss-a.** On mjeri rekonstrukciju; nas zanima
KONTROLA. Kriva `kontrolabilnost vs. koraci` iz `evaluate.py` je pouzdan kriterijum i nju čekamo.

## VELIKI TRENING ZAVRŠEN + KONTROLA U OBJE OSE (13.08, 02:10) — GLAVNI REZULTAT

**Trening (W&B `fpppbwfj`) završen: 8000/8000 koraka, 7.6h, 3.45 s/korak, NULA grešaka.**

| | početak | kraj |
|---|---------|------|
| train loss | 0.52 | **0.1003** |
| val loss (test split) | 0.1483 | **0.1030** (−30.5%) |
| epoha | — | **5.9** (prošli run: 0.46) |
| gap na kraju | — | **+0.0027** — nema ozbiljnog prezasićenja uprkos 5.9 epoha |

Val po prozorima (jedini pouzdan način čitanja, vidi sekciju o šumu):
`500–2500: 0.1244` → `3000–4500: 0.1088` → `5000–6500: 0.1059` → `7000–8000: 0.1050`.
Svaki prozor niži od prethodnog; poslednji donio svega −0.8%, dakle **stvarni plato tek oko koraka
6500–7000** — NE na 2500/3500 kako sam tri puta pogrešno tvrdio. Puštanje do kraja bilo je ispravno.
Najbolji val: **0.1008 na koraku 6500**. 16 checkpointa u `/home/mls10/checkpoints/bair_lora_big/`.

### KONTROLA U OBJE OSE — sve 4 akcije daju tačan smjer

Ista scena (test idx 100), **isti šum**, mijenja se samo akcija. Finalni checkpoint `step_8000.pt`.
Pomjeraj hvataljke mjeren detektorom crvenih piksela, od zadnjeg kontekstnog do zadnjeg generisanog
frejma (`/home/mls10/logs/gen_4actions/`):

| varijanta | dx (px) | dy (px) | očekivano | |
|-----------|---------|---------|-----------|---|
| `real` | +8.67 | −2.41 | — | referenca |
| `right` (dim0 −0.07) | **+7.27** | +2.74 | dx > 0 | ✅ |
| `left` (dim0 +0.07) | **−7.19** | +5.02 | dx < 0 | ✅ |
| `up` (dim1 −0.07) | +9.72 | **−6.76** | dy < 0 | ✅ |
| `down` (dim1 +0.07) | −1.54 | **+3.07** | dy > 0 | ✅ |

- **Horizontalna osa skoro savršeno simetrična:** +7.27 vs −7.19 → razdvojenost **14.5 px** na kadru
  od 64 px (~23% širine), samo promjenom jedne brojke u akciji.
- **Vertikalna osa RADI i testirana je prvi put** (−6.76 gore, +3.07 dolje). Asimetrija je očekivana
  — i sami sirovi podaci su asimetrični po dim1 (+7.69 dolje vs −3.49 gore).
- Vizuelno: svih 6 redova koherentna robotska scena kroz cijelu sekvencu, bez raspada.

**Ovo je glavni rezultat projekta: 2D semantička kontrola akcijom, na neviđenim podacima.**

### Metrike na test splitu (smoke, step_1500, 8 scena) — pun run tek slijedi
PSNR 17.41 | SSIM 0.7272 | divergencija 43.84 (kontekst **0.00**) | smjer 8/8 rel, 15/16 abs.

## SLEDEĆE (dan 3+)
1. **Pun `evaluate.py`** preko svih 16 checkpointa × 64 test scene → kriva *kontrolabilnost vs.
   koraci*. Empirijski testira Nedkovu tvrdnju o 2500 iteracija kao minimumu.
2. **DMD LoRA** na 64×64 — Nedkovo otvoreno pitanje; nestabilnost je VALIDAN nalaz.
3. **Prezentacija.** Rezolucija ide u "future work" (vidi sekciju o datasetima).

## PUNA EVALUACIJA — 16 checkpointa × 64 neviđene scene (13.08, 12:32) — KLJUČNI NALAZ

`evaluate.py`, test split (nikad treniran), 24 koraka uzorkovanja, isti šum po varijanti.
Rezultati sirovo u `lora_action/eval_results_step8000_run.json`.

```
 korak   PSNR    SSIM    FID    div    ctx   smj_rel  smj_aps
   500  14.40  0.6324   43.1  39.73   0.00    98.2%    85.1%
  1000  16.04  0.6873   34.4  44.03   0.00   100.0%    88.3%
  1500  16.19  0.6981   31.4  45.53   0.00   100.0%    86.7%
  2000  15.78  0.6865   32.7  44.56   0.00   100.0%    86.7%
  2500  16.77  0.7178   30.6  44.09   0.00   100.0%    84.4%
  3000  16.67  0.7180   30.9  44.62   0.00   100.0%    89.8%
  3500  17.02  0.7307   29.3  44.61   0.00   100.0%    89.1%
  4000  17.39  0.7480   29.4  43.66   0.00   100.0%    89.1%
  4500  17.65  0.7553   28.2  46.50   0.00   100.0%    86.7%
  5000  17.47  0.7483   28.5  46.99   0.00   100.0%    88.3%
  5500  17.92  0.7621   27.9  45.36   0.00   100.0%    87.5%
  6000  18.01  0.7673   27.8  44.61   0.00   100.0%    90.6%
  6500  17.91  0.7640   28.2  44.43   0.00   100.0%    92.2%
  7000  17.83  0.7607   28.9  43.56   0.00   100.0%    88.3%
  7500  17.53  0.7530   27.9  43.48   0.00   100.0%    89.1%
  8000  18.40  0.7794   27.2  43.78   0.00    98.4%    86.7%
```

### NALAZ: kontrola i vjernost su DVIJE RAZLIČITE VREMENSKE SKALE

- **Kontrola se zasiti rano i stane.** `dir_rel` = **100% od koraka 1000 kroz cijeli trening**;
  divergencija oscilira 43-47 bez trenda. Preostalih 7000 koraka NIJE donijelo ništa kontroli.
- **Vjernost raste do samog kraja.** Najbolji checkpoint po sva tri metrike je **posljednji (8000)**:
  PSNR 18.40, SSIM 0.7794, FID 27.2. Nema platoa.
- Ovo se ne bi vidjelo ni iz val loss-a ni iz bilo koje pojedinačne metrike.

**Prag kontrole u UZORCIMA, ne koracima:** korak 1000 × batch 32 = **~32,000 uzoraka** (0.74 epohe).
"2500 iteracija" (Nedkova procjena) nije prenosiva jedinica bez batch size-a — pri batch 8 to je
20k uzoraka, pri batch 32 je 80k. Naš prvi trening (2500 × 8 = 20k) stao je PRIJE ovog praga, što
vjerovatno objašnjava zašto je djelovao slabije.

**`div_ctx = 0.00` na SVIH 16 checkpointa** — kontekstni frejmovi bit-identični između varijanti,
dakle mjerenje divergencije je čisto od početka do kraja, nije artefakt.

**Za deploy/demoe koristiti `step_8000.pt`** — najbolja vjernost, kontrola ionako zasićena.

## VAE PLAFON — izmjeren, kontekstualizuje sve PSNR brojeve

Dekodiranje PRAVOG latenta naspram sirovih piksela, 32 test scene: **PSNR = 22.74 dB**
(min 21.56, max 24.69). To je **tvrd plafon** — nijedna generacija ne može biti oštrija, jer VAE
stiska 12,288 vrijednosti u 1,024 po frejmu i taj gubitak je nepovratan.

- Naš najbolji (8000): 18.40 dB = **81% plafona**.
- Za poređenje, dobar VAE na nativnoj rezoluciji daje 30+ dB; naših 22.74 pokazuje da ga koristimo
  daleko izvan projektovanog režima (8×8 latent umjesto 60×104).
- **Posljedica:** zamućenost izlaza je DOBRIM DIJELOM VAE, ne model. I ovo je prvi KVANTITATIVAN
  argument za veću rezoluciju — ona podiže PLAFON, nije estetika.
- Preostalih 4.3 dB je model, ali dio toga je nepovratan: model predviđa BUDUĆNOST, koja je stvarno
  neizvjesna, a x0-predikcija usrednjava moguće ishode → zamućenje svojstveno zadatku.

## DRUGA KAMERA (`image_aux1`) — provjerena, opravdano odbačena

BAIR ima drugi ugao koji smo odbacili pri ekstrakciji. Sad izmjereno (6 test scena):

| | `image_main` | `image_aux1` |
|---|---|---|
| udio piksela ruke | **7.50%** | **3.37%** (manje od pola) |
| ugao | bliži, ruka dominira | širi, ruka pri ivici |
| ekspozicija | uredna | jak prebačaj (spržena bijela zona) |

Slika: `/home/mls10/logs/main_vs_aux1.png`. Odluka je bila ispravna, iako donesena bez provjere.
Ostaje kao opcija za 2x podataka, ALI traži oznaku ugla u kondicioniranju — geometrija kamere je
druga, pa je i mapiranje akcija→pomjeraj drugo; bez oznake bi model učio dva suprotstavljena
mapiranja i kontrola bi OSLABILA.

## NAUČENO O SOPSTVENOM RASUĐIVANJU (za buduće sesije)

Tokom ovog projekta sam **PET puta** prerano proglasio plato/konvergenciju — na val loss-u (koraci
2500, 3500, 6000) i na eval krivoj (1500, 6000). Svaki put me demantovala sledeća tačka.
**Pravilo: ne zaključivati trend iz manje od ~4 tačke, i gledati prozore, ne pojedinačne tačke.**
Sve odluke o prekidu treninga koje sam htio donijeti bile bi štetne — final checkpoint (8000) je
ispao najbolji po vjernosti.

## CFG (classifier-free guidance) — TESTIRAN, NE ISPLATI SE (13.08, 13:05)

Trenirali smo null action embedding kroz action dropout (p=0.1) baš da bi CFG bio dostupan, pa ga
nikad nismo iskoristili. `lora_action/cfg_test.py`, sweep na 32 neviđene scene, checkpoint 8000,
isti šum kroz sve jačine. Rezultati u `lora_action/cfg_sweep_results.json`.

```
    w    PSNR    SSIM     div   smj_rel  smj_aps
  1.0   19.21  0.7974   43.05    100%     87.5%     <- obično uzorkovanje
  1.5   19.01  0.7908   43.24    100%     87.5%
  2.0   18.85  0.7822   44.10    100%     89.1%
  3.0   18.33  0.7606   45.42    100%     92.2%
```

**Radi tačno kako teorija predviđa** — monotona razmjena: jače kondicioniranje → jača kontrola,
slabija vjernost, kurs ~1:1 (+5.5% divergencije / +5.4% aps. smjera za −4.6% PSNR i SSIM).

**ODLUKA: generisati BEZ CFG-a (w=1.0).** Ne zato što ne radi, nego zato što nemamo šta dobiti —
relativna tačnost smjera je **već 100% na w=1.0**, pa CFG pojačava kontrolu koja je zasićena, a
naplaćuje vjernošću I dvostrukim računanjem (dva forward prolaza po koraku). Jedina niša: ako bi
za demo bilo bitno da SVAKI pojedinačni klip nedvosmisleno ide u pravom smjeru, w=2.0 podiže
`dir_abs` 87.5% → 89.1% uz umjeren trošak.

Napomena: `w=1.0` daje div=43.05 naspram 43.78 iz pune evaluacije na istom checkpointu — potvrda
da skripta radi ispravno. (PSNR je viši, 19.21 vs 18.40, jer koristi prvih 32 scene umjesto 64;
poređenje je validno UNUTAR sweep-a, ne prema tabeli pune evaluacije.)

## VIZUELNA DEKOMPOZICIJA ZAMUĆENJA — gdje se gubi oštrina

`lora_action/cfg_visual.py` slaže u jednu sliku: sirovi pikseli → VAE round-trip pravog latenta →
generisano na više w. Izlaz: `/home/mls10/logs/cfg_visual/cfg_zoom_idx3.png` (uveličano 8x).

**Nalaz (vizuelno, potvrđuje mjerenje od 22.74 dB):**
- **Najveći pad je sirovi → VAE**, a to je SAVRŠENA rekonstrukcija bez ikakvog generisanja. U
  sirovom su ivice oštre i natpis na ruci čitljiv; već u VAE redu sitni predmeti se stapaju.
  **Dominantni uzrok zamućenja je autoenkoder na 64×64, ne model.**
- **VAE → generisano je MANJI pad** nego prvi. Model gubi manje nego autoenkoder.
- **CFG POGORŠAVA vizuelno:** kroz w=1→2→3 raste zasićenost i pojavljuje se obojena aureola po
  ivicama (roze/magenta), plus šum na w=3.0. Tih artefakata NEMA u sirovim pikselima ni u VAE redu.

**Odbačena metrika:** "oštrina" kao prosječni gradijent NE VALJA ovdje — generisano je imalo veći
gradijent (32.6) od VAE plafona (31.5), što je nemoguće za pravi detalj. Gradijentna energija mjeri
lokalnu varijaciju, pa je artefakti podižu jednako kao detalj. Ne koristiti je.

**Posljedica za prioritete:** ovo je tvrd, vizuelan argument da nema smisla ulagati u model dok je
plafon ovoliko nizak — veća rezolucija podiže PLAFON, sve ostalo se bori za ostatak.

## FIDBEK DRUGOG MENTORA (Danilo, 13.08) — dvije prave rupe

**1. "FID je vremenski slijep, dodajte FVD."** Tačno i posebno relevantno za nas: FID skuplja
per-frame Inception odlike, pa video sa realističnim pojedinačnim frejmovima ali fizički pogrešnim
kretanjem prolazi dobro. Naša cijela tvrdnja je o KRETANJU, dakle FID mjeri baš ono što ne testiramo.
- Nije u `torchmetrics` → traži I3D (Kinetics) backbone zasebno. Planirano za dan 4.
- Ograde koje treba SAMI navesti: FVD normalno traži stotine-hiljade videa (mi imamo 256), a
  standardne implementacije skaliraju na 224×224 dok su naši frejmovi 64×64 → I3D bi uglavnom
  gledao artefakte uvećavanja. Prijaviti kao RELATIVNU mjeru između naših checkpointa.

**2. "Je li sve generisanje do sad teacher-forced? Imate li slobodne rollout-e?"**
**Najoštrije zapažanje dana. Odgovor: DA, svi prijavljeni brojevi su teacher-forced.**
- U svim mjerenjima kontekst su PRAVI frejmovi 0-3, generišu se 4-7 u jednom bloku, `clean_x` je
  pravi latent. To važi za cijelu tabelu evaluacije, CFG sweep i rezultate po akcijama.
- **Slobodni rollout POSTOJI ali samo kvalitativno** — `rollout.py` (danas): kontekst bloka N je
  GENERISANI izlaz bloka N-1. Dva runa: `up up down down` (4 bloka), `right ×6` (6 blokova).

**Šta rollout pokazuje (nepovoljno):**
- Magnitude latenta ostaju u ispravnom opsegu (±3.2 vs pravi ±3-4) kroz 4 bloka — nema numeričke
  eksplozije.
- ALI kvalitet slike progresivno propada; na 6 blokova izlaz je nekoherentan.
- Tačnost komandovanog smjera po bloku je otprilike slučajna (2/6 na `right ×6`), naspram
  ~87% aps. / 100% rel. u teacher-forced režimu.

**Mehanizam (exposure bias):** model je viđao samo ČIST, pravi kontekst. U rollout-u jede sopstveni
izlaz — uslovni prosjek, blago zamućen i van distribucije — koji se tretira kao ground truth. Greška
se MNOŽI, ne sabira. Zamućenje hrani samo sebe: zamućen kontekst = dvosmislenija scena = model
usrednjava preko više budućnosti = još zamućenije. Zato minWM ima Stage 2/3 POSLIJE Stage 1, i zato
je Nedko rekao "sigurniji, ali neće moći dug rollout".

**Dvije konkretne popravke, nijedna uzeta:**
- `noise_augmentation_max_timestep` POSTOJI u repou i kod nas je **0**. Kad je > 0, na čist kontekst
  se tokom treninga dodaje šum → model nauči da toleriše nesavršen kontekst. Jedna linija konfiga,
  ali 7.6h ponovnog treninga.
- Stage 2 self-forcing — prava popravka, cijela dodatna faza treninga.

**OTVORENA RUPA:** nismo pustili metrike NAD slobodnim rollout-ima. Kriva degradacije (metrike vs
dužina rollout-a) je očigledno sledeće mjerenje i jeftino je — kod postoji.

**ODLUKA (dan 3):** prihvatamo ograničenje i prijavljujemo ga. Razlozi: (1) posljedica je namjernog
izbora Stage 1 koji je mentor preporučio kao sigurniji, (2) naš isporučivi rezultat je KRATAK rollout
sa kontrolom, i to radi, (3) 7.6h treninga za neizvjesno poboljšanje sporedne osobine, dva dana prije
roka, nije dobra razmjena. Ide u prezentaciju kao izmjereno ograničenje sa objašnjenim mehanizmom i
dvije identifikovane popravke — što je jače nego da ga nismo primijetili.

Izvještaj za oba mentora: `status_report_day3.md` (u repou i u `/home/mls10/`).

## INTERAKTIVNI DEMO — RADI (13.08, 15:01)

`lora_action/interactive_demo.ipynb` uspješno pokrenut. Model se drži u Jupyter kernelu, pa se
5-6 min učitavanje plaća JEDNOM; svaki klik na dugme košta samo sampling (~4.5 s po bloku od 16
frejmova). Izlaz: `/home/mls10/logs/interactive_left_down_still.mp4` — 61 frejm (13 konteksta +
3 bloka × 16), niz `lijevo → dolje → miruj`.

**Gotcha za sledeći put:** tokom učitavanja modela Jupyter pokazuje `[*]` i djeluje ZAMRZNUTO 5-6
minuta. Nije zamrznuto — provjera je `ps` (kernel troši ~270% CPU, RAM raste do ~34 GB) i
`nvidia-smi` (GPU ostaje na ~400 MB dok se čita sa diska, tek onda skoči na ~15 GB). Ne gasiti.

## `rollout_metrics.py` — NAPISAN, ČEKA POKRETANJE (predaja Mihajlu)

Odgovara na **oba Danilova pitanja jednim eksperimentom**: FVD (koji vidi vrijeme) mjeren kao
funkcija DUBINE SLOBODNOG ROLLOUT-a (gdje model jede sopstveni izlaz).

**Dizajn poređenja — bitno:** prave BAIR epizode imaju samo 30 frejmova, pa iza dubine 1 NEMA
uparenog ground truth-a i PSNR/SSIM su nedefinisani. Umjesto toga, na svakoj dubini uzima se
**posljednjih 16 dekodiranih frejmova** (najsvježije generisani blok) i poredi sa nasumičnim
16-frejmnim prozorima iz pravih held-out klipova. Pitanje je na svakoj dubini isto: *"da li
najsvježiji izlaz i dalje izgleda stvarno?"*

**FVD implementacija i njena ograda:** kanonski FVD koristi određeni Kinetics I3D checkpoint koji
nije instalabilan ovdje (`common_metrics_on_video_quality` nema u indeksu). Koristimo torchvision
**S3D** (takođe Kinetics-400, arhitektonski nasljednik I3D-a), 1024-dim odlike, Fréchet distanca
preko njih. **Označavati kao `FVD*`, NE kao FVD** — nije uporedivo sa objavljenim brojevima:
(1) drugi backbone, (2) 256 epizoda umjesto hiljada, (3) naši frejmovi su 64×64 a mreža očekuje
224×224, pa I3D/S3D uglavnom gleda artefakte uvećavanja. **Validno samo kao RELATIVNA mjera između
naših dubina.** Ista ograda već važi za naš FID.

Pokretanje: `python rollout_metrics.py --max_depth 6 --n_scenes 32` (~15 min).

## KRIVA DEGRADACIJE SLOBODNOG ROLLOUT-a — IZMJERENA (13.08, `rollout_metrics.py`)

32 held-out scene, dubine 1-6, FVD* (S3D) + FID + divergencija + smjer. Sirovi izlaz u
`/home/mls10/logs/rollout_metrics.json`.

```
dubina  frejm    FVD*     FID   divergencija   smjer(ispravljen)
     1     17   106.4    99.6         48.19          96.9%
     2     33   157.6   137.9         52.84          53.1%
     3     49   169.8   154.6         55.68          68.8%
     4     65   176.1   167.6         58.54          62.5%
     5     81   176.7   179.6         61.01          59.4%
     6     97   183.5   183.8         63.22          68.8%
```

- **Dubina 1 (96.9%) nezavisno reprodukuje teacher-forced rezultat (100% na 64 scene)** — dvije
  metodologije se slažu, dakle nije artefakt nove skripte.
- **Kontrola se ruši poslije TAČNO JEDNOG samostalno generisanog bloka:** 96.9% → 53.1% (slučajno).
  Dubine 3-6 su 59-69%, međusobno statistički nerazlučive (±17% pri n=32).
- **FVD\* je osjetljiviji RANO** (+48% na skoku 1→2, FID +38%) i **zasićuje se oko dubine 4-5**
  (176→177→184), dok FID nastavlja da raste (168→180→184). Vremenska nekoherentnost se pojavi prije
  nego što pojedinačni frejmovi propadnu — tačno Danilov argument, i empirijski potvrđen.

### DVIJE METODOLOŠKE POUKE (obje vrijedne prijave)

**1. Preživljavačka pristrasnost u NAŠOJ metrici — nađena i ispravljena.** Detektor ruke (po boji)
ne uspije na jako degradiranim rollout-ima, a skripta je te scene IZBACIVALA umjesto da ih broji kao
neuspjeh. `n` je padao 32→24 na dubini 5. Sirovi brojevi bi čitali 71/71/79/85% na dubinama 3-6 —
prividan OPORAVAK koji ne postoji. Ispravno: nepraćeno = neuspjeh, uvijek /32. **Popravka u skripti
je jedna linija; do tada tabelu prijavljivati ispravljenu.**

**2. Divergencija SAMA je varljiva.** Kroz dubinu divergencija RASTE (48→63) dok tačnost smjera PADA
(97%→~60%). Dva rollout-a se sve više razlikuju, ali ne kontrolisano — oba nezavisno lutaju.
Divergencija mjeri "različito", smjer mjeri "različito NA TAČAN NAČIN". Da smo pratili samo
divergenciju, zaključili bismo da kontrola JAČA s dubinom — suprotno od istine.

**FVD implementacija:** torchvision `s3d` (Kinetics-400), 1024-dim odlike, Fréchet distanca.
Označavati kao **FVD\***, NE FVD — kanonski koristi određeni I3D checkpoint koji nije instalabilan
(`common_metrics_on_video_quality` nema u indeksu). Uz 256 klipova na 64×64 hranjenih mreži koja
očekuje 224×224, apsolutna vrijednost nije uporediva sa literaturom. Validno samo RELATIVNO.

## IZVJEŠTAJ ZA MENTORE — dan 3

`status_report_day3_final.md` (u repou i `/home/mls10/`). Format kao dan 2. Pokriva: Danilova dva
poena sa mjerenjima, punu evaluaciju, VAE plafon, CFG negativan rezultat, i **otvoreno prijavljene
naše metodološke greške** (preživljavačka pristrasnost, varljivost divergencije, neuspjela ablacija).

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

# Pregled projekta — action-conditioned world model na BAIR podacima

*Napravljeno: 12.08.2026*

---

## Dan 1 (11.08) — Infrastruktura i prve mjere

**Šta:** Repo, backbone, venv-ovi.
- Izabran **Wan2.1-T2V-1.3B** kao backbone — najlakši baseline u minWM repou (alternativa HunyuanVideo-8B odbačena, preteška za 5-dnevni rok).
- TF + PyTorch venv-ovi (odvojeni namjerno — TF samo za BAIR download preko TFDS, PyTorch za sve ostalo, da se izbjegnu konflikti zavisnosti).
- flash-attn kompajliran (~26 min, poslije par neuspjelih pokušaja zbog pogrešnog shvatanja CPU kvote — kontejner ima 256 logičkih jezgara ali cgroup limit je 8, `MAX_JOBS` mora pratiti to, ne `nproc`).

**Podaci i checkpoint-i:**
- BAIR (31GB) + checkpoint-i (22GB: VAE, text-encoder, teacher-forcing) preuzeti, prebačeni na `/lustre` preko login node-a — jer je `/home` kvota ~40GB, ne terabajti kako je prvi utisak bio (`df -h` pokazuje cijeli deljeni volumen, ne kvotu).
- DMD checkpoint (5.96GB) namjerno **ostao samo lokalno** — treba samo za demo, ne za trening.

**Prve mjere (native rezolucija repoa, 832×480/77fr):**
- Stress test: 38.36GB peak, 29.6s/chunk, 242s/video. Trebao `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` da izbjegne OOM — pokazalo se da je fragmentacija memorije, ne stvaran nedostatak.
- Provjereno da nema skrivene `no_grad` neefikasnosti — gradijenti su već globalno isključeni kodom repoa za inferencu.
- DMD (distilovan) smoke-test: 27.5GB peak, 1.22s latencija — potvrđuje "real-time" tvrdnju iz README-a.
- Kasno uveče: mock trening test (**sintetički** BAIR-oblika tenzori, rank64 LoRA) pokazao mnogo više memorijske margine — **eksplicitno neoznačen kao pouzdan**, samo dodatni nalaz.

**Podjela rada:** Mihajlo → teorija, Dawidzard → infrastruktura/implementacija.

---

## Dan 2 (12.08) — sve u ovoj sesiji

### Faza 1: Higijena okruženja i kolaboracija
- **Problem:** dvoje ljudi na istom kontejneru/GPU-u, isti Unix nalog. Rizik: sudaranje fajlova, mješanje git kredencijala.
- **Rješenje:** `git worktree` — Mihajlo ostaje na `main` u `/home/mls10/minWM`, ja radim u odvojenom folderu `/home/mls10/minWM-dawidzard` na grani `dawidzard/work`. Isti `.git`, pa sinhronizacija ide **lokalno** (`git merge`) bez potrebe za GitHub-om svaki put. Odvojeni credential fajlovi (`.git-credentials-dawidzard` / `-mihajlo`) da se tokeni ne pomiješaju.
- Očišćene mrtve/prazne Claude Code sesije (nekoliko praznih transkripta, jedan "phantom" socket).

### Faza 2: Provjera Nedkovog fidbeka naspram stvarnog koda
Umjesto da samo prihvatimo/odbacimo njegove komentare, **provjerili smo svaki u kodu**:
- Memorijska zabrinutost (38GB) — odnosila se na native-rezoluciju (inferenca), ne na naš BAIR cilj → nije direktno primjenjivo, ali još nepotvrđeno na pravim podacima u tom trenutku.
- Gradient checkpointing — **već ugrađen i uključen po defaultu** u minWM (`_supports_gradient_checkpointing`, `torch.utils.checkpoint`, sve config YAML-ove imaju `gradient_checkpointing: true`). Ništa nije trebalo implementirati.
- VAE feature caching — poklopio se sa već planiranim korakom.
- 20-30h procjena treninga — ista sumnja kao za memoriju, čekala potvrdu.

### Faza 3: Zašto minWM, ne VideoX-Fun direktno
Objašnjeno i odlučeno: VideoX-Fun je **samo recept** za LoRA injekciju (peft, `q,k,v,ffn.0,ffn.2`, rank 64 primjer) — radi nad vanilla, ne-kauzalnim Wan2.1 T2V. Naš cilj (autoregresivni, action-conditioned world model) zahtijeva kauzalnu/AR arhitekturu koju VideoX-Fun nema, a minWM ima (Stage 0→3 pipeline, samo kondicionisan kamerom umjesto akcijom). **minWM = baza, VideoX-Fun = samo recept za LoRA sloj.**

### Faza 4: BAIR podaci → latentni prostor
- `extract_bair_windows.py` (window-ekstrakcija, urađeno već uveče dan 1) → `build_bair_lmdb.py` (VAE-enkodiranje, urađeno danas): 43,264 sekvenci, latenti `(8,16,8,8)` + sirove akcije `(30,4)`, LMDB kompatibilan sa postojećim `CameraLatentLMDBDataset` (dummy kamera polja, BAIR nema kameru) + novo `actions` polje.
- **Zašto unaprijed (offline), ne "u letu":** VAE je zamrznut, ne mijenja se — enkodovati jednom je jeftinije nego iznova svaki training korak. Ovo je i bila Nedkova sugestija #2.
- Validirano round-trip dekodiranjem (nizak MSE, ne šum).

### Faza 5: Pravi (ne mock) benchmark treninga
- Prvo na batch=1: 0.68s/korak, 15.4GB — skoro identično mock testu (unakrsna potvrda memorije), projekcija ~28min/2500 iter.
- **Ispravka:** repo-ov onboarding skill kaže "bs<8 nije dovoljno za kontrolabilnost" — ponovljeno na batch=8, sweep rank 8/16/64: **rank praktično ne utiče na trošak** (1.28-1.30s/korak, 15.35-15.75GB svi). **Konačan odgovor Nedku: ~1h za 2500 iteracija, ne 20-30h** — njegova procjena je bila iz native-rezolucijskog tajminga, ne BAIR skale.

### Faza 6: Git push infrastruktura
- Fine-grained GitHub token nije radio za tuđ repo (vlasnik mora eksplicitno dozvoliti) → prešli na classic PAT (`repo` scope). Push uspio, `dawidzard/work` → `main`, fast-forward merge (Mihajlov `main` bio čist, ništa pregaženo).

### Faza 7: Dubinski arhitektonski review (jači model)
Pripremljen pun kontekst fajl, poslat na review. Ključni nalazi:
- **Mean-pool akcija (naš placeholder) je diskvalifikujuć, ne suboptimalan** — delte se poništavaju (npr. lijevo pa nazad ≈ nula), model nema signal, demo bi bio mrtav i pored dobrog PSNR-a.
- **Predlog: per-latentni-frejm injekcija kroz postojeći timestep/AdaLN kanal** — **provjereno u kodu** (`causal_model.py:1006-1008`, oblik `[B,F,6,dim]` potvrđen), ne samo teorija.
- **Najveći neprovjeren rizik: da li je 64×64 uopšte dovoljno velika rezolucija za ovaj backbone** (treniran na 480p+). VAE round-trip test ne dokazuje ništa o DiT kapacitetu.
- 5 dodatnih rizika: PSNR/SSIM/FID ne dokazuju kontrolabilnost (treba action-swap divergence), `num_frame_per_block` možda zapečen u checkpoint, timestep sampling mora odgovarati Stage-1 režimu, rank 64 vjerovatno overkill, obavezan grad-check + overfit-one-batch prije dugog treninga.

### Faza 8: Rezoluciona dijagnostika
- Test A (jedan korak, automatski rastući šum po frejmu) — kasniji frejmovi izgledali kao kaša. **Zabrinjavajuće na prvi pogled.**
- Test B (fiksiran umjeren šum + 3 iterativna koraka, čist protokol) — **konzistentno dobra rekonstrukcija kroz cijelu sekvencu**. Zaključak: kaša iz Testa A je bila posljedica visokog šuma po frejmu (training raspored), NE rezolucije. **64×64 potvrđeno radi.**
- Pokušaj na 128×128 pukao strukturno (`flex_attention` block_mask fiksne veličine) — odlučeno da se ne popravlja, nema potrebe s obzirom da je 64×64 već OK.

### Faza 9: Implementacija training loop-a (prava izmjena core koda)
- Model surgery kroz 3 fajla (`wan_wrapper.py`, `causal_model.py`, `camera_diffusion.py`) — novi `action_embed` parametar, dodaje se na per-frame timestep embedding PRIJE `time_projection`, i za context (`e_clean`) i za noisy (`e`) granu. Sve default `None`, regresijski provjereno da ništa staro nije pokvareno.
- **Uhvaćen pravi bug:** pretpostavljena dimenzija 2048 (class default u kodu) nije se poklapala sa stvarnim checkpoint-om (`config.json` kaže 1536). Prvi pokušaj pukao tačno na tome, popravljeno odmah — pouka: provjeravati runtime vrijednosti, ne pretpostavljati iz koda.
- Novi `Dataset` (poravnanje akcija po latentnom frejmu, flatten ne prosjek), pun training loop (odvojen LR za LoRA/ActionEncoder, normalizacija akcija, action dropout za CFG, checkpoint-ovanje).

### Faza 10: Sanity check — overfit jedan batch
- Loss 0.65 → 0.10 (300 koraka), eval loss 0.062, vizuelna provjera potvrđuje da model stvarno uči sadržaj (ne samo da brojevi opadaju). **Cijeli pipeline potvrđen end-to-end.**

### Faza 11: W&B + pravi trening
- Weights & Biases povezan (uz usput trajno riješen "I have no name!" UID bag — sad u `~/.bashrc`, ne treba prefiks za svaku komandu).
- **Odluka tima: rank=16.** Evalovi (action-swap, PSNR/SSIM/FID) svjesno odloženi za kasnije.
- **Pravi trening pokrenut** — 2500 iteracija, batch=8, praćenje uživo: https://wandb.ai/sm220315d-etf-/bair-action-lora/runs/9ljziy6w

*(Napomena: taj prvi trening je kasnije istog dana zamijenjen većim — batch 32, 8000 koraka.)*

---

## Dan 3 (13.08)

### Trening završen i evaluiran
- 8000 koraka, 5.9 epoha, 7.6 h, bez ijedne greške. Val loss 0.1483 → 0.1030, jaz +0.0027 — nema ozbiljnog prezasićenja uprkos skoro 6 epoha.
- **Sve četiri akcije daju tačan smjer**, prvi put testirana i vertikalna osa. Horizontala skoro simetrična: razdvojenost 14.5 px na kadru od 64 px.
- **Puna evaluacija:** 5 metrika × 16 checkpointa × 64 neviđene scene, ~2.5 h.

### Glavni nalaz dana
**Kontrola i vjernost konvergiraju na različitim vremenskim skalama.** Tačnost smjera je 100% već na koraku 1000 i tu stoji do kraja; PSNR/SSIM/FID rastu sve do 8000. Posljednjih 7000 koraka kupilo je vjernost, ne kontrolu. To se ne vidi ni iz val loss-a ni iz bilo koje pojedinačne metrike.

Prag kontrole izražen u **uzorcima** (~32k), ne koracima — "2500 iteracija" nije prenosiva jedinica bez batch size-a, i naš prvi trening je stao **ispod** tog praga, što vjerovatno objašnjava zašto je djelovao slabije.

### VAE plafon — kontekst za sve brojeve
Dekodiranje **pravog** latenta nazad u piksele daje **22.74 dB**. To je tvrda granica; naših 18.40 je 81% od nje. Vizuelno potvrđeno: najveći pad kvaliteta je sirovo → VAE, a to je korak bez ikakvog generisanja. **Dominantni uzrok zamućenja je autoenkoder, ne model.** Prvi kvantitativan argument da veća rezolucija podiže *plafon*, a nije estetika.

### Zatvorena pitanja
- **Veća rezolucija:** originalna BAIR 512×640 **nije javno dostupna** — dokazano, ne pretpostavljeno (Content-Length bajt-u-bajt isti kao naš tar, plus aritmetika sadržaja). RoboNet 128 postoji ali miješa četiri laboratorije i tražio bi filtriranje; Bridge je 441 GB. Ide u "future work".
- **Druga kamera (`image_aux1`):** provjerena i opravdano odbačena — ruka zauzima 3.4% piksela naspram 7.5%, dio kadra spržen svjetlom.
- **CFG:** implementiran i testiran (w ∈ 1/1.5/2/3). Ponaša se po teoriji, ali **ne isplati se** — relativna kontrola je već 100%, pa pojačava zasićen signal uz cijenu vjernosti i dvostrukog računanja. Mjeren negativan rezultat.

### Fidbek drugog mentora (Danilo) — dvije prave rupe
- **"FID je vremenski slijep, dodajte FVD."** Implementiran isti dan (S3D umjesto nedostupnog I3D, označen kao FVD*). Odmah opravdao uvođenje: raste +48% od dubine 1→2 naspram FID-ovih +38% — vremenska nekoherentnost se pojavi **prije** nego što pojedinačni frejmovi propadnu.
- **"Je li sve teacher-forced?"** Da, svi prijavljeni brojevi jesu. Slobodni rollout sad izmjeren: **kontrola preživi tačno jedan samostalno generisan blok** (96.9% → 53.1%, slučajno), a kvalitet monotono propada. Mehanizam je exposure bias, i to je razlog zašto minWM ima Stage 2 poslije faze koju koristimo.

### Demo materijali
- **Interaktivni notebook radi** — model ostaje u kernelu, svaki klik košta samo sampling (~4.5 s po bloku).
- Programi akcija (različita akcija po latentnom frejmu) i klizni-prozor rollout proizvoljne dužine.

### Greške koje smo sami našli i prijavili
- **Preživljavačka pristrasnost u našoj metrici** — detektor ruke ne uspije na degradiranim rollout-ima, a te scene su se **izbacivale** umjesto da se broje kao neuspjeh. Sirovi brojevi su implicirali oporavak koji ne postoji.
- **Divergencija sama je varljiva** — raste s dubinom dok tačnost smjera pada. Mjeri "različito", ne "različito na tačan način".
- **Neuspjela ablacija** — bez fiksiranog seeda, 12 binarnih mjerenja; ista postavka dala 42% i 83% u dva pokretanja.

### Organizacija
- Mihajlo prešao na svoju sesiju; riješena **git kolizija** u `CLAUDE.md` (obje sekcije zadržane, njegovo otvoreno pitanje zatvoreno mojom korekcijom).
- Napisan **naš README** — originalni minWM-ov sačuvan zasebno.
- Izvještaj za oba mentora, duga i kratka verzija (ispod 2000 karaktera za Discord).

### Veče dana 3 — organizacija i predaja
- **Videi podijeljeni mentorima** preko Google Drive-a, u tri foldera: glavni rezultat (ista scena, četiri komande), ograničenja (dekompozicija zamućenja + degradacija rollout-a), demo (interaktivni niz + programi akcija). Folder sa ograničenjima uključen namjerno — pokazuje da znamo gdje su granice.
- **Razgovor sa mentorkom:** rezultat djeluje antiklimaktično, ali to vjerovatno dolazi iz nesklada između intelektualnog i vizuelnog utiska, a ne iz kvaliteta rada. Predložila je variranje veličine parametara i dimenzije latentnog prostora — imamo mjerenja koja to preduhitruju: **rank sweep (8/16/64) je pokazao da kapacitet adaptera nije usko grlo**, a dimenzija latentnog prostora je fiksirana VAE-om (`z_dim=16`), pa bi njena promjena tražila novi VAE. Usko grlo je izmjereno i strukturno: VAE plafon.
- **Tehnički gotcha dana:** `ipykernel` 7 lomi `ipywidgets` 8 — dugmad se ne renderuju bez ikakve greške, i ni promjena browsera ni hard refresh ne pomažu jer je problem prije browsera. Spušteno na 6.31.0. Usput napisan i fallback demo bez widgeta (`go('up')` umjesto klika).
- **Higijena GPU-a:** Jupyter kernel sa učitanim modelom drži ~15 GB i kad ništa ne radi; `torch._inductor` ostavi ~30 compile radnika. Bitno kad dvoje dijeli jednu karticu i jedan nalog.

---

## Gdje smo i šta dalje

**Po osama, iskreno:**

| osa | stanje |
|---|---|
| kvalitet slike | **zaključan** — 81% VAE plafona, pravi skok traži veću rezoluciju koju nemamo |
| snaga kontrole | **zasićena** — 100% relativno, CFG nema šta da pojača |
| dužina rollout-a | popravljivo, ali traži novu fazu treninga |
| brzina | **stvaran prostor** — DMD bi dao 6× brže generisanje |

Ne stajemo jer smo ostali bez ideja, nego jer smo **izmjerili gdje su granice** i većina ih je strukturna.

**Plan za preostala dva dana:**

| kada | šta |
|---|---|
| veče dana 3 | statistika na više podataka (Mihajlo), pa **DMD fine-tuning preko noći** |
| dan 4 | testovi i debug DMD-a, rezultati, **izrada prezentacije**, dublje upoznavanje sa kodom |
| dan 5 | završetak, proba izlaganja dvaput sa mjerenjem vremena, peglanje |

**Prezentacija:** 10 minuta, 5 minuta pitanja, ~10 slajdova. Skica postoji. Dvije vodilje: demo ide **rano** (ne kao kruna na kraju — ako ponestane vremena, bolje izgubiti zaključak nego demo), a ograničenja se prijavljuju sa **izmjerenim brojevima**, jer to djeluje ozbiljnije nego prećutati ih.

**Za upoznavanje sa kodom** (najkraći put, ~2 sata): `bair_dataset.py` (81 linija, nosi ključnu odluku o poravnanju akcija) → `git diff a31b657 -- Wan21/` (cijela naša izmjena repo koda, 15 linija) → `train_lora_action.py` (srce svega) → `CLAUDE.md` (za "zašto" umjesto "kako").

---

## Kraj dana 3 — statistika i pokretanje DMD-a

**Monte Carlo statistika (Mihajlo, 42 min, 256 pokušaja).** Naš raniji `rollout_metrics.py` je
testirao **jednu fiksnu komandu** dijeljenu preko cijelog batch-a (npr. "svi idu desno"). Nova
skripta `rollout_metrics_mc.py` daje **svakoj sceni sopstvenu nasumično izvučenu akciju po
bloku**. To uklanja konfaund koji nas je već bockao: dva demo klipa dala su 50% (up-up-down-down)
naspram 33% (right ×6), što je nagovještavalo da izbor komande utiče na rezultat.

Nalaz potvrđuje ono što smo prijavili mentorima, ali sada na osam puta većem uzorku i sa
intervalima povjerenja: FVD*/FID rastu monotono po dubini, kontrola pada nakon **prvog**
samogenerisanog bloka pa se izravna. Bitno: intervali za dubinu 2 i 3 se **preklapaju
preširoko** da bi se razlika smjela zvati razlikom — ono što je ranije bila ograda ("nerazlučivo
pri n=32") sada je izmjerena tvrdnja.

Smoke test mu je usput uhvatio pravi bug prije punog runa: ocjenjivanje nasumično-komandovane
generacije protiv **prave** (drugačije komandovane) budućnosti davalo je besmislen PSNR (11.2).
Riješeno drugim prolazom uslovljenim stvarnom snimljenom akcijom, koji se koristi **samo** za
PSNR/SSIM; direction accuracy i FVD*/FID ostaju na nasumičnoj komandi, jer je to ono što mjere.

**DMD fine-tuning pokrenut u 21:54**, traje do ~05:45. Drugi, potpuno odvojen model — postojeći
se ne dira, cilj je da sutra imamo **dva** za poređenje.

Jedna ispravka pojma koja se ponavljala: **DMD nije manji model.** Ista arhitektura, isti 1.3B,
checkpoint 5.55 GB. Destilovano je **uzorkovanje** — obučen je da put od šuma do slike pređe u 4
velika skoka umjesto ~50 sitnih. Prečica kroz putanju, ne manja mreža. Ono što dobijamo je
brzina: 4 koraka umjesto naša 24, što bi interaktivni demo dovelo ispod sekunde po pritisku.

Zašto je ishod neizvjestan, i zašto je zato zanimljiv: naš gubitak uči model da predviđa **pravi
tok** — a to je tačno ono od čega ga je destilacija odučila. Ako kontrola proradi, moguće je da
smo mu vratili pravi tok i time pokvarili prečicu zbog koje smo ga i uzeli. Zato se sutra testira
**i sa 4 i sa 24 koraka**: ta razlika razdvaja "radi" od "poništili smo destilaciju". Sva tri
ishoda su izvještajna.

Konfiguracija je namjerno **identična** velikom treningu osim baznog modela, da bi razlika u
rezultatu bila pripisiva isključivo destilaciji. Isti W&B projekat, pa oba runa stoje u istom
grafiku.

**Trenutno stanje:** DMD trening radi, sve ostalo na `main`, `CLAUDE.md` preko 1500 linija sa
svakom odlukom i obrazloženjem.

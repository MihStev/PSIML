# Priprema za izlaganje — teorijska osnova i priča (dan 5)

Interna priprema za Mihajla i Dawidzarda. Slajdovi su u `/home/mls10/presentation/`
(`BAIR_LoRA_Presentation.pptx` + `arm_control_panel.html`, skinuti OBA u isti folder).
Sve brojke ovdje su iz mjerenja zabilježenih u `CLAUDE.md` — ovo je destilat za pričanje,
ne novi izvor istine.

---

## 1. Priča u 90 sekundi (elevator pitch, uvježbati napamet)

> Robot koji umije da ZAMISLI posljedicu svog poteza može da planira prije nego što ga povuče.
> Video difuzioni modeli već znaju kako svijet IZGLEDA — ali ne slušaju komande. Mi smo uzeli
> pretreniran video model (Wan2.1, 1.3B), zamrzli ga, i kroz LoRA + mali action encoder (19M
> parametara, ~8h na jednoj A100) naučili ga da sluša akcije pravog robota iz BAIR dataseta.
> Rezultat: promjena jednog broja u akcionom vektoru pomjera ruku u tačno komandovanom smjeru
> u praktično 100% slučajeva na neviđenim scenama — i to smo dokazali mjerenjem koje ne može
> da se prevari kopiranjem konteksta. Ograničenje smo izmjerili jednako pošteno: kontrola drži
> jedan predviđeni blok, pa opada — poznat mehanizam (exposure bias), sa poznatim lijekom koji
> upravo treniramo. Isti model čitan unatrag bira akciju koja vodi ka zadatom cilju — od
> forward modela ka planiranju.

Tri poente koje priča mora da pogodi: (1) kontrola je DOKAZANA i kvantifikovana,
(2) mjerili smo pošteno uključujući vlastite greške, (3) ograničenja znamo MEHANIZMOM, ne
samo simptomom.

---

## 2. Teorijska osnova — šta moramo umjeti objasniti (po temama)

### 2.1 World modeli i zašto video difuzija
- World model = naučena aproksimacija dinamike okruženja: `s_t, a_t → s_{t+1}`. Za robota:
  "šta se desi ako uradim X" bez stvarnog izvršavanja — jeftino, bezbjedno, paralelizabilno.
- Video difuzioni model je world model u pikselskom/latentnom prostoru: pretreniran zna
  izgled i fiziku svijeta "uopšteno", fine-tuning ga specijalizuje na naš domen + akcije.
- Forward dynamics (naše): akcija → posljedica. Inverse dynamics (goal search demo): ista
  mreža, pretraga po akcijama koja minimizuje udaljenost do ciljnog frejma. Danilova
  formulacija: "two views of the same model".

### 2.2 Difuzija / flow matching — minimum koji treba znati
- Model uči da OD ŠUMA ka podacima prevede uzorak; parametrizacija ovdje je flow/velocity
  (model predviđa smjer kretanja kroz "vrijeme" šuma), iz čega se izvodi x0-predikcija.
- Naš sampler u generaciji: **x0-predikcija → ponovno zašumljavanje na sledeći sigma →
  ponovi** (24 koraka standardno). Euler ODE je probano i LOŠIJE — akumulira grešku bez
  korekcije (latenti odlutali na ±70; x0-pristup svaki korak vraća na manifold podataka).
- Broj koraka: izmjeren prelet 1→12 koraka — FID pada glatko (86→54), koljeno oko 2-4,
  zasićenje 8-12. PSNR RAVAN kroz cijeli opseg (nagrađuje zamućenje — reci ovo ako neko
  pita zašto ne prijavljujemo PSNR po koracima).

### 2.3 Latentni prostor i VAE (ključ za "zašto je mutno")
- Video se ne generiše u pikselima nego u latentima kauzalnog VAE-a: 29 sirovih frejmova →
  8 latentnih (1 + 4×7 — frejm 0 poseban, potom po 4). Latent 8×8×16 po frejmu na 64×64.
- **VAE plafon: 22.74 dB PSNR** — dekodiranje PRAVOG latenta vs sirovi pikseli. Nijedna
  generacija ne može preko toga. Naš najbolji: 18.56 dB = **81% plafona**.
- Vizuelna dekompozicija: najveći pad oštrine je sirovo→VAE (bez ikakvog generisanja!),
  model gubi MANJE nego autoenkoder. Zato "veća rezolucija" nije estetika nego podizanje
  plafona — i zato SR (ESRGAN) NE ulazi ni u jedan broj (izmišlja detalje, nije naš model).
- Preostala razlika do plafona dijelom je nesvodiva: model predviđa BUDUĆNOST koja je
  stvarno neizvjesna, a x0-predikcija usrednjava moguće ishode → zamućenje svojstveno zadatku.

### 2.4 DiT + AdaLN — kako akcija ulazi (najvažnije metodsko pitanje)
- DiT blokovi se kondicionišu kroz AdaLN: timestep embedding se projektuje u 6 modulacionih
  vektora (shift/scale/gate) PO BLOKU. Zbog diffusion-forcing treninga magistrala je
  **po-frejmu** (`[B, F, 6, dim]`) — svaki latentni frejm ima svoj noise-level embedding.
- Mi dodajemo akcioni embedding NA TU magistralu, prije `time_projection` — dakle akcija
  moduliše svih 30 blokova, per-frame. Nikakav novi mehanizam pažnje nije trebao.
- ActionEncoder: `16 → 256 → 256 → 1536` (SiLU), **zadnji sloj zero-init** → na koraku 0
  model je IDENTIČAN pretreniranom, ništa se ne kvari startom.
- Ulaz 16 = 4 uzastopne akcije × 4D, FLATTEN (ne prosjek! akcije su delte — prosjek
  "lijevo pa nazad" ≈ 0, signal bi nestao; ovo je bio kritičan dizajn-ispravak).
- Izmjereno da signal NIJE šapat: `||emb(right) − emb(left)|| ≈ 317%` od `||e||` —
  ravnopravan/dominantan doprinos modulaciji.
- Zašto ne text-slot/cross-attention: tekst je globalan po klipu (jedna rečenica za cijeli
  video) — akcije su per-frame; AdaLN kanal per-frame VEĆ postoji, pa je to najmanji i
  najprincipijelniji rez. (Repo README: cross-attention condition injection je tek planiran.)

### 2.5 LoRA — zašto i šta smo naučili
- Umjesto svih 1.3B parametara, uči se niskorangovna dopuna `ΔW = BA` na q,k,v,ffn.0,ffn.2.
  Rank 16 → 18.9M trenable. Backbone zamrznut.
- Izmjereno: **rank (8/16/64) ne mijenja ni brzinu ni memoriju na ovoj skali** (15.35 vs
  15.75 GB; 1.28 vs 1.30 s/korak) — izbor ranka je o kapacitetu/overfittingu, ne o trošku.
- Novo znanje "akcija→dinamika" primarno živi u ActionEncoderu; LoRA adaptira stil/domen.
  (Zato je i DMD baza dala isti rezultat — vidi 2.8.)

### 2.6 Teacher forcing, blokovi i KV-cache
- Trening: model UVIJEK vidi čist (pravi) kontekst + zašumljenu metu, joint forward
  (`generator_loss`), timestep zajednički po bloku od `num_frame_per_block=4`.
- Block-causal maska: zašumljeni blok vidi čiste frejmove SAMO prethodnih blokova — nema
  trivijalnog prepisivanja, zadatak je legitiman.
- Inferenca teče blok-po-blok autoregresivno sa KV-cache-om. BAIR klip = 8 latentnih
  frejmova = tačno 2 bloka.

### 2.7 Exposure bias — mehanizam raspada rollout-a (znati ispričati bez papira)
- Model je viđao samo pravi kontekst. U slobodnom rollout-u jede SOPSTVENI izlaz — blago
  zamućen, van distribucije — i tretira ga kao istinu. Greška se množi: zamućen kontekst →
  dvosmislenija scena → model usrednjava više budućnosti → još zamućenije.
- Izmjereno: smjer 96.9% → 53.1% (slučajnost) poslije TAČNO JEDNOG samo-generisanog bloka,
  pa ravno ~59-69%; FVD* 106→184 kroz dubine 1-6, monotono.
- Dvije metodološke zamke koje smo SAMI uhvatili (reći otvoreno, jače je nego sakriti):
  (1) preživljavačka pristrasnost — detektor ruke otkazuje na degradiranim rollout-ima i
  scene su ispadale iz uzorka → prividni "oporavak"; ispravno: nepraćeno = neuspjeh.
  (2) divergencija sama je varljiva — RASTE s dubinom dok tačnost PADA (oba rollout-a
  nezavisno lutaju). "Različito" ≠ "različito na tačan način".
- Lijekovi, po cijeni: `noise_augmentation` (jedna linija konfiga + retrening),
  **scheduled sampling** (Danilov prijedlog — trening u toku preko noći, vidi 2.9),
  pravi Stage 2 self-forcing (puna faza, ~30h, van roka — svjesno prihvaćen trade-off,
  mentor je Stage 1 i preporučio kao "safer, ali kratak rollout").

### 2.8 DMD / destilacija — šta jeste a šta nije
- DMD NE smanjuje model — ista arhitektura, 1.3B. Destilovano je UZORKOVANJE: put od šuma
  do slike u 4 velika skoka umjesto ~50; trening drži TRI transformera (generator,
  fake_score, real_score), zato je pravi DMD trening ~3x skuplji, ne jeftiniji.
- Naš nalaz: LoRA preko DMD baze = LoRA preko obične baze (val krive nerazlučive, ±1.4%;
  finalni FID 2x2 tabela: razlika baza 0.2-0.8% — ispod šuma). Ali **4-koračno uzorkovanje
  poslije fine-tuninga i dalje radi** (čak blago bolje od 24 na PSNR/SSIM/smjeru) —
  **destilacija NIJE pokvarena** našim flow-matching gubitkom. To je bio glavni strah.
- Provjera težina (na zahtjev): 885 tenzora, 0 identičnih, medijana rel. razlike 0.63% —
  destilacija je MANJA perturbacija nego naš fine-tuning (18.9M). Otud identične krive.
- Nedkova hipoteza (DMD bolji u rollout-u zbog self-forcing treninga): testirano MC
  rollout-om u režimu gdje se prednost MOGLA pokazati — intervali povjerenja se preklapaju
  na sve tri dubine. Nije potvrđena; moguće da ju je naš teacher-forced fine-tuning odučio.

### 2.9 Scheduled sampling (noćni eksperiment — status reći ujutro)
- Ideja (Danilo; klasika Bengio et al. 2015): sa vjerovatnoćom p=0.5 kontekst se zamijeni
  SOPSTVENOM predikcijom modela (jedan forward bez gradijenta, t=500), meta ostaje prava.
  Model uči da predvidi TAČNO čak i kad je ulaz nesavršen.
- NAŠA aproksimacija, ne pravi self-forcing: BAIR klip ima samo 2 bloka, nema kud dalje
  da se kotrlja — kontekst KVARIMO onako kako ga model sam kvari, umjesto kotrljanja.
- Trošak izmjeren: 4.74 s/korak (+37%), 4000 koraka preko noći.
- **Val loss NE MOŽE reći je li uspjelo** (mjeri teacher-forced predviđanje, cilj je
  rollout) — presuda je jutarnji MC rollout test (256 pokušaja, dubine 1-3, isti seed kao
  za običan/DMD). Poređenje čeka u CLAUDE.md tabeli.

### 2.10 Metrike — zašto baš ove i gdje lažu
- **PSNR/SSIM**: vjernost rekonstrukcije; NAGRAĐUJU zamućenje i kopiranje statične pozadine
  (BAIR: fiksna kamera!) — NE dokazuju kontrolu. FID: per-frame realizam distribucije,
  vremenski SLIJEP (Danilova primjedba). FVD*: naš surogat (S3D umjesto kanonskog I3D,
  koji nije instalabilan ovdje) — striktno RELATIVNA mjera; 64×64 → 224 upsampling znači
  da mreža dijelom gleda artefakte uvećavanja; malo uzoraka naduvava Fréchet procjenu
  (izmjereno: 32→256 scena spustilo FVD* 106→76 BEZ promjene modela).
- **Action-swap divergencija**: ista scena, isti šum, mijenja se SAMO akcija; L1 razlika
  generisanih dijelova. Ugrađena kontrola: kontekst dio mora biti 0.00 (i jeste, svuda).
  **Prag šuma 15.35** (ista akcija dvaput — sampler ima `randn` na međukoracima): naša
  divergencija ~42-44 je ~2.7x iznad praga, ali se broj NE prijavljuje bez tog konteksta.
- **Smjer**: relativno (upareno: da li "right" završi desnije od "left" — 100%/99.6%) vs
  apsolutno (da li se ruka pomjerila u komandovanom smjeru — ~87.6%). Razlika NIJE bug:
  apsolutno mjerenje dijeli varijansu sa dinamikom scene; upareno je čisto poređenje.
- **Delta-PSNR ljestvica** (Nedkov prijedlog + naše kontrole): prava akcija 18.56 / null
  13.27 / pogrešna 12.45 / bez fine-tuninga 7.12. Čista vrijednost informacije o akciji =
  **5.29 dB** (prava − null); kazna za obmanu 0.82 dB. (Ranije prijavljenih 6.11 dB je
  bilo precijenjeno jer je sadržavalo kaznu — ispravljeno, reći tačnu dekompoziciju.)
- Baseline bez fine-tuninga: dir_rel 51.5% (slučajnost), delta-PSNR −0.03 → **ništa od
  kontrole nije "već bilo u modelu"**.

### 2.11 Robot koordinate (sitnica koja obara na demou ako se zaboravi)
- Robotsko `+x` je LIJEVO u slici (koordinatni sistem robota nije poravnat s kamerom) —
  izmjereno na sirovim pikselima, bez modela. `ACTION_OVERRIDES` to već mapira.
- Podaci su asimetrični po dim1 (+7.69 px dolje vs −3.49 gore) — model je naučio i tu
  asimetriju; ako neko primijeti "down je slabiji", to je SVOJSTVO PODATAKA, ne bag.

---

## 3. Brojevi koje treba znati napamet

| šta | broj |
|---|---|
| backbone / adapter | Wan2.1-T2V-1.3B (dim 1536, 30 blokova) / LoRA r16 = 18.9M + ActionEncoder |
| podaci | BAIR 64×64, 43,264 train sekvenci, 256 test scena, akcija 4D |
| veliki trening | bs 32, 8000 koraka = 5.9 epoha, 7.6h, 3.45 s/korak, 16.7 GB |
| kontrola | dir_rel **100%** od koraka ~1000; **99.6%** (255/256) na punom test skupu |
| apsolutni smjer | ~87.6% (84-92% kroz checkpointe) |
| prag kontrole | ~32,000 uzoraka (≈0.74 epohe) — u UZORCIMA, ne koracima |
| vjernost | PSNR 18.56 / SSIM 0.78 / FID 11.1 (256 scena, 24 koraka) |
| VAE plafon | 22.74 dB → mi na 81% |
| ljestvica | 18.56 prava / 13.27 null / 12.45 pogrešna / 7.12 bez FT → **+5.29 dB** akcija |
| divergencija | ~42-44 vs prag šuma 15.35 (~2.7x) |
| rollout | smjer 96.9%→53.1% poslije 1. samo-generisanog bloka; FVD* 106→184 (dubine 1-6) |
| MC rollout (apsolutno) | 78.9% / 71.9% / 74.2% (dubine 1-3, n=256, CI ±5%) |
| DMD | baze nerazlučive; 4 koraka ≈ 24 koraka poslije FT (destilacija preživjela) |
| CFG | ne isplati se: w=3 daje +5% smjera za −4.6% PSNR — kontrola već zasićena na w=1 |
| brzina demoa | ~4.5 s/blok @24 koraka; 4-koračno ispod sekunde |

---

## 4. Očekivana pitanja i odgovori (Q&A priprema)

1. **"100% zvuči predobro — šta tačno mjerite?"** Upareno poređenje: ista scena, isti šum,
   samo akcija zamijenjena; pitamo da li "right" varijanta završi desnije od "left". Na
   punom test skupu 255/256. Apsolutna mjera (komandovani smjer pogođen) je ~87.6% — obje
   prijavljujemo. Kontrole: kontekst-divergencija tačno 0.00 (mjerenje čisto), baseline bez
   fine-tuninga 51.5% (slučajnost), prag šuma samplera izmjeren i odbijen.
2. **"Da li model samo pamti trening?"** Sve mjere su na test splitu koji nikad nije
   treniran; val gap na kraju +0.0027 (2.6%); kontrola se zasiti već na 0.74 epohe.
3. **"Zašto je mutno?"** Većina zamućenja je VAE na 64×64 (plafon 22.74 dB, mi na 81%);
   ostatak dijelom nesvodiva neizvjesnost budućnosti (x0 usrednjava ishode). SR bi bio
   laž — ESRGAN izmišlja detalje, pa je u demou jasno označen i ne ulazi u brojeve.
4. **"Zašto AdaLN injekcija a ne cross-attention?"** Per-frame kanal već postoji (diffusion
   forcing), akcije su per-frame, zero-init čuva pretrenirano znanje; cross-attention bi
   bio novi mehanizam bez per-frame granularnosti po defaultu. Izmjereno da signal dominira
   modulacijom (317% od ||e||).
5. **"Zašto rollout puca i šta biste uradili?"** Exposure bias — mehanizam + brojevi u 2.7;
   lijek po rastućoj cijeni: noise-aug (1 linija), scheduled sampling (trening SINOĆ
   pokrenut, rezultat jutros), pravi Stage 2 self-forcing (van 5-dnevnog roka).
6. **"Šta je DMD donio?"** Ništa kvalitetu — ali to JESTE nalaz: destilacija je manja
   perturbacija od našeg fine-tuninga, a brzo 4-koračno uzorkovanje je PREŽIVJELO
   fine-tuning (glavni rizik opovrgnut). Nedkova rollout hipoteza testirana i neposvrđena
   (CI preklopljeni).
7. **"Zašto 64×64 / zašto ne veća rezolucija?"** BAIR je nativno 64×64 (puna rezolucija
   nikad javno objavljena — provjereno na Berkeley serveru, bajt u bajt); upsampling ne
   dodaje informaciju. Pravi put je RoboNet 128 (isti roboti, 162k epizoda) — u future work.
8. **"Koliko vam vjerujem na FID/FVD?"** Malo uzoraka + 64→224 uvećavanje → apsolutne
   vrijednosti neuporedive s literaturom, koristimo ih ISKLJUČIVO relativno između naših
   checkpointa/dubina. FVD* je S3D surogat i tako je označen.
9. **"Goal search — radi li?"** 15/20 slaganje znaka (x: 6/10, y: 9/10); promašaji skoro svi
   na scenama gdje je stvarna akcija ~0 (znak besmislen); filtrirano |a|>0.01 → 12/14, ali
   to je naknadno filtriranje i glavni broj ostaje 15/20. Bira tačan smjer, pretjeruje u
   jačini. Pošten test: cilj je epizodina SOPSTVENA budućnost, prava akcija = ground truth.
10. **"Šta je trebalo više vremena?"** Stage 2 self-forcing za dug rollout; RoboNet 128 za
    plafon; više scena/seedova za uže intervale; FVD sa kanonskim I3D.

---

## 5. Šta NE tvrditi (ograde — kredibilitet je glavni adut)

- NE "model crta pun kvadrat" — generišu se 3/4 kvadrata (prva stranica je na kontekst
  frejmovima).
- NE "FVD" bez zvjezdice — naš je S3D surogat, samo relativan.
- NE divergenciju bez praga šuma (15.35); NE delta-PSNR 6.11 (ispravno: 5.29 + 0.82).
- NE "DMD ne valja" ni "DMD bolji" — nerazlučivo, sa objašnjenjem zašto.
- NE "kontrola se oporavlja s dubinom" — to je bila preživljavačka pristrasnost.
- NE tvrditi da je scheduled sampling uspio/propao PRIJE jutarnjeg MC testa.
- SR/ESRGAN uvijek označen kao tuđi model za prikaz, nikad u brojevima.
- "2500 iteracija minimum" nije prenosiva jedinica — prag je u UZORCIMA (~32k).

---

## 6. Jutarnja checklista (prije ažuriranja prezentacije ~8:00)

1. `git pull` (Dawidzardov agent gura noćno ažuriranje konteksta ujutro — NE dirati
   prezentaciju prije pull-a).
2. Pročitati novu CLAUDE.md sekciju: rezultat noćnog scheduled sampling treninga +
   jutarnjeg MC testa (`bair_lora_selfpred/step_4000.pt`, 256 pokušaja, dubine 1-3).
3. Ako selfpred POBOLJŠAVA rollout → ažurirati slajd "Next" ("already running" → rezultat)
   i eventualno "Limitations" (dodati treću kolonu u 97→53 priču). Ako NE → takođe
   izvještajno: "izmjerili smo i aproksimaciju lijeka, treba pravi Stage 2".
4. Proći kroz pptx u pravom PowerPointu: widget klikovi (slajd 8), hyperlink na panel,
   šala-slajd, preklapanja (ovdje nema renderera — geometrija je računata, ne viđena).
5. Dvije probe s mjerenjem vremena (plan dana 5); pitanja iz sekcije 4 podijeliti ko šta
   odgovara.

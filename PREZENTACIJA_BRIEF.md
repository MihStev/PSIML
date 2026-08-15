# Brief za izradu prezentacije — svi brojevi, tvrdnje i materijal na jednom mjestu

**Namjena:** ovaj fajl je dovoljan da se napravi prezentacija bez čitanja cijelog `CLAUDE.md`.
Svaki broj je izmjeren; gdje postoji ograda, ona je navedena uz broj a ne zasebno.

**Format izlaganja:** 10 minuta + 5 minuta pitanja, do 10 slajdova.
**Struktura po Danilovom prijedlogu:** 1) motivacija, 2) ciljevi, 3) metod, 4) rezultati,
5) demo i ograničenja.

---

## 1. Motivacija

Modeli svijeta omogućavaju robotu da **zamisli posljedicu svoje akcije** prije nego što je izvede.
Video modeli već umiju da predvide sledeće frejmove iz tekućeg; pitanje je može li se to
predviđanje **voditi akcijom** umjesto tekstom ili pozicijom kamere.

Naše pitanje: **možemo li pretrenirani video difuzioni model natjerati da posluša komandu
robotu, i koliko precizno?**

## 2. Ciljevi

- Dodati **akciono kondicioniranje** u pretrenirani video model, LoRA fine-tuningom
- Izmjeriti **koliko precizno** model prima akciju
- Utvrditi **gdje pristup otkazuje** i zašto

## 3. Metod — dva pitanja koja publika sigurno postavlja

**Model:** Wan2.1-T2V-1.3B kroz minWM (autoregresivni kauzalni video model).
Ulaz: 4 latent frejma konteksta + akcija. Izlaz: sledeća 4 latent frejma (16 piksel frejmova).
**Dataset:** BAIR robot pushing, 64×64, 216 325 trening zapisa, **256 neviđenih test scena**.

**Pitanje A — kako su građeni action embeddingi?**
BAIR daje **30 akcija po epizodi**, jednu po prelazu između frejmova. Udruživanje u jedan vektor
je **diskvalifikujuće, ne samo lošije**: akcije su pomjeraji, pa je prosjek kretanja naprijed-nazad
≈ 0 i model nema signal. Zato kondicioniramo **po latent frejmu**: latent *i* nosi 4 sirove akcije
koje ga pokrivaju, **spljoštene, ne usrednjene** → (F, 16) → `ActionEncoder 16→256→256→1536`.
Posljednji sloj je **zero-init**, pa se na koraku 0 model ponaša tačno kao pretrenirani.

**Pitanje B — kako se generisanje uslovljava akcijom?**
Kroz **postojeću** per-frejm timestep/AdaLN putanju, ne kroz tekst:

```
e = time_embedding(t) + action_embed  →  time_projection  →  AdaLN (shift/scale/gate, svaki DiT blok)
```

**Ukupna izmjena upstream koda: 15 linija u 3 fajla.** Trenira se 18.92M LoRA parametara
(`q,k,v,ffn.0,ffn.2`, rank 16) + 0.46M action encoder = **19.4M nad 1.3B modelom (1.4%)**.

## 4. Rezultati — voditi sa kontrolom

### GLAVNI REZULTAT (256 neviđenih scena, finalni checkpoint)

| metrika | vrijednost |
|---|---|
| **relativna tačnost pravca** | **99.6%** (255/256) |
| apsolutna tačnost pravca | 84.8% |
| **delta-PSNR (upravljivost)** | **5.29 dB** |
| PSNR / SSIM / FID | 18.56 / 0.785 / 11.12 |
| divergencija na kontekst frejmovima | **0.00** (ugrađena kontrola) |

> **Ograda koja MORA ići uz 99.6%:** na 64 scene je bilo 100%; na punom skupu jedna scena pada.
> Prijaviti 99.6%, ne birati podskup na kojem je 100%.

### LJESTVICA UPRAVLJIVOSTI — najjači pojedinačni slajd

| model dobija | PSNR | razmak znači |
|---|---|---|
| **pravu** akciju | 18.56 | |
| **null** akciju (ima mašineriju, nije obaviješten) | 13.27 | ↑ **5.29 dB = čista vrijednost informacije** |
| **pogrešnu** akciju | 12.45 | ↑ 0.82 dB = kazna za obmanu |
| **bez fine-tuninga** (pretrenirani model) | 7.12 | ↑ 5.33 dB = šta je fine-tuning donio |

Pretrenirani model bez nas: PSNR **7.12**, FID **229**, tačnost pravca **51.5% (slučajnost)**,
delta-PSNR **−0.03**. **Ništa od izmjerene kontrole nije bilo "već tu".**

> Ranije smo prijavljivali delta-PSNR 6.11 dB. **To je bilo precijenjeno** — sadržavalo je 0.82 dB
> kazne jer pogrešna akcija model aktivno gura u krivo. Ispravan broj je 5.29 dB.

### NALAZ 1 — kontrola i vjernost sazrijevaju na različitim skalama

Tačnost pravca dostiže 100% na **koraku 1000** i ostaje ravna; PSNR/SSIM/FID rastu do 8000.
Vidjeli smo to samo zato što smo evaluirali **svih 16 checkpointa**.
Izraženo u **uzorcima, ne koracima** (koraci nisu prenosivi između batch veličina):
kontrola se zasićuje na **~32 000 uzoraka** ≈ 0.74 epohe.

### NALAZ 2 — VAE postavlja plafon zamućenja, ne model

Dekodiranje **pravog** latenta nazad u piksele — bez ikakvog generisanja — daje **22.74 dB**.
Naših 18.56 je **81% dostižnog**. Dobar VAE na nativnoj rezoluciji ide preko 30 dB; naš je nizak
jer 64×64 daje **8×8 latent**, daleko izvan režima za koji je enkoder pravljen.
**Ovo je kvantitativni argument da veća rezolucija diže PLAFON**, a ne kozmetička želja.

### NALAZ 3 — broj koraka uzorkovanja je skoro besplatan

| koraka | 1 | 2 | 4 | 8 | 12 | 24 |
|---|---|---|---|---|---|---|
| FID | 86.5 | 65.7 | 60.1 | 55.1 | 54.0 | 52.7 |

Nema praga ni provalije; kriva je **zasićena već na 8–12 koraka**. Od 4 do 24 koraka dobija se
12% FID-a za **šest puta više računa**. PSNR je ravan kroz cijeli opseg (19.0–19.7) — još jedna
potvrda da PSNR ovdje nagrađuje zamućenje, ne kvalitet.
**Praktična posljedica: 6× brži demo, besplatno.**

### NALAZ 4 — kontrola PREŽIVLJAVA rollout; lokalizacija ne

Iz istog samogenerisanog konteksta, suprotne akcije, 64 scene:

| model | dubina | odnos divergencije prema šumu | relativna tačnost |
|---|---|---|---|
| običan | 1 / 2 / 3 | 3.13 / 2.25 / 1.89 | 1.000 / 0.984 / 1.000 |
| selfpred | 1 / 2 / 3 | 3.30 / 2.42 / 2.07 | 1.000 / 1.000 / 0.984 |

**Poslije tri samogenerisana bloka "desno" i dalje završava desno od "lijevo", u 98–100% slučajeva.**

Monte Carlo rollout je na istim dubinama davao 0.789 / 0.719 / 0.742 — **nije protivrječnost**:
MC daje svakoj sceni sopstvenu nasumičnu akciju i pita je li se ruka pomjerila **apsolutno** tako;
ovaj test pita **razdvaja li** model suprotne komande. Razlika između ta dva broja JESTE nalaz:
**model posluša akciju, ali mu scena odluta.** Problem je **drift scene**, ne kondicioniranje.

> **Metodološka pouka za slajd:** prag šuma samplera **raste sa dubinom** (15.35 → 23.16 → 24.53),
> jer sampler vuče svjež šum na svakom međukoraku. Sirova divergencija ide 48 → 52 → 46 i izgleda
> ravno; očišćena od šuma pada 3.13 → 1.89. **Bez mjerenja praga po dubini zaključili bismo da
> kontrola JAČA.** Prag na dubini 1 (15.35) nezavisno se reprodukovao drugom skriptom.

### NALAZ 5 — dva režima otkazivanja su ODVOJIVA (scheduled sampling)

Trening gdje 50% uzoraka dobija **modelov sopstveni** kontekst umjesto pravog:

| model | FID kroz dubine 1→3 | porast |
|---|---|---|
| običan | 35.35 → 54.31 → 69.56 | **+97%** |
| scheduled sampling | 35.29 → 47.71 → 55.52 | **+57%** |

**Raspad slike je prepolovljen. Tačnost pravca nepromijenjena.**
Dakle izloženost (exposure bias) objašnjava **degradaciju slike**, ali ne i gubitak apsolutne
pozicije. Cijena: PSNR na dubini 1 pada 17.26 → 15.78, dok FID ostaje isti — slike izgledaju
jednako uvjerljivo ali se slabije poklapaju sa istinom. **Razmjena, ne pobjeda.**

> Ovo je **aproksimacija** self-forcinga: BAIR klip ima 8 latent frejmova = tačno 2 bloka, pa
> nema trećeg bloka sa istinom u koji bi se rollout produžio.

### IZMJERENI NEGATIVNI REZULTATI — prijaviti, ne prećutati

**CFG ne pomaže.** Trenirali smo null embedding, sweep w ∈ {1, 1.5, 2, 3}: čista razmjena
vjernosti za kontrolu, ali je relativna kontrola **već zasićena na 100%** pri w=1, pa CFG
pojačava ništa uz cijenu vjernosti i drugog prolaza.

**Destilacija nije potrebna.** Fine-tuning na destilovanom (DMD) checkpointu daje iste brojeve
kao na običnom:

| baza | @4 koraka | @24 koraka |
|---|---|---|
| obična | FID 16.80 | FID 11.12 |
| destilovana | FID 16.80 | FID 10.95 |

Razlika između **baza** je 0.2–0.8% (ispod šuma), između **broja koraka** ~52%.
Provjerili smo i same težine: 885 tenzora, **0 identičnih**, medijana razlike 0.63% — dakle
jesmo testirali stvarno destilovan model, ali je destilacija **manja perturbacija nego ono što
naš LoRA fine-tuning ionako radi**. Testirano i u režimu slobodnog rollouta, gdje je razlika
mogla da se pokaže (destilovani je treniran self-forcingom) — intervali se preklapaju na svakoj
dubini.

### NALAZ 6 — INVERSE DYNAMICS: model bira akciju zamišljanjem posljedice

Danilova ideja, `lora_action/goal_action_search.py`. Dati kontekst i **ciljni frejm**, uzorkovati
mrežu 6×6 kandidat-akcija, predvidjeti blok za svaku **u jednom batchovanom prolazu sa istim
šumom**, i izabrati onu čija je zamišljena budućnost najbliža cilju.

**Pošten test, ne trik:** cilj je epizodina SOPSTVENA snimljena budućnost, pa je njena prava
akcija provjerljiv tačan odgovor.

**Rezultat, 10 scena sa najvećom stvarnom akcijom: slaganje znaka 15/20** (x 6/10, y 9/10).
Slučajnost daje 10/20. Primjer (scena 108): izabrano `dx +0.042 dy +0.014`, stvarno
`+0.035 / +0.012` — pogađa smjer i red veličine, a nikad nije vidio tačan odgovor.

> Tri od četiri promašaja po x su na scenama gdje je stvarni dx ≈ 0.001–0.005, gdje je znak
> besmislen. Ograničeno na komponente > 0.01 ispada 12/14, **ali to je naknadno filtriranje** —
> glavni broj ostaje 15/20.
> Sistematska greška: izabrane akcije su **veće** od stvarnih. Pogađa smjer, pretjeruje u jačini.

**Zašto je ovo najvredniji demo:** pretvara "model svijeta" iz tvrdnje u demonstraciju — model
bira akciju **zamišljanjem posljedica**. Danilo to zove forward naspram inverse dynamics: ista
mreža, dva čitanja.

**Materijal:** `logs/goal_search/goal_search_idx*.png` — po panelu:
`CILJ | najbolja zamišljena | najgora zamišljena | mapa ocjena preko (dx,dy)`.
Mapa je 6×6 (otud pikselizirana): **BIJELO je bolje, PLAVO gore**. Da je ravna, akcija ne bi
uticala. Najčistiji primjeri: `idx108`, `idx134`, `idx204`.

## 5. Demo i ograničenja

### Materijal (sve u `/home/mls10/logs/`)

| šta | gdje |
|---|---|
| **interaktivni demo, sirovih 64×64** | `demo/index.html` (7.13 MB, samodovoljan) |
| **interaktivni demo, 256×256 (SR)** | `demo2/index.html` (5.58 MB) |
| glavni rezultat, 4 akcije | `gen_4actions/v2_idx100_{right,left,up,down,real}.mp4` |
| uvećano — pošteno | `upscaled/v2_idx100_*_x8.mp4` (512) |
| uvećano — Real-ESRGAN | `upscaled/BEST_*_x16.mp4` (1024) |
| uvećano — difuzioni | `upscaled/DIFF_*_x4_then_lanczos.mp4` (768) |
| **uporedni klip pošteno/naučeno** | `upscaled/BEST_rollout_lanczos_vs_esrgan.mp4` |
| poređenje SR metoda | `sr_compare/0_SVE_CETIRI_metode.png` |
| akcija po latent frejmu | `teacherforced__base-obican__4koraka/` |
| slobodni rollout, tri modela | `rollout_free__*/`, `cmp_rollout_*/` |
| goal search (inverse dynamics) | `goal_search/goal_search_idx*.png` |
| rollout, običan naspram selfpred | `cmp_rollout_GORE_obican_DOLE_selfpred.{png,mp4}` |

**Redoslijed za izlaganje:** glavni rezultat (4 akcije) → akcija po frejmu → slobodni rollout
(gdje puca) → nedotrenirani prvi model (kako izgleda ispod praga).

> **Superrezolucija NIJE naš model** i ne ulazi ni u jedan broj. Svi brojevi se odnose na sirovih
> 64×64. Ako se pokazuje uljepšan klip, to mora biti rečeno — najbolje pokazati uporedni.

### Ograničenja — sa brojevima, ne kao izgovor

- **Slobodni rollout degradira.** Svi glavni brojevi su teacher-forced. Kvalitet slike opada
  (FID +97% kroz tri dubine), apsolutna pozicija ruke odluta. Relativna kontrola ostaje.
- **Rezolucija je ograničenje PODATAKA, ne ambicije.** Originalni 512×640 BAIR **nije javno
  dostupan** — Berkeley servi samo 64×64 tar (provjereno preko Content-Length i tar aritmetike).
  Meta bi bila **256×256**, jer tek tamo latent (32×32) ulazi u režim za koji je VAE pravljen.
- **FID/FVD\* na ~1024 frejma / 256 klipova** su ispod konvencije; validni samo kao **relativna**
  poređenja između naših uslova. FVD\* koristi torchvision S3D, ne kanonski Kinetics I3D.
- **Apsolutna naspram relativne kontrole**: 84.8% naspram 99.6%, jer apsolutnu putanju ruke
  vodi i dinamika scene koju akcija ne nadjačava.

### Budući rad — svaka stavka sa razlogom, ne sa željom

1. **Sidrenje scene / duži kontekst** — jer smo izmjerili da otkazuje **lokalizacija**, ne kontrola
2. **Pravi self-forcing** (traži klipove sa više od 2 bloka) — naša aproksimacija je prepolovila
   raspad slike, puna verzija bi trebala dalje
3. **256×256** — sa VAE plafonom od 22.74 dB kao kvantitativnim opravdanjem
4. **Kalibracija jačine akcije** — testirali smo 4 smjera pri jednoj jačini; ne znamo je li model
   naučio kontinuirano preslikavanje ili 4 zapamćena režima

---

## Šta NE tvrditi

- **Ne prodavati kontrolu kao otkriće.** Akciono-uslovljeno predviđanje na BAIR-u je razrađeno
  od 2016–2018. Konstrukcija (zero-init encoder, jak pretrenirani model, 1.4% parametara) je
  bila skoro zagarantovana da proradi.
- **Ne reći "novel metoda".** Poštenije: *"izmjerili smo poređenje koje, koliko znamo, nije
  prijavljeno za ovaj setup"*.
- **Ne citirati divergenciju kao goli broj.** 41.83 sadrži prag šuma od 15.35.
- **Ne poredi FID sa 64 scene i sa 256 scena** — FID je pristrasan na malim uzorcima
  (isti model: 27.2 na 64 scene, 11.12 na 256).
- **Ne čitati val loss** kao mjeru uspjeha ovih eksperimenata — tri puta zaredom nije razlikovao
  modele koji se u rollout-u jasno razlikuju.
- **NE koristiti `cmp_rollout_GORE_obican_DOLE_selfpred` kao dokaz da scheduled sampling
  popravlja sliku.** Snimljen je na JEDNOJ sceni i dubini 6, dvostruko dublje nego što je bilo
  šta izmjereno; oba modela se tamo raspadaju podjednako i klip bi radio PROTIV tvrdnje.
  Tvrdnju prijaviti brojem (+57% naspram +97%, 256 pokušaja, dubine 1-3). Klip je koristan samo
  kao ilustracija KAKO izgleda kad rollout ode — za slajd o ograničenjima.

## Šta JESTE naš doprinos

Ne kontrola — nju je konstrukcija skoro garantovala. Nego **mjerenja koja se ne vide iz jednog
broja**:

1. kontrola i vjernost sazrijevaju na **različitim vremenskim skalama**
2. u rollout-u otkazuju **dvije odvojive stvari**; izloženost objašnjava sliku, ne lokalizaciju
3. **destilacija na ovom zadatku nije potrebna** — 6× ubrzanje se dobija besplatno
4. **VAE plafon** kao kontekst bez kojeg 18.56 dB ne znači ništa

Dva od četiri su negativna. Sva četiri smo mogli prećutati.

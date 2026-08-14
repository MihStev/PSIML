# Priprema za poziv sa mentorom (14.08)

## Prvo saopštiti — šta je novo od zadnjeg izvještaja

1. **DMD fine-tuning je urađen preko noći** (8000 koraka, 7h43min, bez greške), pa evaluiran.
2. **Rezultat je negativan, i to čisto izmjeren.** Destilovana baza ne donosi ništa.
3. **Ali smo usput dobili 6× ubrzanje besplatno** — obični model uzorkuje u 4 koraka jednako
   dobro kao u 24.
4. **Monte Carlo statistika** (256 pokušaja) potvrdila raniji nalaz o padu poslije jednog bloka,
   sada sa intervalima povjerenja.

### Tabela koja nosi razgovor (finalni checkpoint, FID, 64 neviđene scene)

| baza | @4 koraka | @24 koraka |
|---|---|---|
| obična (naša) | 34.10 | **27.2** |
| destilovana (DMD) | 34.37 | 27.41 |

- Razlika između **baza**: 0.2–0.8 % → ispod šuma.
- Razlika između **broja koraka**: ~25 %, pojavljuje se isto na obje baze.
- Relativna kontrola: **100 % u sve četiri ćelije.**

Zaključak: broj koraka je jedina ručica koja nešto radi; destilacija ne doprinosi ničemu.

### Provjera koju smo uradili jer je rezultat bio sumnjiv

Uporedili smo same težine dva bazna checkpointa:
- 885 tenzora u oba, **0 identičnih**
- medijana relativne razlike **0.63 %**, prosjek 5.6 %, max 112 %
- 63/885 tenzora se pomjerilo preko 10 %

Dakle jesmo testirali stvarno destilovan model, ali je **destilacija manja perturbacija nego
ono što naš LoRA fine-tuning ionako radi** (18.9M parametara). Zato su se val krive spojile
za manje od 500 koraka.

---

## Pet pitanja, poređanih po tome koliko odgovor mijenja naš plan

### 1. Zašto destilacija nije donijela ništa? *(jedino koje može poništiti naš zaključak)*

**Pitanje:** izmjerili smo da obični model uzorkuje u 4 koraka jednako dobro kao destilovani.
Je li to očekivano na ovako lakom zadatku — 64×64, teacher-forced pravi kontekst, jedan blok,
statična scena — ili nam govori da naša evaluacija ne hvata ono u čemu bi destilacija bila bolja?

**Naša hipoteza:** zadatak je previše lak da bi se prečica isplatila. Model ima malo toga da
izmisli, pa mu 4 koraka dostaju i bez distilacije.

**Šta mijenja:** ako kaže da nam evaluacija promašuje pravu prednost, mijenjamo mjerenje prije
nego što ovo stavimo na slajd kao negativan rezultat.

### 2. Zašto je self-forcing u repou spojen sa DMD distilacijom?

**Pitanje:** `SelfForcingTrainingPipeline` koristi isključivo `model/dmd.py`; nema samostalne
konfiguracije za self-forcing. Je li distribucioni (DMD) gubitak **neophodan da self-forcing
bude stabilan**, ili bi radio i sa običnim flow gubitkom na jednom transformeru?

**Zašto pitamo:** puna varijanta traži tri transformera (generator uči, real_score zamrznut,
fake_score uči) — to ne staje u naše vrijeme. "Self-forcing lite" na jednom transformeru bi
stao, ako nije naivan.

**Šta mijenja:** ovo direktno odlučuje da li je najvredniji sledeći eksperiment izvodljiv ili ne.

### 3. Je li pad poslije TAČNO jednog samogenerisanog bloka tipičan?

**Pitanje:** kod nas 96.9 % → 53.1 % (praktično slučajno), pa ravno kroz dubine 3–6. Je li ta
oštrina normalna za teacher-forced AR video modele, ili nešto u našem setupu to pogoršava?

**Šta mijenja:** ako je netipično oštro, tražimo uzrok kod sebe umjesto da to prijavimo kao
opštu osobinu pristupa.

### 4. Koji je JEDAN sledeći eksperiment sa najviše informacije po uloženom vremenu?

Naši kandidati, da imamo šta da ponudimo umjesto otvorenog pitanja:
- **self-forcing (makar skraćen)** — jedini koji napada izmjerenu slabost umjesto da doda mjeru
- **prelet po broju koraka do kraja** — ako i 2 koraka rade, to je 12× ubrzanje bez distilacije
- **kalibracija kontrole** — odgovara li model *proporcionalno* na jačinu komande; ovo diže
  tvrdnju sa "razlikuje 4 komande" na "naučio je kontinuirano preslikavanje"
- **rezolucija** — 256×256 jer tek tamo latent (32×32) ulazi u režim za koji je VAE pravljen;
  ograničenje su podaci, ne ambicija (originalni BAIR u visokoj rezoluciji nije javan)

### 5. Je li FVD* sa S3D umjesto I3D prihvatljiv kao relativna mjera?

Kanonski FVD traži Kinetics I3D koji nismo mogli instalirati; koristimo torchvision S3D (isti
Kinetics-400). Prijavljujemo ga kao FVD* i samo kao relativnu mjeru između naših uslova.
**Pitanje:** je li to prihvatljivo, ili treba drugačije nazvati.

---

## Ako pita "šta je vaš doprinos"

Ne kontrola — nju je konstrukcija skoro garantovala (zero-init encoder, jak pretrenirani model,
1.4 % parametara). Nego **tri izmjerena nalaza koja se ne vide iz jednog broja**:

1. kontrola i vjernost sazrijevaju na **različitim vremenskim skalama** (kontrola zaključana na
   1000 koraka, vjernost raste do 8000) — vidjeli smo samo jer smo evaluirali svih 16 checkpointa
2. kontrola pada nakon **tačno jednog** samogenerisanog bloka, potvrđeno na 256 nezavisnih
   pokušaja sa intervalima
3. **destilacija na ovom zadatku nije potrebna** — 6× ubrzanje se dobija besplatno

Dva od tri su negativna, i sva tri smo mogli prećutati.

## Šta NE prodavati kao veće nego što jeste

- Rezultat nije nov; akciono-uslovljeno predviđanje na BAIR-u je razrađeno od 2016–2018.
- Glavna tvrdnja je bila skoro zagarantovana; ono što je bilo neizvjesno (rollout) ispalo je
  negativno.
- 18.40 dB nije impresivno u apsolutnom smislu — zato ide uz VAE plafon (22.74 dB, na 81 % smo).
- FID/FVD* na ~1024 frejma / 256 klipova su ispod konvencije; validni samo relativno.

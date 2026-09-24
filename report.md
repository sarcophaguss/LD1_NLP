# Laboratory 1 — Editor's recommendation

Name: Emilis
Date: 2026-09-24

Run instructions, package versions and saved parameters:

- Run: `python valymas.py` (cleaning + sampling), then `python lda_model.py` (workflow, all stages)
- Packages: pandas 2.3.3, scikit-learn 1.7.2
- Seeds: cleaning/1500-doc sample = 42; 20-headline selection = 7; LDA `random_state` = 42
- LDA settings: `max_iter=15`, `learning_method="batch"`, vocabulary cap `max_features=3000` (actual vocab ended up ~212 terms)

## Initial analysis — separate checkpoint file

See `initial-analysis.md` (unchanged, submitted at the checkpoint).

## 1. Decisions and prediction

Why did you choose these two topic counts?

Buvo pasirinkta A = 4 ir B = 7. 4 temos pasirinktos, nes tokios buvo rastos pradinės grupės atliekant `initial-analysis.md`. 7 temos pasirinktos norint patikrinti, ar didesnis skaičius atskleidžia prasmingus pogrupius, ar tai tiesiog papildomas triukšmas/dublikatai (išsamus rezultatas — 2 sekcijos lentelėje).

Which single preprocessing issue did you test, why, and what benefit/cost did you predict before testing?

Buvo pasirinkta pašalinti pasikartojančias šablonines frazes (pvz.: "analitikų teigimu", "dėl rinkos reakcijos", "— analysts say", "as markets react", "amid concerns"), nes A/B bandymuose pastebėta, kad modelis grupuoja temas remdamasis šiomis frazėmis, o ne tikru antraštės turiniu.

**Prognozė prieš bandymą:** tikėtasi, kad pašalinus šias frazes, temos taps aiškesnės ir labiau atitiks realų turinį (sportas, tech, finansai atsiskirs), nes bendra priesaga nebedominuos žodyno. Rizika — jei kuris nors žodis iš frazės realiai neša prasmę pagrindinėje antraštės dalyje, jo pašalinimas gali prastinti tos temos atskyrimą. Efektą turėtų atskleisti antraštės, kurios A bandyme buvo sugrupuotos kartu vien dėl bendros frazės (pvz., `[46954]` sporto antraštė su „as markets react" ir `[21479]` tech antraštė su „analitikų teigimu") - jei prognozė teisinga, jos turėtų persikelti į temas, geriau atitinkančias jų realų turinį.


## 2. Three-run comparison

| Run | Topic count | Preprocessing | Fixed settings / seed | Evidence of usefulness or problems |
| --- | --- | --- | --- | --- |
| A | 4 | Original | seed=42, max_iter=15 | Tik 1 iš 4 temų (Tema 2 „Technologijos/DI") yra nuosekli ir įvardijama, viso kitos likusios 3 yra mišrios/neaiškios. Temos formuojasi pagal bendras šablonines frazes, o ne pagal realų turinį - dėl to finansų, tech ir sporto antraštės susimaišo vienoje temoje. Papildomai pastebėta, kad lietuviškos sporto antraštės patenka į Temą 2 „Technologijos" vien dėl kalbos, ne turinio. |
| B | 7 | Original | seed=42, max_iter=15 | Padidinus temų skaičių nuo 4 iki 7, problema nedingo - tos pačios šabloninės priesagos vis tiek sujungia nesusijusias temas, tik pasiskirsčiusias per daugiau grupių. Jokios naujos, prasmingesnės temos neatsirado - tai patvirtino, kad 4 temos yra tiek pat geras pasirinkimas kaip ir 7, todėl pasirinktas mažesnis, paprastesnis skaičius. |
| C | 4 | Template suffixes removed | seed=42, max_iter=15 | Žodynas sumažėjo tik nuo 212 iki 200 žodžių. Rezultatas mišrus: vienam dokumentui ([46954], sportas) pagerėjo - jis nebeliko sumaišytas su finansais. Kitam ([21479], tech) pablogėjo - jis prarado anksčiau turėtą švarią „Technologijos" grupę ir pateko į finansų+clickbait mišinį. Bendrai temos liko tiek pat susimaišiusios, tik susimaišymo priežastis persikėlė nuo pašalintų frazių prie kitų pasikartojančių žodžių. Jokios aiškios bendros naudos nenustatyta. |

Concrete headline evidence of the preprocessing effect, including a cost or unexpected outcome if observed:

[46954] (sportas, priesaga „as markets react"): A > Tema 3, 0.94 (sumaišyta su finansais/tech). C > Tema 1, 0.92 (finansų priemaiša dingo). Nauda pasitvirtino.

[21479] (Apple/tech, priesaga „analitikų teigimu"): A > Tema 2, 0.94 (švari tech tema). C > Tema 3, 0.92 (finansų+clickbait mišinys). Prarasta švari grupė.

Išvada: pakeitimas davė ir naudą, ir pablogino - bendras temų aiškumas nepagerėjo, tik pasikeitė, kuris pasikartojantis žodynas dominuoja.

<!--
Reference facts you can use (write the explanation yourself):
- doc_id=46954 "Star player signs a multi-year contract with club as markets react (New York)":
  Run A -> Topic 3 (0.94, mixed with Apple/Central bank); Run C -> Topic 1 (0.92, no longer mixed with finance)
- doc_id=21479 "Apple pristatė naują iPhone... analitikų teigimu (Vilnius)":
  Run A -> Topic 2 (0.94, clean tech topic); Run C -> Topic 3 (0.92, shifted into finance+clickbait mix)
- Vocabulary size: 212 (Run A/B) -> 200 (Run C); 0 new empty-vector documents caused by the change.
-->

## 3. Final model and failure analysis

Galutinio modelio (A bandymas, 4 temos) temų pavadinimai, top žodžiai ir po dvi reprezentacines antraštes žr. Priede A.

Compare at least three of your original 20 headlines with their model outputs.

| Headline ID and text | Actual model output | Why it fails the editor's needs | Possible remedy or inherent ambiguity |
| --- | --- | --- | --- |
| [2246] Messi įmušė įspūdingą įvartį Lygų taurėje | Topic 2 "Technologijos/DI", weight 0.89 | Sporto naujiena priskirta „Technologijos" temai, nors joje nėra nieko bendro su sportu — redaktorius, pasitikėdamas šia žyma, klaidingai nukreiptų straipsnį. Panašu, kad lietuviškos antraštės tiesiog sumetamos į vieną „liekamąją" temą dėl kalbos, ne turinio. | Reikėtų atskirti kalbas (versti arba modeliuoti atskirai lietuviškas ir angliškas antraštes) arba padidinti lietuviško sporto žodyną, kad jis galėtų suformuoti savo temą, o ne pakliūti į tech „kibirą". |
| [34143] Investors hedge inflation risk using index-linked bonds... | Topic 1 "unclear mixed", weight 0.94 | Reali finansų antraštė priskirta temai, kurios top žodžiai (gadget, buy, defensive, coach, tactics) atrodo kaip sportas/clickbait — redaktoriui ši tema neinformatyvi ir nepatikima praktiniam naudojimui. | Panašu, kad tai lemia bendra šablono priesaga „dėl rinkos reakcijos", jungianti antraštę su kitomis nesusijusiomis temomis; padėtų arba priesagos pašalinimas (žr. 2 sekciją), arba tokios nekoherentiškos temos žymėjimas kaip nepatikimos, o ne naudojimas be išlygų. |
| [44637] Perėjimų gandai: jaunas puolėjas... dėl rinkos reakcijos | Topic 2 (0.70) / Topic 1 (0.25) split | Ši antraštė pradinėje analizėje buvo pažymėta kaip dviprasmiška (sportas+verslas), tad pasidalijęs priskyrimas atrodytų tinkamas, bet nei Tema 2, nei Tema 1 realiai nereiškia „sportas" ar „verslas" — tai tik atspindi kalbos/priesagos artefaktus, o ne tikrą temų dvilypumą. | Toks pasiskirstymas neturėtų būti laikomas įrodymu apie realią dvi-temiškumą, kol pagrindinės temos pačios nėra švarios; iki tol geriau tokį atvejį žymėti kaip nepatikimą/nevienareikšmį nei juo remtis. |

Use at least two distinct kinds of problem across these three cases.

### Priedas A: galutinio modelio (A, 4 temos) temos

| Tema | Pavadinimas | Top žodžiai | Reprezentacinės antraštės |
| --- | --- | --- | --- |
| 0 | Neaiški / mišri (finansai + clickbait žodžiai) | concerns, amid, trick, life, forever, change, inflation, earnings, expectations, beat | [17691], [20772] |
| 1 | Neaiški / mišri (sportas + verslas + clickbait žodžiai) | new, reasons, gadget, buy, defensive, explains, coach, game, tactics, happens | [43582], [34143] |
| 2 | Technologijos / DI | workflows, source, learning, open, libraries, machine, accelerate, kompiuterį, pagreitinti, paprastais | [21479], [37136] |
| 3 | Neaiški / mišri (finansai + tech + sportas per bendrą priesagą) | won, believe, new, analysts, happened, ai, apple, iphone, features, unveils | [14813], [20269] |

## 4. Recommendation

Chosen model and preprocessing, evidence for the choice, accepted trade-off, and what still requires human judgement:

Pasirinktas **A bandymas — 4 temos su originaliu teksto apdorojimu** kaip galutinis modelis. C bandymas (šablonų frazių šalinimas) nedavė aiškaus bendro patobulinimo: vienam dokumentui pagerėjo, kitam pablogėjo, o bendras temų aiškumas liko toks pat žemas — todėl pagrįstas sprendimas yra likti prie paprastesnio, originalaus apdorojimo, o ne pridėti sudėtingumo be įrodytos naudos. Padidinus temų skaičių iki 7 (B bandymas) taip pat nepagerėjo — modelis tik fragmentavosi į daugiau panašiai susimaišiusių grupių, todėl 7 temos neduoda pakankamai naudos, kad pateisintų papildomą sudėtingumą.

Priimtas kompromisas: iš 4 galutinio modelio temų tik viena (Tema 2, „Technologijos/DI") yra nuosekliai interpretuojama; likusios trys yra mišrios arba neaiškios. Tai reiškia, kad modelis šiuo metu **tinka tik kaip preliminarus, žmogaus prižiūrimas įrankis**, o ne kaip savarankiška antraščių maršrutizavimo sistema — jo išvestimi negalima aklai pasitikėti dėl šablonais ir kalba paremto grupavimo (žr. 3 sekcijos nesėkmės atvejus).

Kas vis tiek reikalauja redaktoriaus sprendimo: (1) antraštių, priskirtų mišrioms/neaiškioms temoms (0, 1, 3), tikroji tema turi būti nustatyta rankiniu būdu; (2) lietuviškos antraštės apskritai negali būti patikimai klasifikuojamos šiuo modeliu dėl kalbos nulemto grupavimo; (3) dviprasmiškos antraštės (pvz., `[44637]`) reikalauja žmogaus sprendimo, nes modelio pateiktas temų svorių pasiskirstymas neatspindi realios dviprasmybės priežasties.

## 5. Sources and AI use

Sources and reused code. AI tools and purposes, how outputs were checked, and one suggestion verified/corrected/rejected; otherwise state that no AI was used. The initial interpretation and defence must be completed without AI assistance.

Šaltiniai: scikit-learn dokumentacija (`LatentDirichletAllocation`, `CountVectorizer`), pandas dokumentacija. Jokio pašalinio kodo nekopijuota.

**DI naudojimas:** Naudojau GitHub Copilot (Claude Sonnet) VS Code aplinkoje.

- **Pradinė analizė (`initial-analysis.md`):** atlikta **pilnai be DI** — 20 antraščių atranka, temų priskyrimas, dviprasmiškumo paaiškinimai ir numatomi sunkumai suformuluoti savarankiškai, kaip reikalauja užduotis.
- **Kodas (`lda_model.py`, 2–5 užduotys):** rašytas **su DI pagalba** — DI padėjo suprojektuoti teksto paruošimo žingsnius (tokenizavimą, stop words, žodyno filtravimą), sukurti LDA bandymus (A/B/C), analizuoti rezultatus ir surasti konkrečius nesėkmės atvejus. Kiekvieną rezultatą (žodyno dydį, temų žodžius, konkrečius `doc_id` svorius) patikrinau paleisdamas kodą pats ir palygindamas išvestį su tuo, kas aprašyta.
- **Šis report'as (`report.md`):** parašytas **mano paties, su nedidele DI pagalba** — DI padėjo suformuluoti kelias pastraipas, bet pagrinde, paruošė man šabloną, kurį pildžiau pats remiantis jau gautais, patikrintais rezultatais.

## Defence preparation

Ensure your program can analyse instructor-provided headlines using the fitted vocabulary and model. Handle inputs with no retained terms. Be prepared to predict, run and explain; no separate new-text experiment or written defence answers are required beforehand.

<!-- `predict_new_headlines()` in lda_model.py already does this: uses the fitted `vectorizer` + `lda_a`, no retraining, reports "insufficient evidence" for empty vectors. -->

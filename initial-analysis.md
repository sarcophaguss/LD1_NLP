# Laboratory 1 — Initial analysis (pre-modelling checkpoint)

Name: Emilis
Date: 2026-09-17

## 1. Cleaning and sampling

- Raw rows in `headlines_train.csv`: 50000
- Rows after removing empty/whitespace-only `text`: 50000
- Unique documents after removing exact duplicates (stripped): 4845
- Sample size used for modelling: 1500
- Sampling method: `pandas.DataFrame.sample`, reproducible random seed = 42
- 20-headline selection method: `pandas.DataFrame.sample` on the 1500-document collection, reproducible random seed = 7
- Stable identifier used: `doc_id` = original row index in `headlines_train.csv`

## 2. Selected 20 headlines

| Headline ID (doc_id) | Full text | Proposed topic(s) | Ambiguous? Why? |
| --- | --- | --- | --- |
| 2246 | 2 Messi įmušė įspūdingą įvartį Lygų taurėje | Sportas | Ne |
| 33321 | 84 Open-source libraries accelerate machine learning workflows | Tech | Ne |
| 44637 | 28 Perėjimų gandai: jaunas puolėjas gali kainuoti rekordą dėl rinkos reakcijos (Vilnius) | Sportas, verslas | Taip, sporto tema maišosi su verslu/rinkos temomis |
| 34143 | Investors hedge inflation risk using index-linked bonds dėl rinkos reakcijos (Vilnius) | Verslas | Ne |
| 6302 | 92 Negalėsite patikėti, kas įvyko toliau — Find out why | Neaiški | Taip, nepateikiama jokia informacija apie temą |
| 49785 | 56 Messi įmušė įspūdingą įvartį Lygų taurėje | Sportas | Ne |
| 29396 | 100 Central bank cuts rates, markets rally on dovish guidance | Verslas | Ne |
| 28784 | 77 Startuolis pritraukė 50 mln. USD DI platformos plėtrai po pranešimo | Tech, verslas | Taip, tech tema maišoma su finansais |
| 1328 | 90 Earnings beat expectations as revenue grows in cloud segment as markets react | Verslas | Ne |
| 41312 | 9 Žvaigždė pasirašė daugiametę sutartį su klubu (Paris) | Sportas | Taip, nežinome ar tai kalbama apie sportininką žvaigždę, taip pat neaišku ar ši tema aktuali Paryžiaus gyventojams ar pats klubas randasi Paryžiuje |
| 42784 | 9 Perėjimų gandai: jaunas puolėjas gali kainuoti rekordą | Sportas, verslas | Taip, sportas maišosi su verslu/finansais |
| 14065 | This trick will change your life forever — Find out why (London) | Neaiški | Taip, neaiški tema, neaišku ar tema aktuali Londono gyventojams ar tai tiesiog random žymė |
| 10908 | 3 Atviro kodo bibliotekos pagreitina mašininio mokymosi procesus analitikų teigimu (Paris) | Tech | Taip, neaišku ar tema aktuali Paryžiaus gyventojams ar tai tiesiog random žymė |
| 479 | 14 Startuolis pritraukė 50 mln. USD DI platformos plėtrai | Tech, verslas | Taip, verslas/finansai maišomi su tech |
| 5477 | 26 How to speed up your computer with simple tweaks | Tech | Ne |
| 21095 | 87 „Apple“ pristatė naują iPhone su pažangia kamera ir DI funkcijomis amid concerns | Tech, verslas | Taip, maišosi tech su verslu/rinka |
| 28988 | 97 Start-up raises $50 million to scale its AI platform | Tech, verslas | Taip, tech maišosi su verslu/finansais |
| 33959 | 71 Start-up raises $50 million to scale its AI platform analitikų teigimu | Tech, verslas | Taip, tech maišosi su verslu/finansais |
| 216 | Vietos komanda laimėjo nacionalinį čempionatą pratęsimuose (London) | Sportas | Taip, neaišku ar tema aktuali Londono gyventojams ar kalbama apie Londono komandą |
| 31650 | 78 Messi įmušė įspūdingą įvartį Lygų taurėje | Sportas | Ne |

## 3. Proposed groups (summary)

- Sportas: 7 antraštės (iš jų 2 persidengia su verslu)
- Tech: 8 antraštės (iš jų 5 persidengia su verslu)
- Verslas: 3 antraštės
- Neaiški: 2 antraštės

## 4. Ambiguous headlines (at least two, with explanation)

- | 41312 | 9 Žvaigždė pasirašė daugiametę sutartį su klubu (Paris) | Sportas | Taip, nežinome ar tai kalbama apie sportininką žvaigždę, taip pat neaišku ar ši tema aktuali Paryžiaus gyventojams ar pats klubas randasi Paryžiuje |

- | 479 | 14 Startuolis pritraukė 50 mln. USD DI platformos plėtrai | Tech, verslas | Taip, verslas/finansai maišomi su tech |


## 5. Predicted modelling problems (at least two, with concrete examples)

1. Prieš kai kurias temas, pradžioje yra prdiėtas skaičius. Taip pat, kai kurios temos yra identiškos tačiau turi mažas variacijas (miestas, papildomos frazės). Dėl šitų skaičių ir mažų variacijų negalime pilnai eliminuoti pasikartojančių temų, nes pandas jas traktuoja dvejomis unikaliomis, atskiromis temomis. Jei tokie dublikatai su nedideliomis variacijomis liks, jie gali dominuoti temoje ir sukurti dirbtinai aiškią, bet nereikšmingą temą, kuri tiesiog atspindi šabloną, o ne tikrą naujienų turinį. PVZ.: Antraštės 2246, 49785, 31650.

2. Modelis gali maišytis tarp temų, nes kai kurios antraštės įvardija kelias temas, apie kurias gali būti parašytas straipsnis. PVZ.: Antraštės 44637, 479.

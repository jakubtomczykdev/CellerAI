# Pomysły badawcze — rozwiązywanie polskich krzyżówek


---

## 1. Taksonomia semantyczna pytań

### Hipoteza
Różne typy pytań obciążają różne zdolności modelu: wiedzę encyklopedyczną, rozumowanie lingwistyczne, znajomość kultury lokalnej. Modele anglocentryczne (GPT, Claude) powinny radzić sobie gorzej niż modele dedykowane polszczyźnie (Bielik, PLLuM) w kategoriach silnie związanych z polską kulturą.

### Realizacja
Należy ręcznie lub półautomatycznie (z pomocą drugiego LLM) przypisać każde pytanie do jednej z kategorii:

1. **Wiedza ogólna / encyklopedyczna** — pytania o fakty (np. „Imię Kurosawy, reżysera" → AKIRA).
2. **Definicje słownikowe** — pytania z wyraźną jednoznaczną definicją (np. „Podwodny grunt" → DNO).
3. **Metafora i gra słowna** — pytania wymagające przeskoku semantycznego (np. „Kropla smutku" → ŁZA).
4. **Nazwy własne i popkultura** — postacie, produkty, marki (np. „Kaczor z kreskówek" → DONALD).
5. **Polskie realia kulturowe** — słownictwo silnie osadzone w polskim kontekście (np. BACA, kazania góralskie, polskie potrawy).

### Analiza
Dla każdego modelu obliczyć accuracy osobno w każdej kategorii. Zwizualizować jako tabelę lub wykres słupkowy grupowany. Szczególnie interesujące: porównanie *różnicy accuracy* między modelami angielskimi a polskimi w kategorii 5.

### Powiązania z literaturą
- Praca 16 (Bielik 7B) i 17 (Bielik v3) — argumentują potrzebę modeli dedykowanych polszczyźnie.
- Praca 18 (PLLuM) — modele dla polskiego kontekstu kulturowego.

---

## 2. Zależność skuteczności od długości słowa

### Hipoteza
Modele LLM mają trudność z operowaniem na poziomie pojedynczych znaków ze względu na tokenizację (słowa są dzielone na podsłowa, nie litery). Przewidujemy, że:
- Dłuższe słowa są trudniejsze do odgadnięcia (więcej „przestrzeni pomyłki").
- Modele częściej zwracają odpowiedzi błędnej długości dla dłuższych słów.
- Modele z tokenizerem zoptymalizowanym pod polski (Bielik v3) popełniają mniej błędów długości.

### Realizacja
Pogrupować pytania po długości odpowiedzi (3, 4, 5, 6, 7+ liter). Dla każdej długości i każdego modelu obliczyć:
- Accuracy.
- Odsetek odpowiedzi o błędnej długości (niezależnie od poprawności semantycznej).
- Średnią liczbę wymaganych podpowiedzi.

### Analiza
Wykres liniowy: oś X — długość słowa, oś Y — accuracy. Osobne linie dla każdego modelu. Drugi wykres dla odsetka błędów długości.

### Powiązania z literaturą
- Praca 13 (Why Do LLMs Struggle to Count Letters?) — pokazuje, że błędy liczenia liter korelują z liczbą liter i tokenów w słowie.
- Praca 14 (Grammar-Constrained Decoding) i 15 (DOMINO) — pokazują, że formalne ograniczenia długości podczas generowania usuwają tę klasę błędów.

---

## 3. Koszt reasoningu vs trudność pytania

### Hipoteza
Modele „myślące" (reasoning models: Claude Opus z thinking, GPT-OSS, Qwen z reasoning_effort) wykorzystują różną liczbę tokenów myślenia dla pytań o różnej trudności. Przewidujemy, że:
- Na banalnych pytaniach („2+2", „Podwodny grunt" → DNO) modele myślące „przesadzają", zużywając setki tokenów na oczywistą odpowiedź.
- Zwiększenie budżetu myślenia powyżej pewnego progu nie poprawia skuteczności (prawo malejących zysków).

### Realizacja
Dla każdego pytania obliczyć jego trudność jako średnią liczbę wymaganych podpowiedzi ze wszystkich modeli. Dla każdego modelu myślącego narysować:
- Oś X: trudność pytania.
- Oś Y: średnia liczba `reasoning_tokens` z `usage.completion_tokens_details`.

### Analiza
Sprawdzić korelację Pearsona między trudnością a liczbą tokenów myślenia. Osobno przeanalizować pytania, w których model odgadł od razu — ile tokenów myślał dla każdej takiej „trywialnej" odpowiedzi?

### Powiązania z literaturą
- Praca 4 (CrossWordBench) — pokazuje, że reasoning LLMs znacząco przewyższają non-reasoning na krzyżówkach dzięki efektywnemu wykorzystaniu ograniczeń.
- Praca 10 (Reasoning-Based Approach to Cryptic Crossword Clue Solving) — formalizuje proces rozumowania w krzyżówkach kryptycznych.

---

## 4. Granica Pareto: koszt vs skuteczność

### Hipoteza
Największy i najdroższy model nie zawsze jest najbardziej efektywny kosztowo. Dla konkretnego zadania (polskie krzyżówki) istnieje zbiór modeli niezdominowanych (granica Pareto), które oferują optymalny kompromis między jakością a ceną.

### Realizacja
Dla każdego modelu obliczyć:
- Accuracy (procent trafionych).
- Średni koszt rozwiązania jednego pytania w USD.

Każdy model to punkt na wykresie dwuwymiarowym: oś X — koszt, oś Y — accuracy. Punkty tworzą chmurę; granica Pareto to zbiór punktów, dla których nie istnieje punkt jednocześnie tańszy i lepszy.

### Analiza
Model *zdominowany* (np. droższy i gorszy od innego) jest praktycznie „zbędny" dla tego zadania. Granica Pareto wskazuje racjonalne wybory. Dodatkowo porównać:
- Najtańszy model osiągający ≥80% accuracy.
- Stosunek jakość/cena (accuracy na 1 USD).

### Wartość dydaktyczna
Prezentacja tej analizy pokazuje, że „więcej" i „drożej" nie są synonimami „lepiej". Pojęcie dominacji Pareto jest uniwersalne (optymalizacja wielokryterialna).

---

## 5. Analiza trybów awarii (failure modes)

### Hipoteza
Błędne odpowiedzi modeli LLM można sklasyfikować na stałą liczbę kategorii. Różne modele mają różny profil błędów — np. jeden częściej podaje synonimy, inny częściej generuje halucynacje.

### Realizacja
Zebrać wszystkie błędne odpowiedzi (`cleaned_answer != correct_answer`, model nie odgadł ostatecznie). Sklasyfikować każdą do jednej z kategorii:

1. **Synonim / bliskoznacznik** — słowo o podobnym znaczeniu (poprawna: STAW, model: JEZIORO).
2. **Powiązana koncepcja** — słowo z tej samej dziedziny, ale niebędące synonimem (poprawna: DNO, model: RYBY).
3. **Hiperonim / hiponim** — szersza lub węższa kategoria (poprawna: JAMNIK, model: PIES).
4. **Błąd długości** — odpowiedź o innej niż oczekiwana liczbie liter.
5. **Błąd formy gramatycznej** — liczba mnoga zamiast pojedynczej, inna odmiana.
6. **Błąd diakrytyki** — „LZA" zamiast „ŁZA", „OWCA" zamiast „ÓWKA".
7. **Halucynacja** — odpowiedź niezwiązana z pytaniem.
8. **Odmowa / nieodpowiedź** — model nie udzielił sensownej odpowiedzi.

Klasyfikację można przyspieszyć, używając drugiego LLM jako anotatora (z ręczną weryfikacją próby).

### Analiza
Macierz: wiersze — modele, kolumny — kategorie błędów, wartości — odsetek błędów danego typu. Pokazuje „osobowość" modelu.

### Powiązania z literaturą
- Praca 9 (Challenges of Cryptic Crosswords for LLMs) — zawiera analogiczną analizę dla krzyżówek kryptycznych (angielskich).

---

## 6. Wpływ kolejności odkrywania liter

### Obecna implementacja
Skrypt odkrywa litery zawsze od lewej do prawej (`answer[:hints]`). Dla słowa RONDO z 2 podpowiedziami maska to `R O _ _ _`.

### Hipoteza
Ludzie w krzyżówkach dostają podpowiedzi na przecięciach haseł — czyli w losowych pozycjach. Różne strategie podpowiadania powinny w różnym stopniu pomagać modelom:
- Odkrywanie początku słowa daje silny sygnał (model potrafi przewidywać „co dalej").
- Odkrywanie końca słowa jest trudniejsze (model musi działać „wstecz").
- Odkrywanie samych spółgłosek daje mało informacji o samogłoskach, ale dużo o strukturze.

### Realizacja
Zmodyfikować funkcję `generate_mask` tak, by obsługiwała różne strategie (parametr `strategy`):
- `prefix` — obecna, od lewej.
- `suffix` — od prawej.
- `random` — losowe pozycje (z ustalonym seedem dla powtarzalności).
- `consonants_first` — najpierw spółgłoski.
- `vowels_first` — najpierw samogłoski.

Uruchomić eksperyment dla każdej strategii, porównać średnią liczbę wymaganych podpowiedzi.

### Wartość naukowa
Wyniki pokazują, *jak* modele wnioskują — czy bardziej „lewo-prawo", czy raczej holistycznie.

---

## 7. Porównanie z ludźmi

### Cel
Ustalić, w których kategoriach pytań modele LLM przewyższają ludzi, a w których pozostają w tyle.

### Realizacja
Zebrać dataset w formie formularza (np. Google Forms) i dać go do rozwiązania grupie 10–30 osób. Każdy respondent dostaje ten sam zestaw pytań bez ograniczeń czasowych (lub z ustalonym limitem — do ustalenia). Dla każdego pytania zbierać:
- Odpowiedź udzieloną.
- Czas udzielenia odpowiedzi.
- Pewność (skala 1–5).

### Analiza
Dla każdego pytania porównać accuracy ludzi (odsetek poprawnych odpowiedzi) z accuracy modelu (z zerem podpowiedzi). Zidentyfikować:
- **Pytania „łatwe dla ludzi, trudne dla modeli"** — zwykle polskie idiomy, gra słów, regionalizmy.
- **Pytania „łatwe dla modeli, trudne dla ludzi"** — zwykle trudne fakty encyklopedyczne, nazwy własne.

### Powiązania z literaturą
- Praca 2 (Berkeley Crossword Solver) — hybryda Dr.Fill + BCS pokonała wszystkich ludzi na amerykańskim turnieju krzyżówkowym.
- Praca 12 (Connections) — analogiczne porównanie człowiek–model na łamigłówce NYT.

---

## 8. Reasoning — czy myślenie pomaga?

### Hipoteza
Włączenie trybu reasoning (parametr `reasoning.effort` w OpenRouter, `thinking` w Anthropic) poprawia accuracy na trudnych pytaniach, ale na łatwych może szkodzić (overthinking — model konstruuje niepotrzebnie skomplikowane hipotezy).

### Realizacja
Dla jednego modelu (np. Claude Sonnet 4.6) uruchomić dwa osobne przebiegi:

1. **Bez reasoning**: `extra_body={"reasoning": {"enabled": false}}` lub bez parametru.
2. **Z reasoning**: `extra_body={"reasoning": {"effort": "high"}}` lub z ustalonym `max_tokens`.

Porównać:
- Accuracy ogólną.
- Accuracy w podziale na trudność (łatwe/średnie/trudne pytania — np. po medianie wymaganych podpowiedzi).
- Całkowity koszt przebiegu.
- Stosunek dodatkowego kosztu do dodatkowej skuteczności.

### Analiza
Wynik ma bezpośrednie znaczenie praktyczne: czy warto płacić więcej za myślenie? Czy lepiej wydać ten budżet na większy model bez myślenia?

---

## 9. Czy model uczy się ze swoich błędów w trakcie rozmowy?

### Obecna implementacja
Gdy model odpowiada błędnie, w kolejnym promptcie dodawana jest lista ostatnich błędnych odpowiedzi z ostrzeżeniem, że są na pewno złe.

### Hipoteza
Duże modele są w stanie wykorzystać tę informację — rzadziej powtarzają błędy. Małe modele ignorują ostrzeżenie i proponują tę samą odpowiedź w nowym kontekście.

### Realizacja
Przeanalizować logi (struktura `attempts` w logach JSON). Dla każdej pary (model, pytanie) policzyć:
- Liczbę unikalnych odpowiedzi udzielonych w kolejnych próbach.
- Liczbę powtórzeń odpowiedzi, która została wcześniej oznaczona jako błędna.

### Analiza
Obliczyć *współczynnik powtarzania błędów* per model. Niski współczynnik oznacza, że model skutecznie wykorzystuje kontekst ujemny.

Dodatkowy eksperyment: uruchomić skrypt w wariancie **bez listy błędnych odpowiedzi w promptcie** i porównać accuracy. Jeśli różnica jest zerowa, oznacza to, że sygnał negatywny nie jest wykorzystywany przez żaden model.

---

## Uwagi metodologiczne

### Powtarzalność pomiarów
Dla modeli z `temperature=0.0` (większość nie-reasoning) pojedyncze uruchomienie daje deterministyczny wynik. Dla modeli z reasoningiem (które zwykle wymagają `temperature=1.0`) wynik jest losowy i każde pytanie powinno być uruchomione co najmniej 3 razy, a accuracy uśrednione.

### Obiektywizm ewaluacji
Porównanie string-matchem (`model_response == correct_answer`) jest restrykcyjne — model podający synonim zostanie uznany za błędnego, choć rzeczowo ma rację. W precyzyjniejszych badaniach warto dodać ewaluację przez drugi LLM lub człowieka („czy ta odpowiedź jest akceptowalnym synonimem?").

### Data cutoff modeli
Modele trenowane do pewnej daty mogą nie znać np. osób, które zasłynęły później. Dataset powinien unikać pytań o aktualności bieżące, żeby nie mieszać dwóch zmiennych: „zdolności rozwiązywania krzyżówek" i „świeżości wiedzy".

---

## Propozycja kolejności eksperymentów

1. **Taksonomia semantyczna** (pkt 1) — szybko daje ciekawe wnioski.
2**Analiza Pareto kosztu/jakości** (pkt 4) — pojedynczy czytelny wykres.
3**Reasoning vs brak reasoningu** (pkt 8) — konkretny eksperyment porównawczy.
4**Porównanie z ludźmi** (pkt 7) — daje najmocniejszą konkluzję.


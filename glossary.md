# 📖 Glossary — Podstawy Analizy Danych

## Wzorce braków danych

### MCAR — Missing Completely At Random
Braki są **całkowicie losowe** — niezależne od jakichkolwiek innych zmiennych w datasecie ani od samej brakującej wartości.

**Przykład:** komputer losowo "gubi" co 10. rekord podczas zapisu.

**Konsekwencja:** można bezpiecznie imputować globalną medianą/średnią lub usunąć wiersze bez wprowadzania błędu systematycznego.

---

### MAR — Missing At Random
Braki **zależą od innych zmiennych** w datasecie (obserwowalnych), ale nie od samej brakującej wartości.

**Przykład w naszym datasecie:** braki w `Age` częściej występują na `JobLevel 3` niż na `JobLevel 1`. Znając `JobLevel`, można przewidzieć prawdopodobieństwo braku.

**Konsekwencja:** należy imputować **grupowo** — np. medianą wg `JobLevel + JobRole`. Globalna mediana byłaby błędem systematycznym.

> ✅ `Age` i `MonthlyIncome` w HR.csv → **MCAR** (potwierdzone testami Mann-Whitney U i Chi-kwadrat — brak istotnej zależności między brakami a innymi zmiennymi)

---

### MNAR — Missing Not At Random
Braki **zależą od samej brakującej wartości** — dane "celowo" znikają z powodu swojej wartości.

**Przykład:** pracownicy z bardzo niskim wynagrodzeniem nie podają `MonthlyIncome` bo się wstydzą. Albo bardzo starzy pracownicy nie podają `Age`.

**Konsekwencja:** najtrudniejszy przypadek — żadna prosta imputacja nie jest w pełni poprawna. Wymaga modelowania mechanizmu braków.

---

## Metody imputacji

### KNN Imputer — K-Nearest Neighbors
Metoda imputacji brakujących wartości oparta na K najbliższych sąsiadach. Dla każdego brakującego rekordu wyszukuje K najbardziej podobnych wierszy (wg odległości euklidesowej w przestrzeni cech) i wypełnia brak ich średnią ważoną.

**Zalety nad medianą grupową:**
- Uwzględnia wiele cech jednocześnie (nie tylko jedną grupę)
- Zachowuje std i rozkład lepiej niż globalna mediana
- Nie wymaga ręcznego definiowania grup

**Pipeline w projekcie:**
1. One-Hot Encoding zmiennych kategorycznych
2. RobustScaler zmiennych numerycznych (odporny na outliery)
3. Wybór K metodą łokcia (MSE na sztucznie usuniętych wartościach)
4. Finalna imputacja KNNImputer z wybranym K

> ✅ `Age` → K=6, `MonthlyIncome` → K=4 (wykresy: `charts/knn/`)

---

### Elbow Method — Metoda łokcia
Technika wyboru optymalnego K dla KNN. Dla każdego K obliczamy MSE na zbiorze testowym (20% znanych wartości sztucznie usuniętych), a następnie wybieramy K w "zgięciu" krzywej — punkt gdzie dalsze zwiększanie K nie przynosi istotnej poprawy.

---

### MSE — Mean Squared Error (Błąd średniokwadratowy)
Miara błędu imputacji — średnia kwadratów różnic między wartościami prawdziwymi a przewidywanymi.

```
MSE = (1/n) × Σ(y_true - y_pred)²
```

Im niższe MSE, tym lepsza imputacja. Używane w metodzie łokcia do wyboru K.

---

## Metody statystyczne

### IQR — Interquartile Range (Rozstęp Międzykwartylowy)
Miara rozproszenia danych odporna na wartości odstające.

```
IQR = Q3 - Q1
```

gdzie:
- **Q1** = 25. percentyl (dolny kwartyl)
- **Q3** = 75. percentyl (górny kwartyl)

**Granice outlierów:**
- dolna: `Q1 - 1.5 × IQR`
- górna: `Q3 + 1.5 × IQR`

Wartości poza tymi granicami uznawane są za **outliery**.

---

### Mutual Information (MI)
Miara **nieliniowej zależności** między dwiema zmiennymi. W przeciwieństwie do korelacji Pearsona wykrywa dowolne zależności, nie tylko liniowe.

- MI = 0 → brak zależności
- MI > 0 → zależność (im wyższe, tym silniejsza)
- MI > 1.0 → bardzo silna zależność (np. `JobLevel` → `MonthlyIncome`)

Używany do wyboru zmiennych grupujących przy imputacji.

---

### Pearson r — Współczynnik korelacji Pearsona
Miara **liniowej zależności** między dwiema zmiennymi numerycznymi.

- r = 1.0 → idealna korelacja dodatnia
- r = 0.0 → brak korelacji liniowej
- r = -1.0 → idealna korelacja ujemna

⚠️ Nie działa dla zmiennych kategorycznych — dla nich używamy Cramer's V lub MI.

---

### Mann-Whitney U Test
Nieparametryczny test statystyczny sprawdzający czy dwie grupy pochodzą z tego samego rozkładu. Używany do wykrywania wzorca **MAR** — czy rozkład zmiennej numerycznej (np. `Age`) różni się istotnie między grupą z brakami a bez braków.

- p-value < 0.05 → istotna różnica → sugeruje MAR
- p-value ≥ 0.05 → brak istotnej różnicy → sugeruje MCAR

---

### Test Chi-kwadrat
Test statystyczny dla zmiennych kategorycznych. W kontekście analizy braków sprawdza czy częstość braków jest niezależna od wartości zmiennej kategorycznej (np. `JobLevel`).

- p-value < 0.05 → zależność → sugeruje MAR
- p-value ≥ 0.05 → niezależność → sugeruje MCAR

---

### Odchylenie standardowe (std)
Pierwiastek kwadratowy z wariancji. Mówi jak bardzo wartości są rozproszone wokół średniej.

```
std = √( (1/n) × Σ(xᵢ - x̄)² )
```

Używane do oceny jakości imputacji — dobra imputacja nie powinna istotnie zmieniać std kolumny.

---

### Średnia arytmetyczna (mean)
Suma wszystkich wartości podzielona przez ich liczbę.

```
mean = (x₁ + x₂ + ... + xₙ) / n
```

⚠️ Wrażliwa na outliery — przy skośnych rozkładach (jak `MonthlyIncome`) mediana jest lepszą miarą centrum.


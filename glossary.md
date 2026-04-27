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

> ✅ `Age` i `MonthlyIncome` w HR.csv → **MAR** (potwierdzone analizą rozkładu JobLevel)

---

### MNAR — Missing Not At Random
Braki **zależą od samej brakującej wartości** — dane "celowo" znikają z powodu swojej wartości.

**Przykład:** pracownicy z bardzo niskim wynagrodzeniem nie podają `MonthlyIncome` bo się wstydzą. Albo bardzo starzy pracownicy nie podają `Age`.

**Konsekwencja:** najtrudniejszy przypadek — żadna prosta imputacja nie jest w pełni poprawna. Wymaga modelowania mechanizmu braków.

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


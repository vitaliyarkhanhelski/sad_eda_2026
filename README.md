![HR Analytics Banner](assets/banner.jpg)
# 🏢 Analiza danych HR — IBM HR Analytics Employee Attrition

**📚 Przedmiot:** Podstawy Analizy Danych, 2025/2026  
**👨‍🏫 Prowadzący:** dr inż. Karol Flisikowski, prof. PG

## 📌 Opis projektu

Eksploracyjna analiza danych (EDA) zbioru IBM HR Analytics dotyczącego rotacji pracowników (*attrition*). Celem projektu jest identyfikacja czynników wpływających na odejście pracowników z firmy oraz przygotowanie danych do przyszłego modelowania klasyfikacyjnego.

### 📋 Treść zadania (wg prowadzącego)

Analityka HR pomaga w interpretacji danych organizacyjnych. Znajduje trendy związane z ludźmi i pozwala działowi HR podjąć odpowiednie kroki, aby organizacja działała sprawnie i przynosiła zyski. *Attrition* w korporacji jest jednym ze złożonych wyzwań, z którymi muszą sobie radzić menedżerowie i pracownicy HR.

Modele uczenia maszynowego można wdrożyć w celu przewidywania potencjalnych przypadków odejść, pomagając odpowiedniemu personelowi HR podjąć niezbędne kroki w celu zatrzymania pracownika.

**✅ Wymagane zadania:**

🧹 *Czyszczenie danych:* usunięcie zbędnych kolumn, zmiana nazw kolumn, usuwanie duplikatów, czyszczenie poszczególnych kolumn, usuwanie wartości NaN, obsługa wartości brakujących i odstających, dodatkowe transformacje.

📊 *Wizualizacja danych:* mapa korelacji dla zmiennych numerycznych, nadgodziny, stan cywilny, rola zawodowa, płeć, dziedzina wykształcenia, dział, podróż służbowa, związek między nadgodzinami a wiekiem, łączna liczba lat pracy, poziom wykształcenia, liczba przepracowanych firm, odległość od domu.

## 🗃️ Dataset

**📄 Plik:** `HR.csv`  
**📏 Rozmiar:** 1470 wierszy × 35 kolumn  
**🔗 Źródło:** IBM HR Analytics (via [kflisikowski/analiza_danych_projekt_zespolowy](https://github.com/kflisikowski/analiza_danych_projekt_zespolowy))

### Opis

Analityka HR pomaga w interpretacji danych organizacyjnych — znajduje trendy związane z ludźmi i pozwala działowi HR podjąć odpowiednie kroki, aby organizacja działała sprawnie i przynosiła zyski. Rotacja pracowników (*attrition*) jest jednym ze złożonych wyzwań, z którymi muszą sobie radzić menedżerowie i pracownicy HR.

Modele uczenia maszynowego można wdrożyć w celu przewidywania potencjalnych przypadków odejść, pomagając działowi HR podjąć niezbędne kroki w celu zatrzymania pracownika.

**🎯 Zmienna docelowa:** `Attrition` — czy pracownik odszedł z firmy (Yes/No)

### ⚠️ Braki danych (celowe, do obsługi w projekcie)

| Kolumna | Braki | % |
|---|---|---|
| `Age` | 100 | 6.8% |
| `Attrition` | 150 | 10.2% |
| `MonthlyIncome` | 150 | 10.2% |

**🗑️ Kolumny stałe (do usunięcia):** `EmployeeCount`, `StandardHours`, `Over18`

### 🔍 Zakres analizy

**🧹 Czyszczenie danych:**
- Usunięcie zbędnych kolumn
- Zmiana nazw kolumn
- Usuwanie duplikatów
- Czyszczenie poszczególnych kolumn
- Usuwanie wartości NaN
- Obsługa wartości brakujących i odstających

**📊 Wizualizacje:**
- Mapa korelacji dla wszystkich zmiennych numerycznych
- Nadgodziny
- Stan cywilny
- Rola zawodowa
- Płeć
- Dziedzina wykształcenia
- Dział
- Podróż służbowa
- Związek między nadgodzinami a wiekiem
- Łączna liczba lat pracy
- Poziom wykształcenia
- Liczba przepracowanych firm
- Odległość od domu

## 👥 Zespół i podział pracy

Projekt podzielony jest na 4 równoważne etapy — każdy członek zespołu odpowiada za jeden etap pipeline'u i samodzielnie implementuje przypisany plik. Etapy są sekwencyjne: każdy kolejny korzysta z wyniku poprzedniego.

| Osoba                | Plik                          | Etap                              |
|----------------------|-------------------------------|-----------------------------------|
| Vitaliy Arkhanhelski | `src/data_preprocessing.py`   | Etap 1 — 🧹 Czyszczenie danych       |
| Osoba 2              | `src/descriptive_analysis.py` | Etap 2 — 📈 Analiza opisowa          |
| Osoba 3              | `src/correlation_analysis.py` | Etap 3 — 🔗 Korelacje i wizualizacje |
| Osoba 4              | `src/data_preparation.py`     | Etap 4 — ⚙️ Przygotowanie do modelu  |

---

### 🧹 Etap 1 — Czyszczenie danych [`src/data_preprocessing.py`] — Vitaliy Arkhanhelski

**Dlaczego ten etap?** Surowy dataset HR.csv zawiera celowo wprowadzone braki (`Age`, `Attrition`, `MonthlyIncome`) oraz kolumny bezużyteczne dla analizy. Bez czyszczenia dalsze etapy dałyby błędne wyniki — każda analiza statystyczna i wizualizacja zależy od jakości danych wejściowych.

**Zadania:**
- Usunięcie kolumn stałych: `EmployeeCount`, `StandardHours`, `Over18`
- Analiza wzorców braków danych (MCAR / MAR / MNAR)
- Imputacja `Age` — medianą wg `JobLevel`
- Imputacja `MonthlyIncome` — medianą wg `JobLevel` + `JobRole`
- `Attrition` — **nie imputować** (zmienna docelowa)
- Wykrycie i obsługa wartości odstających metodą IQR
- Usuwanie duplikatów
- Zapis oczyszczonego datasetu → `HR_clean.csv`

📥 **Wejście:** surowy `df` z `data_loader.load_data()`  
📤 **Wyjście:** oczyszczony `DataFrame` przekazywany dalej

---

### 📈 Etap 2 — Analiza opisowa i wnioskowanie statystyczne [`src/descriptive_analysis.py`] — Osoba 2

**Dlaczego ten etap?** Po oczyszczeniu danych należy zrozumieć ich rozkłady i sprawdzić, które zmienne statystycznie różnią się między pracownikami, którzy odeszli, a tymi, którzy pozostali. Testy statystyczne dają obiektywne podstawy do wyboru zmiennych do modelu.

**Zadania:**
- Statystyki opisowe zmiennych numerycznych: średnia, mediana, odchylenie standardowe, skośność, kurtoza, współczynnik zmienności CV
- Histogramy i boxploty z podziałem na `Attrition = Yes / No`
- Analiza zmiennej docelowej `Attrition` — rozkład klas (niezbalansowanie)
- Testy statystyczne:
  - zmienne numeryczne vs `Attrition` → **test Manna-Whitneya** (nieparametryczny)
  - zmienne kategoryczne vs `Attrition` → **test Chi-kwadrat**
- Wizualizacja wyników testów — ranking istotności zmiennych

📥 **Wejście:** `HR_clean.csv` / oczyszczony `df`  
📤 **Wyjście:** wykresy + podsumowanie wyników testów

---

### 🔗 Etap 3 — Analiza korelacji i wizualizacje [`src/correlation_analysis.py`] — Osoba 3

**Dlaczego ten etap?** Sama statystyka opisowa nie pokazuje wzajemnych zależności między zmiennymi. Wizualizacje korelacji i wykresy wg grup biznesowych (dział, rola, nadgodziny) pozwalają odkryć, jakie profile pracowników są najbardziej narażone na odejście.

**Zadania:**
- Macierz korelacji Spearmana dla zmiennych numerycznych — heatmapa
- Top 10 zmiennych najbardziej skorelowanych z `Attrition`
- Cramer's V — macierz asocjacji zmiennych kategorycznych
- Scatter matrix dla kluczowych zmiennych z kolorowaniem wg `Attrition`
- Parallel Coordinates — profile pracowników odchodzących vs pozostających
- Attrition rate wg grup: `Department`, `JobRole`, `MaritalStatus`, `OverTime`, `BusinessTravel`, `Gender`, `EducationField`
- Wykresy: nadgodziny, stan cywilny, podróż służbowa, odległość od domu, łączne lata pracy, liczba firm

📥 **Wejście:** `HR_clean.csv` / oczyszczony `df`  
📤 **Wyjście:** interaktywne wykresy (Plotly)

---

### ⚙️ Etap 4 — Przygotowanie danych do modelowania [`src/data_preparation.py`] — Osoba 4

**Dlaczego ten etap?** Modele uczenia maszynowego wymagają danych w formacie numerycznym i odpowiednio przeskalowanych. Ten etap zamyka cały pipeline EDA i produkuje gotowe datasety, które w przyszłości trafią bezpośrednio do treningu modelu klasyfikacyjnego.

**Zadania:**
- Encoding zmiennych kategorycznych:
  - Binary: `Attrition` (Yes=1/No=0), `Gender`, `OverTime`
  - Ordinal: `BusinessTravel` (Non-Travel=0, Rarely=1, Frequently=2)
  - One-Hot: `Department`, `EducationField`, `JobRole`, `MaritalStatus`
- Skalowanie zmiennych numerycznych:
  - Standaryzacja Z-score (`StandardScaler`)
  - Normalizacja Min-Max (`MinMaxScaler`)
  - Porównanie rozkładów przed i po skalowaniu (wykresy)
- Krótkie podsumowanie: które zmienne warto uwzględnić w modelu
- Zapis finalnych datasetów → `HR_model_standardized.csv`, `HR_model_normalized.csv`

📥 **Wejście:** `HR_clean.csv` / oczyszczony `df`  
📤 **Wyjście:** dwa pliki CSV gotowe do modelowania

## 📁 Struktura projektu

```
sad_eda_2026/
├── README.md
├── requirements.txt
├── assets/
│   └── banner.png
├── data/
│   ├── HR.csv                        # Surowy dataset
│   ├── HR_clean.csv                  # Generowany przez Etap 1 [Vitaliy Arkhanhelski]
│   ├── HR_model_standardized.csv     # Generowany przez Etap 4
│   └── HR_model_normalized.csv       # Generowany przez Etap 4
├── reports/
│   └── HR_profiling_report.html              # Automatyczny raport (ydata_profiling)
└── src/
    ├── settings.py                   # Ścieżki i konfiguracja projektu
    ├── main.py                       # Punkt wejścia — uruchamia cały pipeline
    ├── data_loader.py                # Wczytanie danych + raport profilujący
    ├── data_preprocessing.py         # Etap 1 — czyszczenie danych [Vitaliy Arkhanhelski]
    ├── descriptive_analysis.py       # Etap 2 — analiza opisowa
    ├── correlation_analysis.py       # Etap 3 — korelacje i wizualizacje
    └── data_preparation.py           # Etap 4 — przygotowanie do modelu
```

## 🔄 Pipeline

```
load_data()
    └── generate_report()             # 📄 raport profilujący HTML
        └── clean_data()              # 🧹 Etap 1 [Vitaliy Arkhanhelski]
            ├── run() [descriptive]   # 📈 Etap 2
            ├── run() [correlation]   # 🔗 Etap 3
            └── run() [preparation]   # ⚙️ Etap 4
```

## 🚀 Uruchomienie

```bash
pip install -r requirements.txt
cd src
python main.py
```

Raport profilujący zostanie zapisany w `reports/HR_profiling_report.html`.

## 🏆 Kryteria oceny (wg prowadzącego)

| Kryterium | Punkty |
|---|---|
| 🤝 Praca zespołowa (git, branche, PR) | 0–2 |
| ✍️ Estetyka, Markdown, komentarze, wnioski | 0–3 |
| 🧹 Porządkowanie i czyszczenie danych | 0–3 |
| 📊 Wizualizacja danych | 0–4 |
| 📈 Analiza opisowa | 0–4 |
| 🔬 Wnioskowanie statystyczne | 0–4 |
| **🎯 Razem** | **0–20** |

## 📜 Licencja

CC BY-SA 4.0 — [Karol Flisikowski](https://statosfera.pl/)

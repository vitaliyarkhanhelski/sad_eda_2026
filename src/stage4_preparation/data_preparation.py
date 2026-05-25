"""
Przygotowanie danych do modelowania — Osoba 4

Zakres:
- Encoding zmiennych kategorycznych:
    * Binary encoding: Attrition (Yes=1/No=0), Gender, OverTime
    * Ordinal encoding: BusinessTravel (Non-Travel=0, Rarely=1, Frequently=2)
    * One-Hot encoding: Department, EducationField, JobRole, MaritalStatus
- Skalowanie zmiennych numerycznych:
    * Standaryzacja Z-score (StandardScaler) — zakres: mean=0, std=1
    * Normalizacja Min-Max (MinMaxScaler) — zakres [0, 1]
    * Porównanie i uzasadnienie wyboru metody
- Wizualizacja rozkładów przed i po skalowaniu (np. MonthlyIncome)
- Zapis finalnych datasetów:
    * HR_model_standardized.csv
    * HR_model_normalized.csv
- Krótkie podsumowanie: które zmienne warto uwzględnić w modelu
  (na podstawie korelacji z Attrition — BEZ uczenia modelu)

Wejście: oczyszczony df z data_preprocessing.clean_data()
Wyjście: dwa pliki CSV gotowe do modelowania w przyszłości

Uwaga: używaj tylko wierszy z wypełnionym Attrition
"""

import pandas as pd


def run(df: pd.DataFrame) -> None:
    """Uruchamia preprocessing i zapisuje finalne datasety."""
    # TODO: implementacja
    raise NotImplementedError

"""
Czyszczenie danych — Osoba 1

Zakres:
- Usunięcie kolumn zbędnych (stałe, duplikaty)
- Zmiana nazw kolumn (opcjonalnie)
- Analiza i wizualizacja wzorców braków danych (MCAR / MAR / MNAR)
- Imputacja wartości brakujących:
    * Age — imputacja grupowa medianą wg JobLevel
    * MonthlyIncome — imputacja grupową medianą wg JobLevel + JobRole
    * Attrition — NIE imputować (zmienna docelowa)
- Usuwanie lub flagowanie wartości odstających (IQR)
- Zapis oczyszczonego pliku HR_clean.csv

Wejście: surowy df z data_loader.load_data()
Wyjście: oczyszczony DataFrame przekazywany dalej do pipeline'u
"""

import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Czyści dataset HR i zwraca oczyszczony DataFrame."""
    # TODO: implementacja
    raise NotImplementedError

"""
Analiza opisowa i wnioskowanie statystyczne — Osoba 2

Zakres:
- Statystyki opisowe zmiennych numerycznych:
    * średnia, mediana, odchylenie standardowe, skośność (skewness), kurtoza
    * współczynnik zmienności CV = std/mean
- Rozkłady zmiennych — histogramy i boxploty z podziałem na Attrition=Yes/No
- Analiza zmiennej docelowej Attrition:
    * rozkład (pie chart / bar chart), niezbalansowanie klas
- Testy statystyczne:
    * zmienne numeryczne vs Attrition → test Manna-Whitneya (nieparametryczny)
    * zmienne kategoryczne vs Attrition → test Chi-kwadrat
- Wizualizacja wyników testów — które zmienne są istotnie różne między grupami

Wejście: oczyszczony df z data_preprocessing.clean_data()
Wyjście: wykresy + podsumowanie wyników testów (print / zapis do pliku)

Uwaga: do testów używaj tylko wierszy z wypełnionym Attrition
"""

import pandas as pd


def run(df: pd.DataFrame) -> None:
    """Uruchamia analizę opisową i statystyczną."""
    # TODO: implementacja
    raise NotImplementedError

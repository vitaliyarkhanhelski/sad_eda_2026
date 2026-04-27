"""
Analiza danych HR — IBM HR Analytics Employee Attrition
Przedmiot: Podstawy Analizy Danych, 2025/2026

Punkt wejścia projektu. Uruchamia kolejno wszystkie etapy analizy.
"""

import src.data_loader as data_loader
import src.stage1_preprocessing.data_preprocessing as data_preprocessing
import src.stage2_descriptive.descriptive_analysis as descriptive_analysis
import src.stage3_correlation.correlation_analysis as correlation_analysis
import src.stage4_preparation.data_preparation as data_preparation


def main():
    # Krok 1: Wczytanie danych
    df = data_loader.load_data()

    # Krok 2: Automatyczny raport profilujący (ydata_profiling)
    data_loader.generate_report(df)

    # Krok 3: Czyszczenie danych [Vitaliy Arkhanhelski]
    df = data_preprocessing.clean_data(df)

    # Krok 4: Analiza opisowa i statystyczna [Osoba 2]
    descriptive_analysis.run(df)

    # Krok 5: Korelacje i wizualizacje [Osoba 3]
    correlation_analysis.run(df)

    # Krok 6: Preprocessing pod model [Osoba 4]
    data_preparation.run(df)


if __name__ == "__main__":
    main()

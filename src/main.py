"""
Analiza danych HR — IBM HR Analytics Employee Attrition
Przedmiot: Podstawy Analizy Danych, 2025/2026

Punkt wejścia projektu. Uruchamia kolejno wszystkie etapy analizy.
"""

import src.data_loader as data_loader
import src.report as report
import src.preprocessing as preprocessing
import src.stage2_descriptive.descriptive_analysis as descriptive_analysis
import src.stage3_correlation.correlation_analysis as correlation_analysis
import src.stage4_anova.anova_analysis as anova_analysis
import src.stage5_preparation.data_preparation as data_preparation


def main():
    # Krok 1: Wczytanie danych
    df = data_loader.load_data()

    # Krok 2: Automatyczny raport profilujący (ydata_profiling)
    report.generate_report(df)

    # Krok 3: Eksploracja datasetu — wykresy braków, statystyki, testy MAR/MCAR
    preprocessing.dataset_analysis(df)

    # Krok 4: Czyszczenie danych — duplikaty, imputacja KNN, outliery
    df = preprocessing.clean_data(df)

    # Krok 5: Analiza opisowa i statystyczna
    descriptive_analysis.run(df)

    # Krok 6: Korelacje i wizualizacje
    correlation_analysis.run(df)

    # Krok 7: ANOVA i ANCOVA
    anova_analysis.run(df)

    # Krok 8: Preprocessing pod model
    data_preparation.run(df)


if __name__ == "__main__":
    main()
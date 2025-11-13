"""
Hauptskript zum Laden der Daten, Feature Engineering und Training
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

from neural_network_model import (
    FinancialDataset,
    NeuralNetworkRegressor,
    MomentumFeatureEngineering,
    DataPreparation,
    ModelTrainer,
    plot_results,
    save_model
)


def load_and_merge_data():
    """Lädt die Excel-Dateien und merged die Daten"""
    print("\n" + "="*70)
    print("SCHRITT 1: DATEN LADEN")
    print("="*70)

    # Lade INDEX.xlsx
    index_df = pd.read_excel('DataStorage/INDEX.xlsx')
    print(f"\nINDEX.xlsx geladen: {index_df.shape}")
    print(f"Columns: {list(index_df.columns)}")

    # Lade Portfolio.xlsx
    portfolio_df = pd.read_excel('DataStorage/Portfolio.xlsx')
    print(f"\nPortfolio.xlsx geladen: {portfolio_df.shape}")
    print(f"Columns: {list(portfolio_df.columns)}")

    return index_df, portfolio_df


def prepare_data(index_df, portfolio_df):
    """
    Bereitet die Daten vor und erstellt Features
    WICHTIG: Anpassung an die tatsächliche Datenstruktur erforderlich!
    """
    print("\n" + "="*70)
    print("SCHRITT 2: DATA PREPARATION & FEATURE ENGINEERING")
    print("="*70)

    # Identifiziere Datum und Preis Spalten (muss an tatsächliche Struktur angepasst werden)
    # Annahme: Es gibt eine Datumsspalte und Preisspalten

    # Beispiel-Logik (muss angepasst werden basierend auf tatsächlicher Struktur)
    df_combined = None

    # Check ob es ein Date/Time Column gibt
    date_cols = [col for col in index_df.columns if 'date' in col.lower() or 'time' in col.lower()]

    if date_cols:
        print(f"\nGefundene Datumsspalten: {date_cols}")
        # Setze erste Datumsspalte als Index
        index_df = index_df.copy()
        index_df[date_cols[0]] = pd.to_datetime(index_df[date_cols[0]])
        index_df = index_df.set_index(date_cols[0])

    # Portfolio ebenfalls vorbereiten
    date_cols_portfolio = [col for col in portfolio_df.columns if 'date' in col.lower() or 'time' in col.lower()]
    if date_cols_portfolio:
        portfolio_df = portfolio_df.copy()
        portfolio_df[date_cols_portfolio[0]] = pd.to_datetime(portfolio_df[date_cols_portfolio[0]])
        portfolio_df = portfolio_df.set_index(date_cols_portfolio[0])

    # Merge dataframes wenn möglich
    if date_cols and date_cols_portfolio:
        df_combined = pd.concat([index_df, portfolio_df], axis=1)
        print(f"\nCombined DataFrame: {df_combined.shape}")
    else:
        # Fallback: Nutze index_df
        df_combined = index_df
        print(f"\nVerwende INDEX DataFrame: {df_combined.shape}")

    # Handle missing values
    print("\nHandle Missing Values...")
    df_combined = DataPreparation.handle_missing_values(df_combined, method='interpolate')

    # Identifiziere numerische Spalten für Feature Engineering
    numeric_cols = df_combined.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\nNumerische Spalten: {len(numeric_cols)}")
    print(numeric_cols[:10])  # Zeige erste 10

    # Feature Engineering für jede numerische Spalte
    feature_dfs = [df_combined[numeric_cols].copy()]

    print("\nBerechne Momentum Features...")
    momentum_eng = MomentumFeatureEngineering()

    # Nehme erste numerische Spalte als Hauptpreis (oder identifiziere Close/Price Spalte)
    price_cols = [col for col in numeric_cols if any(word in col.lower() for word in ['close', 'price', 'value'])]

    if price_cols:
        main_price_col = price_cols[0]
    else:
        main_price_col = numeric_cols[0] if numeric_cols else None

    if main_price_col:
        print(f"\nHauptpreis-Spalte: {main_price_col}")
        prices = df_combined[main_price_col]

        # Berechne verschiedene Momentum-Indikatoren
        returns = momentum_eng.calculate_returns(prices)
        momentum = momentum_eng.calculate_momentum(prices)
        ma = momentum_eng.calculate_moving_averages(prices)
        volatility = momentum_eng.calculate_volatility(prices)

        # RSI und MACD
        rsi = momentum_eng.calculate_rsi(prices)
        macd = momentum_eng.calculate_macd(prices)

        # Füge alle Features hinzu
        feature_dfs.extend([returns, momentum, ma, volatility,
                           pd.DataFrame({'rsi': rsi}), macd])

    # Kombiniere alle Features
    features_df = pd.concat(feature_dfs, axis=1)

    # Entferne Duplikate-Spalten
    features_df = features_df.loc[:, ~features_df.columns.duplicated()]

    print(f"\nFeatures erstellt: {features_df.shape[1]} Features")

    # Entferne Rows mit NaN (durch Rolling Windows entstanden)
    features_df = features_df.dropna()
    print(f"Nach Entfernung von NaN: {features_df.shape}")

    # Outlier Removal
    print("\nEntferne Outliers...")
    features_df = DataPreparation.remove_outliers_iqr(
        features_df,
        features_df.columns.tolist(),
        factor=2.0
    )

    return features_df, main_price_col


def create_target_variable(features_df, target_col, forecast_horizon=1):
    """
    Erstellt die Zielvariable (z.B. zukünftiger Preis oder Return)
    """
    print("\n" + "="*70)
    print("SCHRITT 3: ZIELVARIABLE ERSTELLEN")
    print("="*70)

    # Ziel: Vorhersage des zukünftigen Returns
    # Shift target nach oben (future value)
    if target_col and target_col in features_df.columns:
        # Vorhersage des zukünftigen Returns
        target = features_df[target_col].pct_change(forecast_horizon).shift(-forecast_horizon)
        print(f"\nZielvariable: {forecast_horizon}-Tage Return von {target_col}")
    else:
        # Fallback: Erste Spalte
        target_col = features_df.columns[0]
        target = features_df[target_col].pct_change(forecast_horizon).shift(-forecast_horizon)
        print(f"\nZielvariable (Fallback): {forecast_horizon}-Tage Return von {target_col}")

    # Entferne Rows mit NaN im Target
    valid_idx = target.notna()
    features_df = features_df[valid_idx]
    target = target[valid_idx]

    print(f"Final Dataset Shape: {features_df.shape}")
    print(f"Target Stats: Mean={target.mean():.6f}, Std={target.std():.6f}")

    return features_df, target


def train_neural_network(X, y, test_size=0.2, val_size=0.1):
    """Trainiert das Neural Network"""
    print("\n" + "="*70)
    print("SCHRITT 4: MODELL TRAINING")
    print("="*70)

    # Train/Val/Test Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, shuffle=False
    )

    print(f"\nTrain Set: {X_train.shape}")
    print(f"Validation Set: {X_val.shape}")
    print(f"Test Set: {X_test.shape}")

    # Scaling mit RobustScaler (besser für Outliers)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # PyTorch Datasets
    train_dataset = FinancialDataset(X_train_scaled, y_train)
    val_dataset = FinancialDataset(X_val_scaled, y_val)

    # DataLoaders
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Modell initialisieren
    input_size = X_train.shape[1]
    model = NeuralNetworkRegressor(
        input_size=input_size,
        hidden_sizes=[256, 128, 64, 32],
        dropout_rate=0.3
    )

    print(f"\nModell Architektur:")
    print(model)
    print(f"\nParameter Count: {sum(p.numel() for p in model.parameters()):,}")

    # Training
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    trainer = ModelTrainer(model, device=device)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=150,
        learning_rate=0.001,
        patience=20
    )

    # Evaluation
    print("\n" + "="*70)
    print("SCHRITT 5: EVALUATION")
    print("="*70)

    # Train Set Evaluation
    train_metrics = trainer.evaluate(X_train_scaled, y_train)
    print(f"\nTRAIN SET METRICS:")
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Validation Set Evaluation
    val_metrics = trainer.evaluate(X_val_scaled, y_val)
    print(f"\nVALIDATION SET METRICS:")
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.6f}")

    # Test Set Evaluation
    test_metrics = trainer.evaluate(X_test_scaled, y_test)
    print(f"\nTEST SET METRICS:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")

    print(f"\n{'='*70}")
    print(f"FINAL R² SCORE (TEST): {test_metrics['r2_score']:.6f}")
    print(f"{'='*70}")

    # Visualisierungen
    print("\nErstelle Visualisierungen...")
    y_pred_train = trainer.predict(X_train_scaled)
    y_pred_val = trainer.predict(X_val_scaled)
    y_pred_test = trainer.predict(X_test_scaled)

    plot_results(y_train, y_pred_train, "Train Set")
    plot_results(y_val, y_pred_val, "Validation Set")
    plot_results(y_test, y_pred_test, "Test Set")

    # Speichere Modell
    save_model(model, scaler, 'neural_network_model.pth')

    # Speichere Metriken
    with open('model_metrics.txt', 'w') as f:
        f.write("NEURAL NETWORK REGRESSION - MODEL METRICS\n")
        f.write("="*70 + "\n\n")
        f.write("TRAIN SET:\n")
        for k, v in train_metrics.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\nVALIDATION SET:\n")
        for k, v in val_metrics.items():
            f.write(f"  {k}: {v:.6f}\n")
        f.write("\nTEST SET:\n")
        for k, v in test_metrics.items():
            f.write(f"  {k}: {v:.6f}\n")

    print("\nMetriken gespeichert: model_metrics.txt")

    return model, scaler, test_metrics


def main():
    """Hauptfunktion"""
    print("\n" + "="*70)
    print("NEURAL NETWORK REGRESSION - FINANZMARKT-DATEN")
    print("Mit Momentum-Features und Noise Filtering")
    print("="*70)

    # 1. Daten laden
    index_df, portfolio_df = load_and_merge_data()

    # 2. Feature Engineering
    features_df, target_col = prepare_data(index_df, portfolio_df)

    # 3. Zielvariable erstellen
    X, y = create_target_variable(features_df, target_col, forecast_horizon=1)

    # 4. Konvertiere zu numpy arrays
    X_np = X.values
    y_np = y.values

    # 5. Training
    model, scaler, metrics = train_neural_network(X_np, y_np)

    print("\n" + "="*70)
    print("TRAINING ABGESCHLOSSEN!")
    print("="*70)
    print(f"\nFinaler Test R² Score: {metrics['r2_score']:.6f}")
    print(f"RMSE: {metrics['rmse']:.6f}")
    print(f"MAE: {metrics['mae']:.6f}")

    print("\nErstellte Dateien:")
    print("  - neural_network_model.pth (Gespeichertes Modell)")
    print("  - model_metrics.txt (Metriken)")
    print("  - train_set.png (Visualisierung)")
    print("  - validation_set.png (Visualisierung)")
    print("  - test_set.png (Visualisierung)")


if __name__ == "__main__":
    main()

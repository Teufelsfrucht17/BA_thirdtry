# Neural Network Regressionsmodell für Finanzmarkt-Daten

## Überblick
Dieses Projekt implementiert ein Deep Learning Regressionsmodell mit PyTorch und Scikit-learn zur Vorhersage von Finanzmarkt-Returns. Der Fokus liegt auf:
- **Momentum-Features**: Berechnung verschiedener Momentum-Indikatoren (ROC, RSI, MACD, etc.)
- **Noise Filtering**: Robuste Data Preparation mit Outlier Removal und Signal-Filtering
- **Deep Neural Network**: Multi-Layer Architektur mit Batch Normalization und Dropout

## Projektstruktur

```
BA_thirdtry/
├── DataStorage/
│   ├── INDEX.xlsx          # Index-Daten
│   └── Portfolio.xlsx      # Portfolio-Daten
├── neural_network_model.py # Modell-Definitionen und Klassen
├── train_model.py          # Hauptskript für Training
├── data_analysis.py        # Datenexploration
└── requirements.txt        # Python-Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

## Verwendung

### 1. Datenanalyse durchführen
```bash
python data_analysis.py
```

### 2. Modell trainieren
```bash
python train_model.py
```

Das Skript führt folgende Schritte aus:
1. Lädt Excel-Dateien aus DataStorage/
2. Führt Feature Engineering durch (Momentum, Returns, Volatilität, etc.)
3. Filtert Noise und entfernt Outliers
4. Trainiert Deep Neural Network mit Early Stopping
5. Evaluiert auf Train/Val/Test Sets
6. Speichert Modell und Metriken

## Features

### Momentum-Indikatoren
- **Returns**: 1-Tage, 5-Tage, 10-Tage, 20-Tage Returns
- **Momentum**: Simple Momentum über verschiedene Windows (5, 10, 20, 50 Tage)
- **ROC**: Rate of Change
- **Moving Averages**: SMA und EMA (5, 10, 20, 50 Tage)
- **Price-to-MA Ratios**: Verhältnis von Preis zu Moving Averages
- **Volatility**: Rolling Standardabweichung der Returns
- **RSI**: Relative Strength Index (14 Tage)
- **MACD**: Moving Average Convergence Divergence mit Signal und Differenz

### Data Preparation
- **Missing Values**: Interpolation und Forward/Backward Fill
- **Outlier Removal**: IQR-Methode mit konfigurierbarem Faktor
- **Savitzky-Golay Filter**: Optional für zusätzliche Noise Reduction
- **Robust Scaling**: Verwendet RobustScaler für bessere Outlier-Handhabung

### Neural Network Architektur
- Input Layer: Dynamisch basierend auf Feature-Anzahl
- Hidden Layers: [256, 128, 64, 32] Neuronen (konfigurierbar)
- Batch Normalization nach jedem Layer
- Dropout (30%) zur Regularisierung
- ReLU Aktivierungsfunktion
- Output Layer: Single Neuron für Regression

### Training Features
- **Optimizer**: Adam mit Weight Decay (1e-5)
- **Learning Rate Scheduling**: ReduceLROnPlateau
- **Early Stopping**: Patience von 20 Epochen
- **Batch Size**: 32
- **Max Epochs**: 150

## Ausgabe

Nach dem Training werden folgende Dateien erstellt:

1. **neural_network_model.pth**: Gespeichertes PyTorch-Modell und Scaler
2. **model_metrics.txt**: R² Score, MSE, RMSE, MAE für Train/Val/Test
3. **train_set.png**: Visualisierung der Vorhersagen vs. Actual (Train)
4. **validation_set.png**: Visualisierung (Validation)
5. **test_set.png**: Visualisierung (Test)

## Metriken

Das Modell wird evaluiert anhand:
- **R² Score**: Bestimmtheitsmaß (Ziel: möglichst nah an 1.0)
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error

## Anpassungen

### Modell-Architektur ändern
In `train_model.py`, Zeile ~210:
```python
model = NeuralNetworkRegressor(
    input_size=input_size,
    hidden_sizes=[256, 128, 64, 32],  # Anpassen
    dropout_rate=0.3                   # Anpassen
)
```

### Forecast Horizon ändern
In `train_model.py`, Zeile ~165:
```python
X, y = create_target_variable(features_df, target_col, forecast_horizon=1)  # Ändern auf 5, 10, etc.
```

### Momentum Windows anpassen
In `neural_network_model.py`, MomentumFeatureEngineering Klasse:
```python
def calculate_momentum(prices: pd.Series, windows: List[int] = [5, 10, 20, 50])  # Anpassen
```

## Technische Details

### Device Support
- Automatische Erkennung von CUDA/GPU
- Falls verfügbar: Training auf GPU
- Fallback auf CPU

### Data Split
- Train: 70%
- Validation: 10%
- Test: 20%
- Zeitreihen-Split (shuffle=False)

## Nächste Schritte

1. **Hyperparameter Tuning**: Grid Search oder Random Search für optimale Parameter
2. **Feature Selection**: Verwende Feature Importance Methoden
3. **Ensemble Methods**: Kombiniere mehrere Modelle
4. **LSTM/Transformer**: Teste sequenzielle Modelle für Zeitreihen
5. **Cross-Validation**: Time Series Cross-Validation implementieren

## Lizenz

MIT License

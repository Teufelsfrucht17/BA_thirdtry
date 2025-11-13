"""
Neural Network Regressionsmodell für Finanzmarkt-Daten
Mit Fokus auf Momentum-Features und Noise-Filtering
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class FinancialDataset(Dataset):
    """PyTorch Dataset für Finanzdaten"""

    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class NeuralNetworkRegressor(nn.Module):
    """
    Deep Neural Network für Regression
    Architektur: Input -> Hidden Layers (mit Dropout & Batch Normalization) -> Output
    """

    def __init__(self, input_size: int, hidden_sizes: List[int] = [256, 128, 64, 32],
                 dropout_rate: float = 0.3):
        super(NeuralNetworkRegressor, self).__init__()

        layers = []
        prev_size = input_size

        # Hidden Layers mit BatchNorm und Dropout
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output Layer
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MomentumFeatureEngineering:
    """Berechnet Momentum und technische Indikatoren"""

    @staticmethod
    def calculate_returns(prices: pd.Series, periods: List[int] = [1, 5, 10, 20]) -> pd.DataFrame:
        """Berechnet Returns über verschiedene Perioden"""
        returns_df = pd.DataFrame()
        for period in periods:
            returns_df[f'return_{period}d'] = prices.pct_change(period)
        return returns_df

    @staticmethod
    def calculate_momentum(prices: pd.Series, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Berechnet Momentum-Indikatoren"""
        momentum_df = pd.DataFrame()

        for window in windows:
            # Simple Momentum (Preis / Preis vor n Tagen - 1)
            momentum_df[f'momentum_{window}d'] = prices / prices.shift(window) - 1

            # Rate of Change (ROC)
            momentum_df[f'roc_{window}d'] = (prices - prices.shift(window)) / prices.shift(window) * 100

        return momentum_df

    @staticmethod
    def calculate_moving_averages(prices: pd.Series, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        """Berechnet gleitende Durchschnitte und deren Verhältnisse"""
        ma_df = pd.DataFrame()

        for window in windows:
            ma_df[f'sma_{window}'] = prices.rolling(window=window).mean()
            ma_df[f'ema_{window}'] = prices.ewm(span=window, adjust=False).mean()
            # Verhältnis Preis zu MA
            ma_df[f'price_to_sma_{window}'] = prices / ma_df[f'sma_{window}']

        return ma_df

    @staticmethod
    def calculate_volatility(prices: pd.Series, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """Berechnet Volatilität (Standardabweichung der Returns)"""
        vol_df = pd.DataFrame()
        returns = prices.pct_change()

        for window in windows:
            vol_df[f'volatility_{window}d'] = returns.rolling(window=window).std()

        return vol_df

    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Berechnet Relative Strength Index (RSI)"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Berechnet MACD (Moving Average Convergence Divergence)"""
        macd_df = pd.DataFrame()
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd_df['macd'] = ema_fast - ema_slow
        macd_df['macd_signal'] = macd_df['macd'].ewm(span=signal, adjust=False).mean()
        macd_df['macd_diff'] = macd_df['macd'] - macd_df['macd_signal']
        return macd_df


class DataPreparation:
    """Data Cleaning und Noise Filtering"""

    @staticmethod
    def remove_outliers_iqr(df: pd.DataFrame, columns: List[str], factor: float = 1.5) -> pd.DataFrame:
        """Entfernt Outliers mit IQR-Methode"""
        df_clean = df.copy()

        for col in columns:
            if col in df_clean.columns:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

        return df_clean

    @staticmethod
    def apply_savgol_filter(series: pd.Series, window_length: int = 11, polyorder: int = 3) -> pd.Series:
        """Anwendung von Savitzky-Golay Filter für Noise Reduction"""
        from scipy.signal import savgol_filter

        # Sicherstellen, dass window_length ungerade ist
        if window_length % 2 == 0:
            window_length += 1

        # Sicherstellen, dass window_length > polyorder
        if window_length <= polyorder:
            window_length = polyorder + 2
            if window_length % 2 == 0:
                window_length += 1

        filtered = savgol_filter(series.fillna(method='ffill').fillna(method='bfill'),
                                window_length, polyorder)
        return pd.Series(filtered, index=series.index)

    @staticmethod
    def handle_missing_values(df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """Behandelt Missing Values"""
        df_clean = df.copy()

        if method == 'interpolate':
            df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        elif method == 'ffill':
            df_clean = df_clean.fillna(method='ffill').fillna(method='bfill')
        elif method == 'drop':
            df_clean = df_clean.dropna()

        return df_clean


class ModelTrainer:
    """Trainiert und evaluiert das Neural Network"""

    def __init__(self, model: nn.Module, device: str = 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def train_epoch(self, train_loader: DataLoader, criterion, optimizer) -> float:
        """Trainiert eine Epoche"""
        self.model.train()
        total_loss = 0.0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device).view(-1, 1)

            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader: DataLoader, criterion) -> float:
        """Validiert das Modell"""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device).view(-1, 1)

                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(self, train_loader: DataLoader, val_loader: DataLoader,
              epochs: int = 100, learning_rate: float = 0.001, patience: int = 15):
        """Haupttraining mit Early Stopping"""
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                          factor=0.5, patience=5, verbose=True)

        epochs_no_improve = 0

        print(f"\n{'='*50}")
        print(f"Training Neural Network")
        print(f"{'='*50}")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            val_loss = self.validate(val_loader, criterion)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            scheduler.step(val_loss)

            # Early Stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = self.model.state_dict().copy()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break

        # Lade bestes Modell
        self.model.load_state_dict(self.best_model_state)
        print(f"\nBest Validation Loss: {self.best_val_loss:.6f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Macht Vorhersagen"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor).cpu().numpy()
        return predictions.flatten()

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluiert das Modell"""
        predictions = self.predict(X)

        metrics = {
            'r2_score': r2_score(y, predictions),
            'mse': mean_squared_error(y, predictions),
            'rmse': np.sqrt(mean_squared_error(y, predictions)),
            'mae': mean_absolute_error(y, predictions)
        }

        return metrics


def plot_results(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Predictions vs Actual"):
    """Visualisiert die Ergebnisse"""
    plt.figure(figsize=(12, 5))

    # Plot 1: Scatter plot
    plt.subplot(1, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{title} - Scatter Plot')

    # Plot 2: Time series
    plt.subplot(1, 2, 2)
    plt.plot(y_true, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title(f'{title} - Time Series')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{title.replace(" ", "_").lower()}.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_model(model: nn.Module, scaler: StandardScaler, filepath: str = 'neural_network_model.pth'):
    """Speichert Modell und Scaler"""
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': model,
        'scaler': scaler
    }, filepath)
    print(f"\nModell gespeichert: {filepath}")


if __name__ == "__main__":
    print("Neural Network Regression Model - Modul geladen")
    print("Verwende die Klassen für Training und Prediction")

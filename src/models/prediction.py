import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import logging
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import os
import joblib


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class PricePredictor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lookback_days = config["models"]["prediction"]["lookback_days"]
        self.prediction_days = config["models"]["prediction"]["prediction_days"]
        self.confidence_threshold = config["models"]["prediction"][
            "confidence_threshold"
        ]
        self.logger = logging.getLogger(__name__)
        self.model_dir = os.path.join("data", "models")
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)

    def predict(
        self, symbol: str, data: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Predict future stock prices.

        Args:
            symbol: Stock symbol to predict
            data: Optional DataFrame containing historical data

        Returns:
            Dictionary containing prediction results or None if error
        """
        try:
            # Load or prepare data
            if data is None:
                from src.market.data_loader import MarketDataLoader

                market_loader = MarketDataLoader(self.config)
                data = market_loader.get_stock_data(symbol)

            if data is None or data.empty:
                self.logger.error(f"No data available for prediction: {symbol}")
                return None

            # Prepare data for prediction
            X, y = self._prepare_data(data)

            # Load or train model
            model = self._get_model(symbol)
            if model is None:
                model = self._train_model(symbol, X, y)

            # Make prediction
            prediction = self._make_prediction(model, X[-1:])

            # Calculate confidence
            confidence = self._calculate_confidence(prediction, data)

            return {
                "predicted_price": prediction[0][0],
                "confidence": confidence,
                "horizon": self.prediction_days,
                "current_price": data["Close"].iloc[-1],
                "prediction_date": (
                    pd.Timestamp.now() + pd.Timedelta(days=self.prediction_days)
                ).strftime("%Y-%m-%d"),
            }

        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            return None

    def _prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data for model training/prediction.

        Args:
            data: DataFrame containing historical data

        Returns:
            Tuple of (X, y) arrays for model training
        """
        try:
            # Select features
            features = ["Close", "Volume", "RSI", "MACD"]
            df = data[features].copy()

            # Scale data
            scaled_data = self.scaler.fit_transform(df)

            # Create sequences
            X, y = [], []
            for i in range(
                self.lookback_days, len(scaled_data) - self.prediction_days + 1
            ):
                X.append(scaled_data[i - self.lookback_days : i])
                y.append(
                    scaled_data[i + self.prediction_days - 1, 0]
                )  # Predict Close price

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Error preparing data: {str(e)}")
            raise

    def _get_model(self, symbol: str) -> Optional[LSTMModel]:
        """
        Load existing model or return None if not found.

        Args:
            symbol: Stock symbol

        Returns:
            Loaded model or None
        """
        try:
            model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
            scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")

            if os.path.exists(model_path) and os.path.exists(scaler_path):
                model = LSTMModel(input_size=4, hidden_size=50, num_layers=2)
                model.load_state_dict(torch.load(model_path))
                model.to(self.device)
                self.scaler = joblib.load(scaler_path)
                return model

            return None

        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None

    def _train_model(self, symbol: str, X: np.ndarray, y: np.ndarray) -> LSTMModel:
        """
        Train a new LSTM model.

        Args:
            symbol: Stock symbol
            X: Input features
            y: Target values

        Returns:
            Trained model
        """
        try:
            # Convert data to PyTorch tensors
            X_tensor = torch.FloatTensor(X).to(self.device)
            y_tensor = torch.FloatTensor(y).to(self.device)

            # Create and train model
            model = LSTMModel(input_size=4, hidden_size=50, num_layers=2)
            model.to(self.device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters())

            # Training loop
            model.train()
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X_tensor)
                loss = criterion(outputs.squeeze(), y_tensor)
                loss.backward()
                optimizer.step()

            # Save model and scaler
            model_path = os.path.join(self.model_dir, f"{symbol}_model.pth")
            scaler_path = os.path.join(self.model_dir, f"{symbol}_scaler.pkl")

            torch.save(model.state_dict(), model_path)
            joblib.dump(self.scaler, scaler_path)

            return model

        except Exception as e:
            self.logger.error(f"Error training model: {str(e)}")
            raise

    def _make_prediction(self, model: LSTMModel, X: np.ndarray) -> np.ndarray:
        """
        Make prediction using the model.

        Args:
            model: Trained model
            X: Input features

        Returns:
            Predicted values
        """
        try:
            # Convert input to PyTorch tensor
            X_tensor = torch.FloatTensor(X).to(self.device)

            # Make prediction
            model.eval()
            with torch.no_grad():
                prediction = model(X_tensor)

            # Convert prediction to numpy array
            prediction = prediction.cpu().numpy()

            # Inverse transform prediction
            prediction = self.scaler.inverse_transform(
                np.concatenate([prediction, np.zeros((len(prediction), 3))], axis=1)
            )[:, 0]

            return prediction

        except Exception as e:
            self.logger.error(f"Error making prediction: {str(e)}")
            raise

    def _calculate_confidence(
        self, prediction: np.ndarray, data: pd.DataFrame
    ) -> float:
        """
        Calculate confidence score for the prediction.

        Args:
            prediction: Predicted values
            data: Historical data

        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Calculate recent volatility
            recent_volatility = data["Close"].pct_change().std()

            # Calculate prediction range
            prediction_range = (
                abs(prediction[0] - data["Close"].iloc[-1]) / data["Close"].iloc[-1]
            )

            # Calculate confidence based on volatility and prediction range
            confidence = 1 - (recent_volatility * prediction_range)

            # Ensure confidence is between 0 and 1
            confidence = max(0, min(1, confidence))

            return confidence

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5

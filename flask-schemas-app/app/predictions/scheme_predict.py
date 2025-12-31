import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from datetime import datetime, timedelta
import pickle
import os

# Database configuration
MYSQL_HOST = 'localhost'
MYSQL_USER = 'anjanghosh'
MYSQL_PASSWORD = 'MyPassword123'
MYSQL_DB = 'nps_schemes'

# Create SQLAlchemy engine
DATABASE_URI = f'mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}/{MYSQL_DB}'
engine = create_engine(DATABASE_URI)


class NavPredictor:
    """
    Machine Learning model to predict NAV values for NPS schemes
    Uses Linear Regression with time-based features
    """
    
    def __init__(self, scheme_code):
        self.scheme_code = scheme_code
        self.scaler = MinMaxScaler()
        self.model = None
        self.model_path = f'models/nav_model_{scheme_code}.pkl'
        
    def fetch_data_from_db(self):
        """Fetch historical NAV data from MySQL database"""
        query = """
            SELECT e_date, nav 
            FROM historical 
            WHERE scheme_code = %s 
            ORDER BY e_date ASC
        """
        
        # Use SQLAlchemy engine with pandas
        df = pd.read_sql(query, engine, params=(self.scheme_code,))
        
        if df.empty:
            raise ValueError(f"No data found for scheme {self.scheme_code}")
        
        # Convert date to datetime
        df['e_date'] = pd.to_datetime(df['e_date'])
        df = df.sort_values('e_date').reset_index(drop=True)
        
        return df
    
    def create_features(self, df):
        """Create time-based features for prediction"""
        df = df.copy()
        
        # Extract time features
        df['year'] = df['e_date'].dt.year
        df['month'] = df['e_date'].dt.month
        df['day'] = df['e_date'].dt.day
        df['dayofweek'] = df['e_date'].dt.dayofweek
        df['dayofyear'] = df['e_date'].dt.dayofyear
        df['quarter'] = df['e_date'].dt.quarter
        
        # Create lag features (previous NAV values)
        for lag in [1, 7, 30, 90]:
            df[f'nav_lag_{lag}'] = df['nav'].shift(lag)
        
        # Rolling statistics
        df['nav_rolling_mean_7'] = df['nav'].rolling(window=7).mean()
        df['nav_rolling_std_7'] = df['nav'].rolling(window=7).std()
        df['nav_rolling_mean_30'] = df['nav'].rolling(window=30).mean()
        df['nav_rolling_std_30'] = df['nav'].rolling(window=30).std()
        
        # Drop rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_model(self):
        """Train the prediction model"""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge
        
        # Fetch data
        df = self.fetch_data_from_db()
        print(f"Fetched {len(df)} records for scheme {self.scheme_code}")
        
        # Create features
        df = self.create_features(df)
        print(f"Created features, {len(df)} records after feature engineering")
        
        # Prepare features and target
        feature_columns = [
            'year', 'month', 'day', 'dayofweek', 'dayofyear', 'quarter',
            'nav_lag_1', 'nav_lag_7', 'nav_lag_30', 'nav_lag_90',
            'nav_rolling_mean_7', 'nav_rolling_std_7',
            'nav_rolling_mean_30', 'nav_rolling_std_30'
        ]
        
        X = df[feature_columns]
        y = df['nav']
        
        # Split data - use more recent data for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Use Ridge regression for better generalization
        self.model = Ridge(alpha=10.0)
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Calculate RMSE
        from sklearn.metrics import mean_squared_error
        train_pred = self.model.predict(X_train_scaled)
        test_pred = self.model.predict(X_test_scaled)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        print(f"Training R² Score: {train_score:.4f}")
        print(f"Testing R² Score: {test_score:.4f}")
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Testing RMSE: {test_rmse:.4f}")
        
        # Save model
        self.save_model()
        
        return {
            'train_score': train_score,
            'test_score': test_score,
            'train_rmse': float(train_rmse),
            'test_rmse': float(test_rmse),
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
    
    def save_model(self):
        """Save the trained model to disk"""
        os.makedirs('models', exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'scheme_code': self.scheme_code
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load the trained model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train the model first.")
        
        with open(self.model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        
        # Check if scaler exists in saved model
        if 'scaler' not in model_data:
            # Old model without scaler - need to retrain
            print(f"Warning: Model was trained without scaler. Retraining required.")
            os.remove(self.model_path)
            raise FileNotFoundError(f"Model needs to be retrained with new version. Please check 'Retrain Model'.")
        
        self.scaler = model_data['scaler']
        print(f"Model loaded from {self.model_path}")
    
    def predict_future(self, days=30):
        """
        Predict NAV for the next N days
        
        Args:
            days: Number of days to predict
            
        Returns:
            DataFrame with predictions
        """
        if self.model is None:
            try:
                self.load_model()
            except FileNotFoundError:
                print("Model not found. Training new model...")
                self.train_model()
        
        # Fetch latest data
        df = self.fetch_data_from_db()
        df = self.create_features(df)
        
        # Get the last date and NAV
        last_date = df['e_date'].iloc[-1]
        last_nav = df['nav'].iloc[-1]
        
        predictions = []
        current_data = df.tail(1900).copy()  # Use last 1900 days for rolling features
        
        for i in range(1, days + 1):
            # Create next date
            next_date = last_date + timedelta(days=i)
            
            # Create features for prediction
            features = {
                'year': next_date.year,
                'month': next_date.month,
                'day': next_date.day,
                'dayofweek': next_date.dayofweek,
                'dayofyear': next_date.dayofyear,
                'quarter': next_date.quarter,
                'nav_lag_1': current_data['nav'].iloc[-1] if len(current_data) >= 1 else last_nav,
                'nav_lag_7': current_data['nav'].iloc[-7] if len(current_data) >= 7 else last_nav,
                'nav_lag_30': current_data['nav'].iloc[-30] if len(current_data) >= 30 else last_nav,
                'nav_lag_90': current_data['nav'].iloc[-90] if len(current_data) >= 90 else last_nav,
                'nav_rolling_mean_7': current_data['nav'].tail(7).mean(),
                'nav_rolling_std_7': current_data['nav'].tail(7).std(),
                'nav_rolling_mean_30': current_data['nav'].tail(30).mean(),
                'nav_rolling_std_30': current_data['nav'].tail(30).std(),
            }
            
            # Make prediction
            X_pred = pd.DataFrame([features])
            X_pred_scaled = self.scaler.transform(X_pred)
            predicted_nav = self.model.predict(X_pred_scaled)[0]
            
            predictions.append({
                'date': next_date.strftime('%Y-%m-%d'),
                'predicted_nav': round(float(predicted_nav), 4)
            })
            
            # Update current_data for next iteration
            new_row = pd.DataFrame({
                'e_date': [next_date],
                'nav': [predicted_nav],
                'year': [next_date.year],
                'month': [next_date.month],
                'day': [next_date.day],
                'dayofweek': [next_date.dayofweek],
                'dayofyear': [next_date.dayofyear],
                'quarter': [next_date.quarter]
            })
            
            current_data = pd.concat([current_data, new_row], ignore_index=True)
        
        return predictions


def predict_scheme_nav(scheme_code, days=30, retrain=False):
    """
    Main function to predict NAV for a scheme
    
    Args:
        scheme_code: Scheme code to predict
        days: Number of days to predict
        retrain: Whether to retrain the model
        
    Returns:
        Dictionary with predictions and model metrics
    """
    predictor = NavPredictor(scheme_code)
    
    if retrain:
        metrics = predictor.train_model()
    else:
        metrics = {}
    
    predictions = predictor.predict_future(days=days)
    
    return {
        'scheme_code': scheme_code,
        'predictions': predictions,
        'metrics': metrics
    }


if __name__ == '__main__':
    # Example usage
    scheme_code = 'SM001005'
    
    print(f"Training model for {scheme_code}...")
    result = predict_scheme_nav(scheme_code, days=30, retrain=True)
    
    print(f"\nMetrics: {result['metrics']}")
    print(f"\nPredictions for next 30 days:")
    for pred in result['predictions'][:5]:
        print(f"  {pred['date']}: {pred['predicted_nav']}")

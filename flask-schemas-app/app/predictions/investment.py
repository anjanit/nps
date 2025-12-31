import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression

# Database configuration
DATABASE_URI = 'mysql+mysqlconnector://anjanghosh:MyPassword123@localhost/nps_schemes'

class InvestmentRecommender:
    """
    Analyzes historical NAV data to recommend best schemes for investment
    """
    
    def __init__(self):
        self.engine = create_engine(DATABASE_URI)
    
    def fetch_all_schemes_data(self, days_back=365):
        """
        Fetch historical data for all schemes
        
        Args:
            days_back: Number of days of historical data to fetch
        
        Returns:
            DataFrame with all schemes' historical data
        """
        query = """
            SELECT h.scheme_code, s.scheme_name, h.e_date, h.nav
            FROM historical h
            JOIN schemes s ON h.scheme_code = s.scheme_code
            WHERE h.e_date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
            ORDER BY h.scheme_code, h.e_date
        """
        
        df = pd.read_sql(query, self.engine, params=(days_back,))
        df['e_date'] = pd.to_datetime(df['e_date'])
        
        return df
    
    def calculate_performance_metrics(self, scheme_data):
        """
        Calculate performance metrics for a scheme
        
        Args:
            scheme_data: DataFrame with historical NAV data for a scheme
        
        Returns:
            Dictionary with performance metrics
        """
        if len(scheme_data) < 30:
            return None
        
        scheme_data = scheme_data.sort_values('e_date')
        navs = scheme_data['nav'].values
        
        # Calculate returns
        total_return = ((navs[-1] - navs[0]) / navs[0]) * 100
        
        # Calculate daily returns
        daily_returns = np.diff(navs) / navs[:-1]
        avg_daily_return = np.mean(daily_returns) * 100
        volatility = np.std(daily_returns) * 100
        
        # Calculate trend (using linear regression)
        X = np.arange(len(navs)).reshape(-1, 1)
        y = navs
        
        model = LinearRegression()
        model.fit(X, y)
        trend_slope = model.coef_[0]
        
        # Predict next 30 days
        future_X = np.arange(len(navs), len(navs) + 30).reshape(-1, 1)
        future_navs = model.predict(future_X)
        predicted_return = ((future_navs[-1] - navs[-1]) / navs[-1]) * 100
        
        # Calculate Sharpe ratio (assuming risk-free rate = 6% annual = 0.016% daily)
        risk_free_rate = 0.00016
        sharpe_ratio = (np.mean(daily_returns) - risk_free_rate) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
        
        # Recent performance (last 30 days)
        recent_navs = navs[-30:] if len(navs) >= 30 else navs
        recent_return = ((recent_navs[-1] - recent_navs[0]) / recent_navs[0]) * 100
        
        # Consistency score (higher is better, lower volatility is good)
        consistency_score = avg_daily_return / volatility if volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'trend_slope': trend_slope,
            'predicted_30day_return': predicted_return,
            'sharpe_ratio': sharpe_ratio,
            'recent_30day_return': recent_return,
            'consistency_score': consistency_score,
            'current_nav': float(navs[-1]),
            'data_points': len(navs)
        }
    
    def recommend_schemes(self, top_n=5, days_back=365, min_data_points=90):
        """
        Recommend top schemes based on multiple performance criteria
        
        Args:
            top_n: Number of top schemes to recommend
            days_back: Days of historical data to analyze
            min_data_points: Minimum data points required for analysis
        
        Returns:
            List of recommended schemes with scores and metrics
        """
        # Fetch all schemes data
        all_data = self.fetch_all_schemes_data(days_back)
        
        if all_data.empty:
            return []
        
        # Group by scheme
        scheme_groups = all_data.groupby('scheme_code')
        
        recommendations = []
        
        for scheme_code, group_data in scheme_groups:
            metrics = self.calculate_performance_metrics(group_data)
            
            if metrics is None or metrics['data_points'] < min_data_points:
                continue
            
            # Calculate composite score
            # Weight: predicted return (40%), sharpe ratio (30%), recent return (20%), consistency (10%)
            score = (
                metrics['predicted_30day_return'] * 0.4 +
                metrics['sharpe_ratio'] * 100 * 0.3 +
                metrics['recent_30day_return'] * 0.2 +
                metrics['consistency_score'] * 10 * 0.1
            )
            
            scheme_name = group_data['scheme_name'].iloc[0]
            
            recommendations.append({
                'scheme_code': scheme_code,
                'scheme_name': scheme_name,
                'score': score,
                'predicted_return': metrics['predicted_30day_return'],
                'current_nav': metrics['current_nav'],
                'volatility': metrics['volatility'],
                'sharpe_ratio': metrics['sharpe_ratio'],
                'recent_return': metrics['recent_30day_return'],
                'total_return': metrics['total_return'],
                'trend': 'Upward' if metrics['trend_slope'] > 0 else 'Downward',
                'data_points': metrics['data_points']
            })
        
        # Sort by score and return top N
        recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return recommendations[:top_n]


def get_investment_recommendations(top_n=5, days_back=365):
    """
    Get top investment recommendations
    
    Args:
        top_n: Number of recommendations to return
        days_back: Days of historical data to analyze
    
    Returns:
        List of recommended schemes
    """
    recommender = InvestmentRecommender()
    return recommender.recommend_schemes(top_n=top_n, days_back=days_back)


if __name__ == '__main__':
    # Test the recommender
    print("Analyzing schemes for investment recommendations...")
    recommendations = get_investment_recommendations(top_n=10)
    
    print(f"\nTop {len(recommendations)} Investment Recommendations:\n")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['scheme_code']} - {rec['scheme_name']}")
        print(f"   Score: {rec['score']:.2f}")
        print(f"   Predicted 30-day Return: {rec['predicted_return']:.2f}%")
        print(f"   Recent 30-day Return: {rec['recent_return']:.2f}%")
        print(f"   Current NAV: {rec['current_nav']:.2f}")
        print(f"   Volatility: {rec['volatility']:.2f}%")
        print(f"   Sharpe Ratio: {rec['sharpe_ratio']:.4f}")
        print(f"   Trend: {rec['trend']}")
        print()

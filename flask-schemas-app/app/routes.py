from flask import Blueprint, render_template, request, jsonify
from sqlalchemy import desc

main = Blueprint('main', __name__)

@main.route('/')
def index():
    from app.models import Scheme
    schemes = Scheme.query.all()
    return render_template('index.html', schemes=schemes)

@main.route('/compare')
def compare():
    from app.models import Scheme
    schemes = Scheme.query.all()
    return render_template('compare.html', schemes=schemes)

@main.route('/predict')
def predict_page():
    from app.models import Scheme
    schemes = Scheme.query.all()
    return render_template('predict.html', schemes=schemes)

@main.route('/api/scheme/<scheme_code>')
def get_scheme_data(scheme_code):
    from app.models import Historical
    data = Historical.query.filter_by(scheme_code=scheme_code)\
        .order_by(Historical.e_date)\
        .all()
    
    result = {
        'dates': [d.e_date.strftime('%Y-%m-%d') for d in data],
        'navs': [float(d.nav) for d in data]
    }
    return jsonify(result)

@main.route('/api/schemes/compare', methods=['POST'])
def compare_schemes():
    from app.models import Historical, Scheme
    
    scheme_codes = request.json.get('scheme_codes', [])
    
    if not scheme_codes:
        return jsonify({'error': 'No scheme codes provided'}), 400
    
    result = {
        'datasets': []
    }
    
    # Get all unique dates across all schemes
    all_dates = set()
    
    for scheme_code in scheme_codes:
        data = Historical.query.filter_by(scheme_code=scheme_code)\
            .order_by(Historical.e_date)\
            .all()
        
        if data:
            scheme = Scheme.query.filter_by(scheme_code=scheme_code).first()
            scheme_name = scheme.scheme_name if scheme else scheme_code
            
            dates = [d.e_date.strftime('%Y-%m-%d') for d in data]
            navs = [float(d.nav) for d in data]
            
            all_dates.update(dates)
            
            result['datasets'].append({
                'label': f'{scheme_code} - {scheme_name}',
                'data': navs,
                'dates': dates,
                'scheme_code': scheme_code
            })
    
    result['all_dates'] = sorted(list(all_dates))
    
    return jsonify(result)

@main.route('/api/predict/<scheme_code>', methods=['GET'])
def predict_nav(scheme_code):
    """
    Predict NAV values for a scheme
    Query params:
        - days: Number of days to predict (default: 30)
        - retrain: Whether to retrain the model (default: false)
    """
    from app.predictions.scheme_predict import predict_scheme_nav
    
    days = int(request.args.get('days', 30))
    retrain = request.args.get('retrain', 'false').lower() == 'true'
    
    try:
        result = predict_scheme_nav(scheme_code, days=days, retrain=retrain)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/api/train/<scheme_code>', methods=['POST'])
def train_model(scheme_code):
    """Train or retrain the prediction model for a scheme"""
    from app.predictions.scheme_predict import NavPredictor
    
    try:
        predictor = NavPredictor(scheme_code)
        metrics = predictor.train_model()
        
        return jsonify({
            'success': True,
            'scheme_code': scheme_code,
            'metrics': metrics
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@main.route('/recommend')
def recommend():
    """Display investment recommendations page"""
    return render_template('recommend.html')

@main.route('/api/recommendations', methods=['GET'])
def get_recommendations():
    """
    Get investment recommendations
    Query params:
        - top_n: Number of recommendations (default: 5)
        - days_back: Days of historical data to analyze (default: 365)
    """
    from app.predictions.investment import get_investment_recommendations
    
    top_n = int(request.args.get('top_n', 5))
    days_back = int(request.args.get('days_back', 365))
    
    try:
        recommendations = get_investment_recommendations(top_n=top_n, days_back=days_back)
        return jsonify({
            'recommendations': recommendations,
            'analysis_period': days_back,
            'count': len(recommendations)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
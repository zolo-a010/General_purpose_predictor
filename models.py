from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class PredictionApp(db.Model):
    __tablename__ = 'prediction_apps'
    id = db.Column(db.Integer, primary_key=True)
    app_name = db.Column(db.String(100), unique=True, nullable=False)
    target_column = db.Column(db.String(100), nullable=False)
    model_path = db.Column(db.String(255), nullable=False)
    
class AppFeature(db.Model):
    __tablename__ = 'app_features'
    id = db.Column(db.Integer, primary_key=True)
    app_id = db.Column(db.Integer, db.ForeignKey('prediction_apps.id'), nullable=False)
    feature_name = db.Column(db.String(100), nullable=False)
    feature_type = db.Column(db.String(50), nullable=False)
    
    # NEW: Store the unique categories as a stringified JSON list
    categories = db.Column(db.Text, nullable=True)
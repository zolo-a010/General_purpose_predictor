import os
import json
import pandas as pd
import joblib
from flask import Flask, request, jsonify, render_template
from models import db, PredictionApp, AppFeature
from ml_engine import train_and_export_model

app = Flask(__name__)

# --- Configuration ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///automl.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = './uploads'
app.config['MODEL_FOLDER'] = './models'

db.init_app(app)

# Ensure necessary directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['MODEL_FOLDER'], exist_ok=True)

# Create database tables if they don't exist
with app.app_context():
    db.create_all()

# ==========================================
# FRONTEND ROUTE
# ==========================================
@app.route('/')
def home():
    """Serves the main single-page application UI."""
    return render_template('index.html')


# ==========================================
# ADMIN SECTION (Training & Deployment)
# ==========================================
@app.route('/admin/train', methods=['POST'])
def admin_train():
    """Receives a CSV, trains the optimal model, and saves metadata to the DB."""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
        
    file = request.files['file']
    app_name = request.form.get('app_name')
    target_col = request.form.get('target_column')
    
    if not app_name or not target_col:
         return jsonify({"error": "App name and target column are required"}), 400

    csv_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(csv_path)
    
    model_path = os.path.join(app.config['MODEL_FOLDER'], f"{app_name.replace(' ', '_')}.joblib")
    
    # 1. Train the model dynamically (Filters out bad columns automatically)
    try:
        num_cols, cat_mappings = train_and_export_model(csv_path, target_col, model_path)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # 2. Save core application metadata to Database
    new_app = PredictionApp(app_name=app_name, target_column=target_col, model_path=model_path)
    db.session.add(new_app)
    db.session.flush() # Get the new ID before committing
    
    # 3. Save Numerical Features
    for col in num_cols:
        db.session.add(AppFeature(
            app_id=new_app.id, 
            feature_name=col, 
            feature_type='numeric'
        ))
        
    # 4. Save Categorical Features (along with their dropdown options)
    for col, unique_vals in cat_mappings.items():
        db.session.add(AppFeature(
            app_id=new_app.id, 
            feature_name=col, 
            feature_type='categorical', 
            categories=json.dumps(unique_vals) # Convert the python list to a JSON string
        ))
        
    db.session.commit()
    total_features = len(num_cols) + len(cat_mappings)
    return jsonify({"message": f"Successfully deployed! Extracted {total_features} highly predictive features."})


# ==========================================
# USER SECTION (Dynamic Interface & Prediction)
# ==========================================
@app.route('/api/apps', methods=['GET'])
def get_apps():
    """Returns a list of deployed models for the UI dropdown."""
    apps = PredictionApp.query.all()
    return jsonify([{"id": app.id, "name": app.app_name} for app in apps])

@app.route('/api/features/<int:app_id>', methods=['GET'])
def get_features(app_id):
    """Returns the required form fields based on the selected model."""
    features = AppFeature.query.filter_by(app_id=app_id).all()
    result = []
    
    for f in features:
        item = {"name": f.feature_name, "type": f.feature_type}
        
        # If it's a categorical feature, decode the JSON string back into a list for the UI
        if f.feature_type == 'categorical' and f.categories:
            item['categories'] = json.loads(f.categories)
            
        result.append(item)
        
    return jsonify(result)

@app.route('/api/predict', methods=['POST'])
def predict():
    """Takes user JSON data, runs it through the model, and returns the prediction."""
    data = request.json
    app_id = data.get('app_id')
    user_inputs = data.get('inputs') 
    
    # 1. Clean empty strings from user inputs
    cleaned_inputs = {k: v for k, v in user_inputs.items() if v != ""}
    
    # 2. Fetch the model and the required features from the DB
    pred_app = PredictionApp.query.get_or_404(app_id)
    features = AppFeature.query.filter_by(app_id=app_id).all()
    
    # 3. Get the exact column names the model expects
    expected_columns = [f.feature_name for f in features]
    
    # 4. Create an empty DataFrame strictly enforcing those columns.
    # This prevents Scikit-Learn from crashing if columns are missing or misaligned.
    input_df = pd.DataFrame(columns=expected_columns)
    
    # 5. Insert the user's data. Pandas automatically aligns the keys.
    # Any fields the user left blank become NaN, which the pipeline's Imputer handles safely.
    input_df.loc[0] = cleaned_inputs
    
    # 6. Load the pipeline and predict
    pipeline = joblib.load(pred_app.model_path)
    prediction = pipeline.predict(input_df)
    
    return jsonify({"prediction": str(prediction[0])})

if __name__ == '__main__':
    # Run the Flask development server
    app.run(debug=True, port=5000)
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# Classifier (Fallback for Yes/No predictions)
from sklearn.ensemble import RandomForestClassifier

# Regressors (The Tournament Candidates)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

def train_and_export_model(csv_path, target_col, model_save_path):
    df = pd.read_csv(csv_path)
    X_full = df.drop(columns=[target_col])
    y = df[target_col]
    
    num_cols = X_full.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_full.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Determine if this is Classification (Categories) or Regression (Numbers)
    is_classification = (y.dtype == 'object' or y.dtype.name == 'category' or y.nunique() < 20)
    y_numeric = pd.factorize(y)[0] if is_classification else y

    # ==========================================
    # PHASE 1: FEATURE SELECTION (Unchanged)
    # ==========================================
    selected_num_cols = []
    if len(num_cols) > 0:
        correlations = X_full[num_cols].corrwith(pd.Series(y_numeric, index=y.index)).abs()
        selected_num_cols = correlations[correlations > 0.5].index.tolist()

    selected_cat_cols = []
    if len(cat_cols) > 0:
        valid_cat_cols = [col for col in cat_cols if X_full[col].nunique() < (len(X_full) * 0.4)]
        if len(valid_cat_cols) > 0:
            oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
            X_cat_encoded = oe.fit_transform(X_full[valid_cat_cols].astype(str))
            
            if is_classification:
                mi_scores = mutual_info_classif(X_cat_encoded, y_numeric, discrete_features=True, random_state=42)
            else:
                mi_scores = mutual_info_regression(X_cat_encoded, y_numeric, discrete_features=True, random_state=42)
            
            mi_series = pd.Series(mi_scores, index=valid_cat_cols)
            selected_cat_cols = mi_series[mi_series > 0.02].index.tolist()

    selected_features = selected_num_cols + selected_cat_cols
    if len(selected_features) == 0:
        raise ValueError("Filters were too strict! No valid columns found.")

    X_selected = X_full[selected_features]
    
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), selected_num_cols),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), selected_cat_cols)
    ])
    
    # ==========================================
    # PHASE 2: THE MODEL TOURNAMENT
    # ==========================================
    
    if is_classification:
        # Fallback: If user uploads Classification data (e.g. Placed: Yes/No), 
        # we skip the RMSE tournament and use a standard Classifier.
        final_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
        final_pipeline.fit(X_selected, y)
    
    else:
        # REGRESSION TOURNAMENT: Splitting data 80% Train / 20% Test
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y_numeric, test_size=0.2, random_state=42)
        
        # Define the candidate models
        candidate_models = {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
            "SVR": SVR(),
            "GradientBoosting": GradientBoostingRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42)
        }
        
        best_rmse = float('inf')
        best_pipeline = None
        best_model_name = ""
        
        # Evaluate each model
        for name, model in candidate_models.items():
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])
            
            # Train on 80%
            pipeline.fit(X_train, y_train)
            
            # Predict on 20%
            preds = pipeline.predict(X_test)
            
            # Calculate RMSE
            rmse = np.sqrt(mean_squared_error(y_test, preds))
            print(f"[{name}] RMSE: {rmse:.4f}") # Prints to your terminal for debugging
            
            # Track the winner
            if rmse < best_rmse:
                best_rmse = rmse
                best_pipeline = pipeline
                best_model_name = name
                
        print(f"🏆 WINNER: {best_model_name} with RMSE: {best_rmse:.4f}")
        
        # (Optional but recommended Data Science practice): 
        # Retrain the winning architecture on 100% of the data before saving it for production!
        best_pipeline.fit(X_selected, y_numeric)
        final_pipeline = best_pipeline

    # Save the absolute best model
    joblib.dump(final_pipeline, model_save_path)
    
    # Extract unique categories for dropdowns
    cat_mappings = {}
    for col in selected_cat_cols:
        cat_mappings[col] = X_full[col].dropna().astype(str).unique().tolist()
    
    return selected_num_cols, cat_mappings
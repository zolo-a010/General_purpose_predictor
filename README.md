# 🚀 General-Purpose AutoML Prediction Engine

A dynamic, full-stack Auto Machine Learning (AutoML) web application built with **Flask**, **Scikit-Learn**, and **SQLite**. 

This platform allows administrators to upload any raw CSV dataset. The engine automatically cleans the data, performs feature selection, runs a "Model Tournament" to find the most accurate algorithm, and deploys it. The frontend then dynamically generates a user-friendly prediction form based *only* on the optimized features.

## ✨ Key Features

* **🧠 Automatic Feature Selection:** Drops noisy data automatically. Uses Pearson Correlation (>0.5) for numerical data and Mutual Information for categorical data to ensure only highly predictive columns are used.
* **🏆 The Model Tournament:** Automatically splits data (80/20) and trains multiple candidate models (Random Forest, XGBoost, Ridge, Lasso, Gradient Boosting, etc.). It calculates the RMSE for each and permanently deploys the statistical winner.
* **⚡ Dynamic UI Generation:** The frontend requires zero hardcoding. It queries the database to understand what features the winning model requires and generates the exact necessary inputs (e.g., number fields for numeric data, dropdown menus for categorical data).
* **🛡️ Bulletproof Data Pipelines:** Utilizes `scikit-learn` Pipelines (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) to seamlessly handle missing user inputs and scale data without crashing.

## 🛠️ Tech Stack

* **Backend Server:** Python, Flask
* **Database:** SQLite (via Flask-SQLAlchemy)
* **Machine Learning:** Scikit-Learn, XGBoost, Pandas, NumPy, Joblib
* **Frontend:** HTML5, CSS3, Vanilla JavaScript (Fetch API)

## 📂 Project Structure

```text
automl_project/
│
├── app.py                  # Main Flask server and API routes
├── ml_engine.py            # Core ML logic (Feature selection, Model Tournament)
├── models.py               # SQLite database schema (SQLAlchemy)
├── requirements.txt        # Python dependencies
│
├── templates/
│   └── index.html          # Dynamic Single-Page Application (UI)
│
├── uploads/                # (Auto-created) Stores raw CSVs uploaded by Admin
└── models/                 # (Auto-created) Stores the winning .joblib model files
🚀 Getting Started
Prerequisites
Make sure you have Python 3.9+ installed on your machine.

1. Installation
Clone this repository and navigate into the project directory:

Bash
git clone [https://github.com/yourusername/automl-engine.git](https://github.com/yourusername/automl-engine.git)
cd automl-engine
Install the required Python dependencies:

Bash
pip install -r requirements.txt
2. Running the Application
Start the Flask development server. The database and necessary folders will be generated automatically on the first run.

Bash
python app.py
Open your web browser and navigate to: http://127.0.0.1:5000/

📖 How to Use the Platform
Phase 1: Admin Deployment
Go to the Admin section of the web interface.

Enter a name for your application (e.g., "Housing Price Predictor").

Enter the exact name of the column you want to predict (e.g., "SalePrice").

Upload your raw .csv dataset.

Click Train & Deploy.

Behind the scenes, the server will extract features, run the Model Tournament, and save the winning .joblib pipeline to your disk.

Phase 2: User Prediction
Scroll down to the User section.

Select your newly deployed model from the dropdown menu.

The UI will instantly communicate with the backend and generate a custom form asking only for the highly correlated features.

Fill out the form and click Calculate Prediction to get real-time results from your trained model!

🤝 Contributing
Contributions are welcome! If you want to add new models to the tournament (e.g., Neural Networks) or improve the UI, feel free to fork the repository and submit a pull request.
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import io

# Machine Learning - Model Selection & Evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

app = Flask(__name__)

# Function to process uploaded file
def process_file(file):
    # Read the uploaded CSV file content into a DataFrame
    df = pd.read_csv(io.BytesIO(file.read()))

    # Display initial rows of the DataFrame
    initial_rows = df.head().to_html()

    # Perform further processing (e.g., machine learning tasks)
    # Dummy processing for demonstration
    X = df.drop('Default', axis=1) if 'Default' in df.columns else df.iloc[:, :-1]
    y = df['Default'] if 'Default' in df.columns else None

    # Dummy machine learning tasks for demonstration
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)),
        ('lr', LogisticRegression(max_iter=1000))
    ]
    meta_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    stacking_model = StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
    stacking_model.fit(X_train_scaled, y_train)
    y_pred = stacking_model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    # Return initial rows HTML and evaluation metrics
    return initial_rows, accuracy, cr, cm

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('upload.html', error="No file part in the request")

        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', error="No selected file")

        if file:
            initial_rows, accuracy, cr, cm = process_file(file)
            return render_template('result.html', initial_rows=initial_rows, accuracy=accuracy, cr=cr, cm=cm)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)

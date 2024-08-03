# smart_healthcare_monitoring_system.py

import time
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# Directory to save model
model_dir = 'model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Function to simulate sensor data
def get_vital_signs():
    return {
        'heart_rate': random.randint(60, 100),
        'systolic_bp': random.randint(110, 130),
        'diastolic_bp': random.randint(70, 90),
        'temperature': round(random.uniform(36.5, 37.5), 1)
    }

def simulate_sensor_data():
    while True:
        vitals = get_vital_signs()
        print(f"Heart Rate: {vitals['heart_rate']}, Blood Pressure: {vitals['systolic_bp']}/{vitals['diastolic_bp']}, Temperature: {vitals['temperature']}")
        time.sleep(5)  # Simulating data collection every 5 seconds

# Function to train the model
def train_model():
    # Generate synthetic data for illustration
    data = pd.DataFrame({
        'heart_rate': [random.randint(60, 100) for _ in range(1000)],
        'systolic_bp': [random.randint(110, 130) for _ in range(1000)],
        'diastolic_bp': [random.randint(70, 90) for _ in range(1000)],
        'temperature': [round(random.uniform(36.5, 37.5), 1) for _ in range(1000)],
        'condition': [random.choice([0, 1]) for _ in range(1000)]  # 0: Normal, 1: Emergency
    })

    # Data Preprocessing
    X = data[['heart_rate', 'systolic_bp', 'diastolic_bp', 'temperature']]
    y = data['condition']

    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Model Evaluation
    predictions = model.predict(X_test)
    print(classification_report(y_test, predictions))

    # Save the model
    joblib.dump(model, os.path.join(model_dir, 'health_monitoring_model.pkl'))

# Function for real-time monitoring
def monitor_patient():
    # Load the trained model
    model_path = os.path.join(model_dir, 'health_monitoring_model.pkl')
    if not os.path.exists(model_path):
        print("Model not found. Training model first...")
        train_model()

    model = joblib.load(model_path)

    while True:
        vitals = get_vital_signs()
        features = [[vitals['heart_rate'], vitals['systolic_bp'], vitals['diastolic_bp'], vitals['temperature']]]
        condition = model.predict(features)[0]
        
        print(f"Heart Rate: {vitals['heart_rate']}, Blood Pressure: {vitals['systolic_bp']}/{vitals['diastolic_bp']}, Temperature: {vitals['temperature']}, Condition: {'Emergency' if condition == 1 else 'Normal'}")

        if condition == 1:
            # Send alert to healthcare providers
            print("ALERT: Emergency condition detected!")

        time.sleep(5)

if __name__ == "__main__":
    print("Starting Smart Healthcare Monitoring System...")

    # Uncomment the following line if the model needs to be retrained
    # train_model()
    
    monitor_patient()

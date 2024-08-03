# Smart Healthcare Monitoring System

This project demonstrates a Smart Healthcare Monitoring System using IoT sensors and Machine Learning to monitor patients' vital signs in real-time and predict potential health emergencies.

## Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/SmartHealthcareMonitoringSystem.git
    cd SmartHealthcareMonitoringSystem
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Train the model (optional):
    Uncomment the `train_model()` function call in the `if __name__ == "__main__"` block of `smart_healthcare_monitoring_system.py`.

4. Run the real-time monitoring system:
    ```bash
    python smart_healthcare_monitoring_system.py
    ```

## Project Components

### 1. Simulate Sensor Data
- Simulates real-time sensor data for heart rate, blood pressure, and temperature.

### 2. Train the Model
- Trains a Random Forest model on synthetic health data and saves the model.

### 3. Real-Time Monitoring
- Monitors patients' vital signs in real-time and predicts health conditions using the trained model.

## Contributions

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License.

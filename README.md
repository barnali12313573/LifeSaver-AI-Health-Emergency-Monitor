LifeSaver AI is a real-time health dashboard designed to detect medical emergencies using a hybrid machine learning approach. By analyzing live vitals—such as Heart Rate, Blood Oxygen (SpO2), and Blood Pressure—the system employs both Supervised Ensemble Learning and Unsupervised Anomaly Detection to trigger automated emergency protocols.

🚀 Key Features
Hybrid ML Architecture: Combines supervised models (Logistic Regression, Random Forest, SVM) with unsupervised clustering (K-Means) for robust risk assessment.

Intelligent Emergency Protocol: Automatically initiates a 30-second countdown when high risk is detected. If the user does not respond, the system simulates an emergency call to pre-configured numbers.

Dynamic UI: Features a "Classy Black" high-contrast neon theme designed for visibility in emergency situations.

Real-time Anomaly Detection: Uses K-Means distance-based scoring to identify physiological patterns that deviate from the "normal" baseline.

Automated Audio Alerts: Integrates AI voice protocols to interact with the user during critical events.

🧠 Machine Learning Overview
Supervised Ensemble
The app uses an ensemble of three models to calculate an Emergency Probability Score:

Logistic Regression: For baseline linear probability.

Random Forest: To capture non-linear relationships in vitals.

Support Vector Machine (SVM): For high-dimensional classification accuracy.

Unsupervised Detection
Using K-Means Clustering, the system calculates the Euclidean distance of live data points from a "normal health" centroid. A high distance score indicates a physiological anomaly, even if supervised models haven't been specifically trained on that exact scenario.
Technical Stack:
Frontend: Streamlit (Python-based web framework)

Machine Learning: Scikit-Learn

Data Processing: Pandas, NumPy

Model Management: Joblib (for serialization/loading)

Styling: Custom CSS injection for the neon-black aesthetic

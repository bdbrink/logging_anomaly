import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TfidfVectorizer
from sklearn.ensemble import IsolationForest  # Replace with your chosen anomaly detector

# Define functions for data preprocessing (modify as needed)
def parse_log_message(message):
    # Sample logic to extract features from log message (adapt based on your log format)
    features = {}
    split_message = message.split(" ", maxsplit=2)  # Split on space, keeping max 3 parts

    # Extract timestamp (assuming it's the first part)
    features["timestamp"] = split_message[0]

    # Extract log level (assuming it's the second part)
    features["log_level"] = split_message[1] if len(split_message) > 1 else "INFO"  # Default level

    # Extract message content (assuming it's the rest)
    features["message"] = " ".join(split_message[2:]) if len(split_message) > 2 else ""
    return features

def preprocess_data(data):
    # Parse log messages
    features = data["message"].apply(parse_log_message)

    # Handle categorical features (e.g., log level)
    categorical_encoder = OneHotEncoder(sparse=False)
    categorical_features = pd.DataFrame(categorical_encoder.fit_transform(features[categorical_columns]))

    # Handle text features (e.g., message content)
    tfidf_vectorizer = TfidfVectorizer()
    text_features = pd.DataFrame(tfidf_vectorizer.fit_transform(features["message"]))

    # Combine features
    preprocessed_data = pd.concat([features, categorical_features, text_features], axis=1)
    return preprocessed_data

# Load your log data
data = pd.read_csv("your_logs.csv")

# Preprocess data
preprocessed_data = preprocess_data(data.copy())

# Split data into training and testing sets
X_train, X_test = train_test_split(preprocessed_data, test_size=0.2, random_state=42)

# Define and train the anomaly detection model
model = IsolationForest(contamination=0.01)  # Adjust contamination parameter as needed
model.fit(X_train)

# Function to predict anomalies on new data
def predict_anomaly(log_message):
    # Preprocess the new message
    new_data = pd.DataFrame([parse_log_message(log_message)]).pipe(preprocess_data)
    # Get anomaly score
    score = model.decision_function(new_data)[0]
    # Check if score exceeds a threshold (define your threshold)
    return score > threshold

new_log_message = "This is an unusual error message"
if predict_anomaly(new_log_message):
    print("Anomaly detected in log message!")

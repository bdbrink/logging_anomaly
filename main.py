import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TfidfVectorizer
from sklearn.ensemble import IsolationForest  # Replace with your chosen anomaly detector

# Maintain a counter for logs processed in the current time window
log_count_window = 0
# Define the time window size (e.g., in seconds)
window_size = 60  

# Define functions for data preprocessing (modify as needed)
def parse_log_message(message):
    # Sample logic to extract features from log message (adapt based on your log format)
    features = {}
    
    # Define regular expressions for message and body patterns (modify as needed)
    message_pattern = r"(?P<message>.*?):"  # Capture everything before colon (:)
    body_pattern = r"(?P<body>.*)"  # Capture everything after colon (:)

    # Search for message and body patterns
    match = re.search(message_pattern + body_pattern, message)

    # Extract features based on match results
    if match:
        features["message"] = match.group("message")
        features["body"] = match.group("body")
    else:
        # Handle cases where message and body patterns are not found
        features["message"] = message
        features["body"] = ""  # Set body as empty string if not captured
    
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

def predict_anomaly(log_message):
    global log_count_window

    # Preprocess the new message
    new_data = pd.DataFrame([parse_log_message(log_message)]).pipe(preprocess_data)
    
    # Feature anomaly score
    feature_score = model.decision_function(new_data)[0]

    # Update log count and reset if window expires
    log_count_window += 1
    if time.time() - window_start_time > window_size:
        log_count_window = 0
        window_start_time = time.time()

    # Rate anomaly score (implement your chosen anomaly detection for log count)
    # This is a placeholder, replace with your implementation
    rate_score = detect_rate_anomaly(log_count_window)  

    # Combine scores with weights (adjust weights as needed)
    combined_score = 0.7 * feature_score + 0.3 * rate_score

    # Check if combined score exceeds a threshold (define your threshold)
    return combined_score > threshold


new_log_message = "This is an unusual error message"
if predict_anomaly(new_log_message):
    print("Anomaly detected in log message!")

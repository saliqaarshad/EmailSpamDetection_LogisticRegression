import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
data = pd.read_csv('enron_spam_data.csv')
print('Data Loaded successfully')
print('Processing Data....')

data.dropna(inplace=True)

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['Message'] = data['Message'].apply(preprocess_text)
data['Subject'] = data['Subject'].apply(preprocess_text)

# Separate vectorizers for message and subject
vectorizer_message = TfidfVectorizer(stop_words='english', max_features=1000)
vectorizer_subject = TfidfVectorizer(stop_words='english', max_features=1000)

X_message = vectorizer_message.fit_transform(data['Message']).toarray()
X_subject = vectorizer_subject.fit_transform(data['Subject']).toarray()

# Combined features
X_combined = np.hstack((X_message, X_subject))
y = data['Spam/Ham'].apply(lambda x: 1 if x == 'spam' else 0).values

x_features = {
    'Message': X_message,
    'Subject': X_subject,
    'Combined': X_combined
}

accuracy = []
conf_matrix = []
class_reports = []
precision = []
recall = []
f1 = []

print('Processed data successfully')
print('Training Model....')

# Train for each feature set
for feature_name, x in x_features.items():
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred))
    conf_matrix.append(confusion_matrix(y_test, y_pred))
    
    class_report = classification_report(y_test, y_pred, output_dict=True)
    class_reports.append(class_report)
    precision.append(class_report['1']['precision'])
    recall.append(class_report['1']['recall'])
    f1.append(class_report['1']['f1-score'])

# Best model
best_index = accuracy.index(max(accuracy))
best_feature = list(x_features.keys())[best_index]

print(f"Best Feature Set: {best_feature}")
print(f"Accuracy: {accuracy[best_index]:.2f}")
print(f"Confusion Matrix:\n{conf_matrix[best_index]}")
print(f"Precision: {precision[best_index]:.2f}")
print(f"Recall: {recall[best_index]:.2f}")
print(f"F1 Score: {f1[best_index]:.2f}")

# Save the best model & corresponding vectorizers
final_model = LogisticRegression(max_iter=1000)
final_model.fit(x_features[best_feature], y)

joblib.dump(final_model, 'spam_model.pkl')
joblib.dump(vectorizer_message, 'vectorizer_message.pkl')
joblib.dump(vectorizer_subject, 'vectorizer_subject.pkl')
joblib.dump(best_feature, 'best_feature.pkl')

print("Model and vectorizers saved successfully.")

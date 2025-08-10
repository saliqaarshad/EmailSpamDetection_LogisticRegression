import streamlit as st
import joblib
import numpy as np

# Load saved model, vectorizers, and best feature set
model = joblib.load('spam_model.pkl')
vectorizer_message = joblib.load('vectorizer_message.pkl')
vectorizer_subject = joblib.load('vectorizer_subject.pkl')
best_feature = joblib.load('best_feature.pkl')

st.title("📧 Email Spam Detection App (Logistic Regression)")
st.write(f"**Using Feature Set:** {best_feature}")

subject_input = st.text_input("Subject")
message_input = st.text_area("Message")

if st.button("Predict"):
    if subject_input.strip() == "" or message_input.strip() == "":
        st.warning("Please fill in both Subject and Message.")
    else:
        X_m = vectorizer_message.transform([message_input]).toarray()
        X_s = vectorizer_subject.transform([subject_input]).toarray()

        if best_feature == "Message":
            X_input = X_m
        elif best_feature == "Subject":
            X_input = X_s
        else:  # Combined
            X_input = np.hstack((X_m, X_s))

        prediction = model.predict(X_input)[0]
        probability = model.predict_proba(X_input)[0][1] * 100  # Spam probability

        if prediction == 1:
            st.error(f"🚨 Spam Detected! ({probability:.2f}% confidence)")
        else:
            st.success(f"✅ This email is not spam ({100 - probability:.2f}% confidence)")

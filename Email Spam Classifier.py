import os
import email
from email.policy import default
import pandas as pd

def parse_email(file_path):
    """Parse an email file to extract the subject and body."""
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        msg = email.message_from_file(f, policy=default)
    subject = msg.get('Subject', '')
    if msg.is_multipart():
        body = ''.join(
            part.get_payload(decode=True).decode('utf-8', errors='ignore') 
            for part in msg.walk() if part.get_content_type() == 'text/plain'
        )
    else:
        body = msg.get_payload(decode=True).decode('utf-8', errors='ignore')
    return subject, body

def load_email_data(folder_path, label):
    """Load emails from a folder and assign a label."""
    data = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            subject, body = parse_email(file_path)
            data.append({'subject': subject, 'body': body, 'label': label})
    return pd.DataFrame(data)

def load_datasets(ham_folder, spam_folder):
    """Load ham and spam emails into a combined dataset."""
    ham_data = load_email_data(ham_folder, label=0)  # Label 0 for ham
    spam_data = load_email_data(spam_folder, label=1)  # Label 1 for spam
    return pd.concat([ham_data, spam_data], ignore_index=True)


import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    """Clean and tokenize text."""
    text = re.sub(r'[^\w\s]', '', text.lower())  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

def preprocess_dataset(email_data):
    """Apply preprocessing to the dataset."""
    email_data['cleaned_body'] = email_data['body'].apply(preprocess_text)
    return email_data

def save_to_csv(email_data):

    # Define the directory and file name
    directory = r'C:\Users\win 10\OneDrive\Documents\Vs Code\Email Spam Classifier'
    file_name = 'emails_dataset.csv'
    output_path = os.path.join(directory, file_name)

    # Ensure the directory exists
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

    email_data.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

def train_model(email_data):
    # Feature extraction
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(email_data['cleaned_body'])
    y = email_data['label']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    return model


if __name__ == '__main__':
    ham_folder = 'C:/Users/win 10/OneDrive/Documents/Vs Code/Email Spam Classifier/ham'
    spam_folder = 'C:/Users/win 10/OneDrive/Documents/Vs Code/Email Spam Classifier/spam'
    output_csv = 'C:/Users/win 10/OneDrive/Documents/Vs Code/EmaiEmail Spam Classifier Spam Cl'
    
    # Load, preprocess, and save dataset
    email_data = load_datasets(ham_folder, spam_folder)
    email_data = preprocess_dataset(email_data)
    save_to_csv(email_data)
    
    # Train and evaluate model
    train_model(email_data)
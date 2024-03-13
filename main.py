import kivy
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.utils import get_color_from_hex
from kivy.graphics import Color
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class LanguageIdentificationApp(App):
    def build(self):
        self.title = 'Language Identification Tool'
        Window.clearcolor = get_color_from_hex('#f0f0f0')  # Light gray background color

        layout = BoxLayout(orientation='vertical', padding=dp(20), spacing=dp(20))
        self.input_text = TextInput(hint_text='Enter text in any language...', multiline=True,
                                     background_color=get_color_from_hex('#ffffff'))  # White text input
        identify_button = Button(text='Identify Language', size_hint_y=None, height=dp(50),
                                 background_color=get_color_from_hex('#007bff'))  # Blue button
        identify_button.bind(on_press=self.identify_language)
        self.result_label = Label(text='', size_hint_y=None, height=dp(50),
                                  font_size=dp(20), color=get_color_from_hex('#333333'))  # Dark text color

        layout.add_widget(self.input_text)
        layout.add_widget(identify_button)
        layout.add_widget(self.result_label)

        # Train the language detection model
        self.train_language_detection_model()

        return layout

    def preprocess_text(self, row):
        text = row['Text']
        # Lowercasing
        text = text.lower()

        # Removing special characters and punctuation
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

        # Removing numbers
        text = re.sub(r'\d+', '', text)

        # Tokenization
        text = text.split()

        # Removing stopwords
        stop_words = set(stopwords.words('english'))  # Adjust language as needed
        text = [word for word in text if word not in stop_words]

        # Stemming
        stemmer = PorterStemmer()
        text = [stemmer.stem(word) for word in text]

        return ' '.join(text)

    def train_language_detection_model(self):
        # Load your dataset
        df = pd.read_csv('Language Detection.csv')  # Adjust the file path and format as needed

        # Preprocess your dataset
        df['text'] = df.apply(self.preprocess_text, axis=1)

        # Split your dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['Language'], test_size=0.2, random_state=42)

        # Feature extraction
        vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 3))
        X_train = vectorizer.fit_transform(X_train)
        X_test = vectorizer.transform(X_test)

        # Train your model
        nb_classifier = MultinomialNB()
        nb_classifier.fit(X_train, y_train)

        # Evaluate your model
        y_pred = nb_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print("Accuracy:", accuracy)

        # Save your model and CountVectorizer for later use
        joblib.dump(nb_classifier, 'language_detection_model.pkl')
        joblib.dump(vectorizer, 'language_detection_vectorizer.pkl')

    def identify_language(self, instance):
        text = self.input_text.text.strip()
        if text:
            # Load the trained model
            nb_classifier = joblib.load('language_detection_model.pkl')

            # Load the CountVectorizer vocabulary
            vectorizer = joblib.load('language_detection_vectorizer.pkl')

            # Preprocess the input text
            text = self.preprocess_text({'Text': text})

            # Feature extraction using the loaded CountVectorizer
            X_test = vectorizer.transform([text])

            # Predict language using the trained model
            language = nb_classifier.predict(X_test)[0]

            self.result_label.text = f'Identified Language: {language}'
        else:
            self.result_label.text = 'Please enter text to identify language.'


if __name__ == '__main__':
    LanguageIdentificationApp().run()

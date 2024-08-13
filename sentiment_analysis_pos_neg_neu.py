import os
import fitz 
import torch
import nltk
import pandas as pd
import matplotlib.pyplot as plt

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from docx import Document
from wordcloud import WordCloud

nltk.download('punkt')

# Load pre-trained model and tokenizer
'''
"cardiffnlp/twitter-roberta-base-sentiment-latest" - Twitter-roBERTa-base for Sentiment Analysis - UPDATED (2022)

10.5 M downloads and 450 likes - Very popular on Hugging Face Hub

This is a RoBERTa-base model trained on ~124M tweets from January 2018 to December 2021, and finetuned for sentiment analysis 
    with the TweetEval benchmark. 

This model is suitable for English.

Labels: 0 -> Negative; 1 -> Neutral; 2 -> Positive
'''
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def read_pdf(file_path):
    document = fitz.open(file_path)
    text = ""
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        text += page.get_text()
    return text.replace('\n', ' ')

def read_docx(file_path):
    document = Document(file_path)
    text = ""
    for para in document.paragraphs:
        text += para.text
    return text.replace('\n', ' ')

def process_files(folder_path):
    files_text = {}
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith('.pdf'):
            #print(f"Processing PDF file: {file_path}")
            text = read_pdf(file_path)
            files_text[filename] = text
            #print(text)
        elif filename.endswith('.docx'):
            #print(f"Processing Word file: {file_path}")
            text = read_docx(file_path)
            files_text[filename] = text
            #print(text)
    return files_text

# Function to analyze sentiment
def analyze_sentiment(sentence):
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    logits = outputs.logits
    sentiment = torch.argmax(logits, dim=1).item()
    scores = torch.softmax(logits, dim=1).detach().numpy()[0]
    return sentiment, scores

# Specify the folder containing the PDF and Word files
folder_path = '/reflection paper submissions' #Use appropriate path here
files_text = process_files(folder_path)

overall_analysis = pd.DataFrame(columns=['Filename','Sentence','Lable','Probabilities'])

# Perform sentiment analysis on each sentence
for file_name in files_text.keys():
    sentences = sent_tokenize(files_text[file_name])
    for sentence in sentences:
        l = len(overall_analysis)
        sentiment, score = analyze_sentiment(sentence)
        if sentiment == 2:
            sentiment_label = "POSITIVE" 
        elif sentiment == 1:
            sentiment_label = "NEUTRAL" 
        elif sentiment == 0:
            sentiment_label = "NEGATIVE" 
        
        overall_analysis.loc[l,'Filename'] = file_name
        overall_analysis.loc[l,'Sentence'] = sentence
        overall_analysis.loc[l,'Lable'] = sentiment_label
        overall_analysis.loc[l,'Probabilities'] = score

overall_analysis.to_csv("overall_analysis.csv")

print("\nSummary of analysis:\n")
print(overall_analysis["Lable"].value_counts())

counts = overall_analysis['Lable'].value_counts()
percentages = counts / counts.sum() * 100

colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']

percentages.plot.pie(autopct='%1.2f%%', figsize=(8, 8), colors=colors, title='Sentiment Analysis Distribution', ylabel='')

plt.show()

# Wordcloud with positive tweets
positive_feedback = overall_analysis['Sentence'][overall_analysis["Lable"] == 'POSITIVE']
positive_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(positive_feedback))
plt.figure()
plt.title("Positive Feedback - Wordcloud")
plt.imshow(positive_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()
 
# Wordcloud with negative tweets
negative_feedback = overall_analysis['Sentence'][overall_analysis["Lable"] == 'NEGATIVE']
negative_wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(str(negative_feedback))
plt.figure()
plt.title("Negative Feedback - Wordcloud")
plt.imshow(negative_wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

import os
import fitz 
import torch
import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from docx import Document

nltk.download('punkt')

model_name = "SamLowe/roberta-base-go_emotions"
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

emotion_labels = model.config.id2label

# Perform sentiment analysis on each sentence
for file_name in files_text.keys():
    sentences = sent_tokenize(files_text[file_name])
    for sentence in sentences:
        l = len(overall_analysis)
        sentiment, score = analyze_sentiment(sentence)
        
        overall_analysis.loc[l,'Filename'] = file_name
        overall_analysis.loc[l,'Sentence'] = sentence
        overall_analysis.loc[l,'Lable'] = emotion_labels[sentiment]
        overall_analysis.loc[l,'Probabilities'] = score

overall_analysis.to_csv("overall_analysis_emotions.csv")

print("\nSummary of analysis:\n")
print(overall_analysis["Lable"].value_counts())

counts = overall_analysis['Lable'].value_counts()
percentages = counts / counts.sum() * 100

# Convert to DataFrame for Seaborn
percentages_df = pd.DataFrame({'Label': percentages.index, 'Percentage': percentages.values})

# Plotting using Seaborn
plt.figure(figsize=(10, 6))
sns.barplot(x='Label', y='Percentage', data=percentages_df)
plt.title('Emotion Analysis Distribution')
plt.ylabel('Percentage')
plt.xlabel('Sentiment Label')
plt.xticks(rotation=45)
plt.show()

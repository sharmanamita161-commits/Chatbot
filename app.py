import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import tkinter as tk

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())
model = load_model('model.h5')

words = []
classes = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words]))
classes = sorted(set(classes))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append(classes[r[0]])
    return return_list

def get_response(ints):
    tag = ints[0]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

def send():
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0", tk.END)

    if msg != '':
        ChatLog.config(state=tk.NORMAL)
        ChatLog.insert(tk.END, "You: " + msg + '\n\n')

        ints = predict_class(msg)
        res = get_response(ints)

        ChatLog.insert(tk.END, "Bot: " + res + '\n\n')
        ChatLog.config(state=tk.DISABLED)
        ChatLog.yview(tk.END)

base = tk.Tk()
base.title("AI Chatbot")
base.geometry("400x500")

ChatLog = tk.Text(base, bd=0, bg="white", height="8", width="50", font="Arial")
ChatLog.config(state=tk.DISABLED)

scrollbar = tk.Scrollbar(base, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = tk.Button(base, text="Send", command=send)

EntryBox = tk.Text(base, bd=0, bg="white", height="3", width="29", font="Arial")

scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
ChatLog.pack()
EntryBox.pack()
SendButton.pack()

base.mainloop()

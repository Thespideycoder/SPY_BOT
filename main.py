import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy as np
import tflearn
import tensorflow
import random
import json
import pickle

with open('intents.json') as file:
    data = json.load(file)
    
#print(data['intents'])    
try:
    
    with open("data.pickle", "rb") as f:
        words, labels, training , output = pickle.load(f)
    
except :  
    words = []               #all the diff words
    labels = []              #  
    docs_x = []              # all diff patterns
    docs_y = []              # all diff tags for patterns

    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
            
            if intent['tag'] not in labels :
                labels.append(intent['tag'])
    
    #remove duplicate            
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]   
    words = sorted(list(set(words)))         
            
    labels = sorted(labels)  
        
    #one hot encoding
    #[0,0,0,0,0,1,1,1,0,1]  
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]   

    for x, doc in enumerate(docs_x):
        bag = []
        
        wrds = [stemmer.stem(w) for w in doc] 
        
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]      
        output_row[labels.index(docs_y[x])] = 1   
        
        training.append(bag)
        output.append(output_row)

    tarining = np.array(training)
    output = np.array(output)
    
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training , output),f)

'''
from tensorflow.python.framework import ops
ops.reset_default_graph()'''
tensorflow.compat.v1.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,len(output[0]), activation = 'softmax')
net = tflearn.regression(net)

model = tflearn.DNN(net)

'''
try:
    model.load("Model.tflearn")
except:    
    model.fit(training,output, n_epoch=800, batch_size=8, show_metric=True)
    model.save("Model.tflearn")
    '''
model.fit(training,output, n_epoch=800, batch_size=8, show_metric=True)
model.save("Model.tflearn")
model.load("Model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i , w in enumerate(words):
            if w == se:
                bag[i] = 1
                
    return np.array(bag)            
        
def chat():
    print("Start talking with the Bot(Type quit or exit to stop)!")   
    while True:
        query = input("You: ")     
        if query.lower() == "quit":
            break
        
        results = model.predict([bag_of_words(query,words)])[0]
        results_index = np.argmax(results)         #index of the gratest value
        tag = labels[results_index]
        
        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                
        print("Bot: ",random.choice(responses))        
       
chat()        


'''
from tkinter import *
from click import command
from numpy import size
#from main import chat


                        # for creating tkinter object
root = Tk()

root.title("SPY_BOT 2.0")
root.geometry('380x500')

                        #create chat-area
chat_window = Text(root,bg='gray',width=50,height=8)
chat_window.place(x=6,y=6,height=385,width=370)

#create the message wi ndow
message_window = Text(root , bg='gray',width=30,height=4)
message_window.place(x=140,y=400,height=88,width=260)

button = Button(root,text='Send',bg='gray', activebackground='light blue',font=('Ariel',20))
button.place(x=6,y=400,height=88,width=120)

root.run()
root.mainloop()
         '''            

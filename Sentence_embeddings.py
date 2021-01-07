import tensorflow as tf
import numpy as np
import os
import tensorflow_hub as hub
import tensorflow_text

model_url="https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
model = hub.load(model_url)  # Loading the model that will create the emdeddings

counter=0
encodings=[]

my_dir='C:/Users/User/Desktop/subtitle segments'

for file in os.listdir(my_dir):  # For every file in the directory where the segmented subtitles are stored
    if file.endswith('.txt'):
        
        counter += 1
        
        if counter<=len(file):
        
            f=open(my_dir +'/subsegment{}.txt'.format(counter), "r")
               
            for segment in f:                       # For every paragraph in the .txt file
                encodings.append(segment)           # It is stored to the list 
            
                enc_result = str(model(encodings))  # The embedding is created
            
            enc_file = open(my_dir + '/word_embedding{}.txt'.format(counter), "w") # A new .txt file with the embeddings is created
            enc_file.write(enc_result)
            enc_file.close()

        else:
            
            break
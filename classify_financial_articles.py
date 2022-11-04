import pickle

import pandas as pd
from gravityai import gravityai as grav

model = pickle.load(open('financial_text_classifier.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('financial_text_vectorizer.pkl', 'rb'))
label_encoder = pickle.load(open('financial_text_encoder.pkl','rb'))

def process(inaPath, outPath):
    #read input files
    input_df = pd.read_csv(inaPath)
    #Vectorize the Data
    features = tfidf_vectorizer.transform(input_df['body'])
    #Predict the classes
    predictions = model.predict(features)
    #convert output levels to catagories
    input_df['catagory']= label_encoder.inverse_transform(predictions)
    #Save results to csv
    output_df= input_df[['id', 'catagory']]
    output_df.to_csv(outPath, index=False)

    grav.wait_for_requests(process)
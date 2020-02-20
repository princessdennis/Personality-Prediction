import pandas as pd
import os
import joblib
from keras.models import load_model
# from static.clean_text import dineise_clean_text

path = os.path.expanduser(f'~/dlproject1/')

def dineise_predict(text, top_n_results=3):
  #Load the saved pipelines and model
  clean_vec = joblib.load(os.path.join(path,'Output/clean_vec_pipeline.pkl'))
  model = load_model(os.path.join(path,"Output/mlp.h5"))
  en = joblib.load(os.path.join(path,'Output/encoder.pkl'))

  text_sr= pd.DataFrame([text]).iloc[0]   #Text must be a pd series, not pd dataframe, to put into cleaner object

  # These are objects already in memory, not loaded objects. We will use the loaded objects.
  # text_cleaned = cleaner.transform(text_sr)
  # text_vec = vectorizer.transform([text_cleaned.loc[0]])  #Transform text into tfidf, the input of this must be a string inside a list

  text_vec = clean_vec.transform(text_sr)

  # text_pred = mlp.predict_classes(text_vec)  #Output numerical encoding of class
  text_prob = model.predict(text_vec)  #Output probabilities of each class
  text_pred = text_prob[0].argsort()[-top_n_results:][::-1]  #Sort and get index of smallest to largest probability, get last "n_results" largest probabilities and then reverse the array with [::-1]

  preds =[]
  for index, pred in enumerate(text_pred):
    text_en = en.inverse_transform([pred])  #Inverse transform the numerical encoding into the label
    preds.append((index+1, text_en[0]))

  return(preds)






#
# Others texts to try out
# Lexi_text = 'NEW YORK CITY TRIP VLOG. it is 40min long and more of a documentary at this point, so just watch 14:35 - 20:26 for Patrick and Hannahs wedding banquet!!! if you are my real friend you would watch 11:05 - the end :O jk there is a table of contents in the description with time stamps'
# Dineise_text = 'Nobody does fb status anymore but I don’t have an insta and need an outlet Somebody stole my honey wheat bread at work today you. do. not. steal. peoples. honey. wheat. bread. noooooooooooooooo I’ll give you some if you ask'
# Jess_text = "LOL LEE HYORIII w o w this performance has all the feeeeelz T.T Big Bang looks so clean and cute here LOL and !!!u went to a Taeyang concert too!!! These are the songs jacob and I will play once in a while and go wow... old kpop was the best LOL"
# Kristin_text = "I love to read. Somehow I had forgotten that in college. Since last year, this is a compilation of some books I’ve read - listed in chronological order - that have challenged, shaped, and/or broadened my views as I wrestle to understand what a life well-lived means to me. They are A++ reads and I’d be glad to tell you why if you’re curious. But I am mainly posting this because I am hungry and looking for more recommendations!! I don’t even know how little I know so I am searching for more A++ reads that challenged/shaped/broadened your views (tell me why too!) let me know in the comments please 🤓📚"
# Kyla_text = "I had so much fun in Barcelona! I was with my professor and my friends, and we went to all sorts of fun places in Barcelona! Really love the food and the people too! The architecture was amazing!!!!!!!!! :)"
# Susan_text = "Surprise! Most of you may know, but Daniel and I got married on February 1st in Dekalb county court. It was very laid back, but very special. Can't wait for our late July wedding to continue the celebration <3 Thanks for congratulating our marriage :) AND SORRY FOR POSTING A LOT 😆, all of these were/are wonderful and worth sharing! Shout out to Maekoi Photography for taking these beautiful pictures <3 Jumping into October like 🍂🍁 #pumpkinpatch Rainy weather didn’t stop us from having fun 😊!"

# lexi_text, lexi_output = dineise_predict("Lexi", Lexi_text, actual="INTP")
# print(lexi_text)
# print(lexi_output)
# dineise_predict("Dineise", Dineise_text, actual="INTJ", top_n_results=5)
# dineise_predict("Jess", Jess_text, actual="ENFP")
# dineise_predict("Kristin", Kristin_text, actual="INFJ")


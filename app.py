from flask import Flask,render_template,request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download("vader_lexicon")

# ==============Hugging face's transformers library and pre trained model RoBERTa Model=======
# from transformers import AutoTokenizer,AutoConfig, AutoModelForSequenceClassification
# import torch
# from scipy.special import softmax

# MODEL="cardiffnlp/twitter-roberta-base-sentiment-latest"
# tokenizer=AutoTokenizer.from_pretrained(MODEL)
# config=AutoConfig.from_pretrained(MODEL)
# model=AutoModelForSequenceClassification.from_pretrained(MODEL)
# def get_sentiment_score_roberta_model(text):
#    encoded_text=tokenizer(text,return_tensors="pt")
#    output=model(**encoded_text)
#    scores=output[0][0].detach().numpy()
#    scores=softmax(scores)
#    return {
#       "roberta_negative": scores[0],
#       "roberta_neutral": scores[1],
#       "roberta_positive": scores[2],
#    }
# ================RoBERTa model ends here=====================
# for pushing my code on github, i am commenting out this RoBERTa model because it's size is large
app=Flask(__name__)
@app.route("/",methods=["GET", "POST"])
def main():
  try:
    sentiment = "Please submit a comment for analysis"
    colour = "black"
    negative_percentage = ""
    neutral_percentage = ""
    positive_percentage = ""
    overall_percentage = 0

    if request.method == "POST":
      comment=request.form.get("comment")
    #   robertaScore=get_sentiment_score_roberta_model(comment)
    #   print(robertaScore)
      print(comment)
      tokens=nltk.word_tokenize(comment)
      print(tokens)
      tagged=nltk.pos_tag(tokens)
      print(tagged)
      
      analyzer=SentimentIntensityAnalyzer()
      #  SentimentIntensityAnalyzer is used to get neg/neu/pos score. This used to remove "stop words(the words that dont have positive or negative feeling such as and, the etc)" and then each words are scored and combined to a total score
      
    #   print("Analyzer",analyzer)
      score=analyzer.polarity_scores(comment)
      print(score)
      #  {'neg': 0.279, 'neu': 0.604, 'pos': 0.118, 'compound': -0.395}
      #  here compound value is aggregation  of negative, positive and neutral. it is the value of negative one to positive one, and represent how negative to positive it is.
      negative_percentage = "{:.2f}%".format(score['neg'] * 100)
      neutral_percentage = "{:.2f}%".format(score['neu'] * 100)
      positive_percentage = "{:.2f}%".format(score['pos'] * 100)
      overall_percentage = "{:.2f}".format(abs(score['compound'])*100)
      if score['compound']<0:
        sentiment="Negative"
        colour="red"
      elif score['compound']>0:
        sentiment="Positive"
        colour="green"
      else:
          sentiment="Neutral"
          colour="blue"
    return render_template("index.html", negative_percentage=negative_percentage, neutral_percentage=neutral_percentage, positive_percentage=positive_percentage, overall_percentage=overall_percentage, sentiment=sentiment, colour=colour)   
  except Exception as e:
     print(f"Error processing sentiment: {e}")   
     return "An error occurred while processing the sentiment analysis.", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
from flask import Flask, render_template, request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
import nltk
nltk.download("vader_lexicon")
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)

def extract_sentiments(text):
    print(text)
    entities = []
    attributes = []

    # Tokenize the text into sentences
    sentences = sent_tokenize(text)
    print(sentences)
    
    # Initialize SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    for sentence in sentences:
        # Tokenize the sentence into words and get Part-of-Speech tags
        words = word_tokenize(sentence)
        print(words)
        tagged_words = pos_tag(words)
        print(tagged_words)

        # Initialize lists to store entities and attributes in the current sentence
        current_entities = []
        current_attributes = []

        # Iterate through the tagged words to extract entities and attributes
        for word, tag in tagged_words:
            if tag.startswith('NN') or tag == 'PRP':  # Nouns or pronouns
                sentiment_score = analyzer.polarity_scores(word)
                current_entities.append((word, sentiment_score))
            elif tag.startswith('JJ'):  # Adjectives
                sentiment_score = analyzer.polarity_scores(word)
                current_attributes.append((word, sentiment_score))

        # Add entities and attributes of the current sentence to the overall lists
        entities.extend(current_entities)
        attributes.extend(current_attributes)

    return entities, attributes

@app.route("/", methods=["GET", "POST"])
def main():
    try:
        sentiment = "Please submit a comment for analysis"
        colour = "black"
        negative_percentage = ""
        neutral_percentage = ""
        positive_percentage = ""
        overall_percentage = 0
        entities = []
        attributes = []

        if request.method == "POST":
            comment = request.form.get("comment")
            
            # Perform sentiment analysis
            entities, attributes = extract_sentiments(comment)

            # Calculate overall sentiment
            analyzer = SentimentIntensityAnalyzer()
            score = analyzer.polarity_scores(comment)
            negative_percentage = "{:.2f}%".format(score['neg'] * 100)
            neutral_percentage = "{:.2f}%".format(score['neu'] * 100)
            positive_percentage = "{:.2f}%".format(score['pos'] * 100)
            overall_percentage = "{:.2f}".format(abs(score['compound']) * 100)

            if score['compound'] < 0:
                sentiment = "Negative"
                colour = "red"
            elif score['compound'] > 0:
                sentiment = "Positive"
                colour = "green"
            else:
                sentiment = "Neutral"
                colour = "blue"

        return render_template("index.html", entities=entities, attributes=attributes,
                               negative_percentage=negative_percentage, neutral_percentage=neutral_percentage,
                               positive_percentage=positive_percentage, overall_percentage=overall_percentage,
                               sentiment=sentiment, colour=colour)

    except Exception as e:
        return "An error occurred while processing the sentiment analysis.", 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

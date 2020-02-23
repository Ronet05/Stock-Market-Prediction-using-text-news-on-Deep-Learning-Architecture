from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import math


def sent_avg_score(text):
    tb_object = TextBlob(text)
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_score = vader_analyzer.polarity_scores(text)['compound']
    textblob_score = tb_object.sentiment.polarity
    return (vader_score + textblob_score) / 2


def sent_score(text):
    vader_analyzer = SentimentIntensityAnalyzer()
    vader_score = vader_analyzer.polarity_scores(text)['compound']
    return vader_score


def sent_magnitude(text, alpha):
    vader_analyzer = SentimentIntensityAnalyzer()
    c = vader_analyzer.polarity_scores(text)['compound']
    mag = math.sqrt((math.pow(c, 2) * alpha) / (1 - math.pow(c, 2)))
    return mag

# from flask import Flask, render_template

# app = Flask(__name__)

# @app.route('/')
# def index():
#     return render_template('index.html')


# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request, render_template
import matplotlib
matplotlib.use('Agg')  # Use the non-interactive backend
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from googleapiclient.discovery import build
import re
import emoji
import os
import tensorflow as tf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from dash import Dash, html, dcc
from dash.dependencies import Input, Output

app = Flask(__name__)

API_KEY = os.getenv('YOUTUBE_API_KEY', 'default_fallback_key')
youtube = build('youtube', 'v3', developerKey=API_KEY)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

positive_comments = []
negative_comments = []
neutral_comments = []

@app.context_processor
def utility_processor():
    return dict(zip=zip)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Create a pipeline for sentiment analysis
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

def fetch_video_details(video_id):
    request = youtube.videos().list(
        part='snippet',
        id=video_id
    )
    response = request.execute()

    if response['items']:
        video_details = response['items'][0]['snippet']
        title = video_details['title']
        channel_title = video_details['channelTitle']

        publish_date_iso = video_details['publishedAt']
        publish_date = format_publish_date(publish_date_iso)
        
        thumbnails = video_details['thumbnails']
        thumbnail_url = thumbnails['high']['url'] if 'high' in thumbnails else thumbnails['default']['url']
        return title, channel_title, publish_date, thumbnail_url
    else:
        return None, None, None, None
    
def format_publish_date(iso_date):
    # Convert ISO 8601 date string to datetime object
    publish_datetime = datetime.strptime(iso_date, "%Y-%m-%dT%H:%M:%SZ")
    # Format datetime object to a more readable format
    formatted_date = publish_datetime.strftime("%B %d, %Y %H:%M:%S")
    return formatted_date

def fetch_comments(video_id, uploader_channel_id, page_token=None):
    request = youtube.commentThreads().list(
        part='snippet', videoId=video_id, maxResults=100, pageToken=page_token)
    response = request.execute()
    comments, likes, timestamps = [], [], []
    for item in response['items']:
        comment = item['snippet']['topLevelComment']['snippet']
        if comment['authorChannelId']['value'] != uploader_channel_id:
            comment_text = comment['textDisplay'].replace('&#39;', "'").replace('&quot;', '"').replace('<br>', ' ')
            comments.append(comment_text)
            likes.append(comment['likeCount'])
            timestamps.append(comment['publishedAt'])
    return comments, likes, timestamps, response.get('nextPageToken')

def analyze_video(video_url, keywords):
    global positive_comments, negative_comments, neutral_comments

    video_id = video_url[-11:]
    video_response = youtube.videos().list(part='snippet', id=video_id).execute()
    video_snippet = video_response['items'][0]['snippet']
    uploader_channel_id = video_snippet['channelId']

    # Fetch video details
    title, channel_title, publish_date, thumbnail_url = fetch_video_details(video_id)

    comments, likes, timestamps = [], [], []
    nextPageToken = None

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        while len(comments) < 600:
            futures.append(executor.submit(fetch_comments, video_id, uploader_channel_id, nextPageToken))
            if nextPageToken is None:
                break

        for future in futures:
            result_comments, result_likes, result_timestamps, nextPageToken = future.result()
            comments.extend(result_comments)
            likes.extend(result_likes)
            timestamps.extend(result_timestamps)

    hyperlink_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    threshold_ratio = 0.65
    relevant_comments, relevant_likes, relevant_timestamps = [], [], []
    for comment_text, like_count, timestamp in zip(comments, likes, timestamps):
        comment_text = comment_text.lower().strip()
        emojis = emoji.emoji_count(comment_text)
        text_characters = len(re.sub(r'\s', '', comment_text))
        if (any(char.isalnum() for char in comment_text)) and not hyperlink_pattern.search(comment_text):
            if emojis == 0 or (text_characters / (text_characters + emojis)) > threshold_ratio:
                relevant_comments.append(comment_text)
                relevant_likes.append(like_count)
                relevant_timestamps.append(datetime.strptime(timestamp, '%Y-%m-%dT%H:%M:%SZ'))

    f = open("ytcomments.txt", 'w', encoding='utf-8')
    for idx, comment in enumerate(relevant_comments):
        f.write(str(comment) + "\n")
    f.close()

    polarity = []
    positive_comments, negative_comments, neutral_comments = [], [], []

    def sentiment_scores(comment, polarity):
        inputs = tokenizer(comment, truncation=True, max_length=512, return_tensors="pt")
        sentiment_result = model(**inputs)
        sentiment_score = sentiment_result.logits.softmax(dim=-1).max().item()
        sentiment_label = sentiment_result.logits.argmax().item()
        if sentiment_label == 0:  # Adjust according to model's output labels
            polarity.append(-sentiment_score)
        elif sentiment_label == 4:  # Adjust according to model's output labels
            polarity.append(sentiment_score)
        else:
            polarity.append(0)
        return polarity

    f = open("ytcomments.txt", 'r', encoding='utf-8')
    comments = f.readlines()
    f.close()
    for index, items in enumerate(comments):
        polarity = sentiment_scores(items, polarity)
        if polarity[-1] > 0.05:
            positive_comments.append(items)
        elif polarity[-1] < -0.05:
            negative_comments.append(items)
        else:
            neutral_comments.append(items)

    avg_polarity = sum(polarity) / len(polarity)
    sentiment = "Neutral"
    if avg_polarity > 0.05:
        sentiment = "Positive"
    elif avg_polarity < -0.05:
        sentiment = "Negative"

    positive_count = len(positive_comments)
    negative_count = len(negative_comments)
    neutral_count = len(neutral_comments)

    comment_lengths = [len(comment) for comment in comments]
    avg_comment_length = sum(comment_lengths) / len(comment_lengths)

    # Time Series Analysis
    time_intervals = pd.date_range(start=min(relevant_timestamps), end=max(relevant_timestamps), freq='D')
    daily_sentiment = pd.Series(0, index=time_intervals)
    daily_count = pd.Series(0, index=time_intervals)
    
    for timestamp, score in zip(relevant_timestamps, polarity):
        day = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        if day in daily_sentiment:
            daily_sentiment[day] += score
            daily_count[day] += 1

    daily_sentiment = daily_sentiment / daily_count
    daily_sentiment = daily_sentiment.fillna(0)

    time_fig = px.line(x=daily_sentiment.index, y=daily_sentiment.values, labels={'x': 'Date', 'y': 'Average Sentiment'})
    time_fig.update_layout(title_text='Time Series Sentiment Analysis', title_x=0.5)
    time_fig.write_html('static/time_series_sentiment.html')

    # Compute daily sentiment scores
    df = pd.DataFrame({
        'Timestamp': relevant_timestamps,
        'Sentiment': polarity
    })
    df['Date'] = df['Timestamp'].dt.date
    daily_sentiment = df.groupby('Date')['Sentiment'].mean()

    # Plot time series of sentiment scores
    plt.figure(figsize=(12, 6))
    plt.plot(daily_sentiment.index, daily_sentiment.values, marker='o', linestyle='-')
    plt.xlabel('Date')
    plt.ylabel('Average Sentiment Score')
    plt.title('Daily Average Sentiment Score Over Time')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('static/time_series_sentiment.png')
    plt.close()

    labels = ['Positive', 'Negative', 'Neutral']
    comment_counts = [positive_count, negative_count, neutral_count]
    bar_fig = px.bar(x=labels, y=comment_counts, labels={'x': 'Sentiment', 'y': 'Comment Count'})
    bar_fig.update_layout(title_text='Sentiment Analysis of Comments', title_x=0.5)
    bar_fig.write_html('static/bar_chart_interactive.html')

    pie_fig = px.pie(values=comment_counts, names=labels, title='Sentiment Distribution')
    pie_fig.update_traces(textposition='inside', textinfo='percent+label')
    pie_fig.write_html('static/pie_chart_interactive.html')

    hist_fig = px.histogram(comment_lengths, nbins=30, labels={'value': 'Comment Length', 'count': 'Number of Comments'})
    hist_fig.update_layout(title_text='Distribution of Comment Lengths', title_x=0.5)
    hist_fig.write_html('static/comment_length_histogram.html')

    def generate_word_cloud(text, filename):
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    generate_word_cloud(' '.join(positive_comments), 'static/word_cloud_positive.png')
    generate_word_cloud(' '.join(negative_comments), 'static/word_cloud_negative.png')
    generate_word_cloud(' '.join(neutral_comments), 'static/word_cloud_neutral.png')

    positive_comments = [comment for comment, score in zip(relevant_comments, polarity) if score > 0.05]
    negative_comments = [comment for comment, score in zip(relevant_comments, polarity) if score < -0.05]
    neutral_comments = [comment for comment, score in zip(relevant_comments, polarity) if -0.05 <= score <= 0.05]

    most_positive_comment = positive_comments[0:3]
    most_negative_comment = negative_comments[0:3]

    most_liked_comments = sorted(zip(relevant_comments, relevant_likes), key=lambda x: x[1], reverse=True)[:3]

    keyword_results = {}
    for keyword in keywords:
        keyword = keyword.strip().lower()
        keyword_comments = [comment for comment in relevant_comments if keyword in comment]
        keyword_polarity = [polarity[index] for index, comment in enumerate(relevant_comments) if keyword in comment]
        keyword_sentiment = "Neutral"
        if keyword_polarity:
            avg_keyword_polarity = sum(keyword_polarity) / len(keyword_polarity)
            if avg_keyword_polarity > 0.05:
                keyword_sentiment = "Positive"
            elif avg_keyword_polarity < -0.05:
                keyword_sentiment = "Negative"
        keyword_results[keyword] = {
            "count": len(keyword_comments),
            "avg_polarity": avg_keyword_polarity if keyword_polarity else 0,
            "sentiment": keyword_sentiment,
            "comments": keyword_comments[:3]  # Add top 3 comments for each keyword
        }

    results = {
        "avg_polarity": avg_polarity,
        "sentiment": sentiment,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "neutral_count": neutral_count,
        "most_positive_comment": positive_comments[:3],
        "most_negative_comment": negative_comments[:3],
        "avg_comment_length": avg_comment_length,
        "most_liked_comments": most_liked_comments,
        "keyword_results": keyword_results,
        "daily_sentiment": daily_sentiment.to_dict(),
        "title": title,
        "channel_title": channel_title,
        "publish_date": publish_date,
        "thumbnail_url": thumbnail_url
    }
    return results

# Create a Dash app within your Flask app
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Define the layout of the Dash app
dash_app.layout = html.Div([
    dcc.Dropdown(
        id='sentiment-dropdown',
        options=[
            {'label': 'Positive', 'value': 'positive'},
            {'label': 'Negative', 'value': 'negative'},
            {'label': 'Neutral', 'value': 'neutral'},
        ],
        value='positive'
    ),
    dcc.Graph(id='wordcloud' , style={'width': '100%', 'height': '600px'})
])

@dash_app.callback(
    Output('wordcloud', 'figure'),
    [Input('sentiment-dropdown', 'value')]
)
def update_wordcloud(sentiment):
    global positive_comments, negative_comments, neutral_comments

    if sentiment == 'positive':
        text = ' '.join(positive_comments)
    elif sentiment == 'negative':
        text = ' '.join(negative_comments)
    else:
        text = ' '.join(neutral_comments)

    # Generate word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)

    # Plotly figure for the word cloud
    fig = px.imshow(wordcloud, binary_string=True)
    fig.update_layout(
        title=f'Word Cloud for {sentiment.capitalize()} Comments',
        plot_bgcolor='white',
    )
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)

    return fig

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    video_url = request.form['video_url']
    keywords = request.form.get('keywords', '').split(',')
    results = analyze_video(video_url, keywords)
    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run()
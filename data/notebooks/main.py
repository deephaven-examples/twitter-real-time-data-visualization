os.system("pip install requests")
os.system("pip install nltk")
os.system("pip install sklearn")
os.system("pip install wordcloud")

import json
import matplotlib.pyplot as plt
import numpy as np
import random
import re
import requests
import threading
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS

import deephaven.dtypes as dht
from deephaven import SortDirection
from deephaven import DynamicTableWriter
from deephaven import pandas as dhpd
from deephaven.plugin.matplotlib import TableAnimation

# you will need to have a set of keys and tokens to authenticate your request
# https://developer.twitter.com/en/docs/authentication
BEARER_TOKEN = <INSERT YOUR TOKEN HERE>
TWITTER_ENDPOINT_URL = "https://api.twitter.com/2/tweets/search/stream"

TOP_N = 20  # TOP N most frequent words in tweets
TERM_1 = 'good'
TERM_2 = 'news'

twitter_table_col_definitions = {"tweet": dht.string, "clean_tweet": dht.string, "length": dht.int32, "category": dht.string, "point_x": dht.double, "point_y": dht.double}
twitter_table_writer = DynamicTableWriter(twitter_table_col_definitions)
tweet_table = twitter_table_writer.table

count_table_col_definitions = {"word": dht.string, "count": dht.int32}
count_table_writer = DynamicTableWriter(count_table_col_definitions)
count_table = count_table_writer.table


def bearer_oauth(r):
    """
    Method required by bearer token authentication.
    """
    r.headers["Authorization"] = f"Bearer {BEARER_TOKEN}"
    r.headers["User-Agent"] = "v2FilteredStreamPython"
    return r


def get_rules():
    """
    Method to get a list of rules that have been added to the stream
    """
    response = requests.get(f"{TWITTER_ENDPOINT_URL}/rules", auth=bearer_oauth)
    if response.status_code != 200:
        raise Exception(
            "Cannot get rules (HTTP {}): {}".format(response.status_code, response.text)
        )
    return response.json()


def delete_all_rules(rules):
    """
    Method to  remove the list of all the rules from the stream
    """
    if rules is None or "data" not in rules:
        return None

    ids = list(map(lambda rule: rule["id"], rules["data"]))
    payload = {"delete": {"ids": ids}}
    response = requests.post(f"{TWITTER_ENDPOINT_URL}/rules", auth=bearer_oauth, json=payload)
    if response.status_code != 200:
        raise Exception(
            "Cannot delete rules (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )

def set_rules():
    """
    Method to add rules to the stream
    """
    demo_rules = [{"value": "news has:media", "tag": "news"}]
    payload = {"add": demo_rules}
    response = requests.post(f"{TWITTER_ENDPOINT_URL}/rules", auth=bearer_oauth, json=payload)
    if response.status_code != 201:
        raise Exception(
            "Cannot add rules (HTTP {}): {}".format(response.status_code, response.text)
        )

def get_tweets():
    """
    Method to add rules to the stream
    """
    response = requests.get(f"{TWITTER_ENDPOINT_URL}?tweet.fields=lang", auth=bearer_oauth, stream=True)

    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    return response


# Check if there are any active rules, delete them from the stream and add our demo rules to the stream
rules = get_rules()
delete_all_rules(rules)
set_rules()


def preprocess(tweet):
    """
    Method to preprocess tweets (remove @ user names, non-alphabetic characters, stem words)
    """
    ps = PorterStemmer()
    txt = ' '.join(word for word in tweet.split() if not word.startswith('@'))
    txt = re.sub('[^a-zA-Z]', ' ', txt)
    txt = txt.lower().split()
    txt = [ps.stem(word) for word in txt if not word in stopwords.words('english') and len(word) > 3]
    txt = ' '.join(txt)
    return txt


RADIUS_1 = 10
SHIFT_1 = -6
CENTER_1 = (SHIFT_1, 0)
X_1 = [(-RADIUS_1 + SHIFT_1) + i * 0.001 for i in range(2 * 1000 * RADIUS_1 + 1)]
Y1_UPPER = [np.sqrt(RADIUS_1 ** 2 - (i - SHIFT_1) ** 2) for i in X_1]
Y1_LOWER = [-y for y in Y1_UPPER]

RADIUS_2 = 10
SHIFT_2 = 6
CENTER_2 = (SHIFT_2, 0)
X_2 = [(-RADIUS_2 + SHIFT_2) + i * 0.001 for i in range(2 * 1000 * RADIUS_2 + 1)]
Y2_UPPER = [np.sqrt(RADIUS_2 ** 2 - (i - SHIFT_2) ** 2) for i in X_2]
Y2_LOWER = [-y for y in Y2_UPPER]


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def in_circle(point, center, radius):
    if distance(point, center) < radius - 0.7:
        return True
    return False

def out_circle(point, center, radius):
    if distance(point, center) > radius + 0.7:
        return True
    return False

def collide(rand, points):
    r = 0.4
    for i in points:
        if distance([rand[0], rand[1]], [i[0], i[1]]) < 2 * r:
            return True
    return False

def write_live_data():
    """
    The function to write twitter data to a table
    """
    response = get_tweets()
    diag_points = []
    for response_line in response.iter_lines():
        
        if response_line:
            json_response = json.loads(response_line)
            lang = json_response["data"]["lang"]

            # we are interested only in english tweets
            if lang == "en":
                tweet = json_response["data"]["text"]
                clean_tweet = preprocess(tweet)

                words = clean_tweet.split()
                for word in words:
                    count_table_writer.write_row([word, 1])
                length = len(words)

                # check if our special terms are used within the messages
                count = 0
                category = ''
                point_x = None
                point_y = None
                if TERM_1 in words:
                    category = 'left'
                    count += 1

                if TERM_2 in words:
                    category = 'right'
                    count += 1

                if count == 2:
                    category = 'middle'

                if category == 'middle':
                    
                    rand = [random.uniform(CENTER_1[0], CENTER_2[0]),
                            random.uniform(CENTER_1[1] - RADIUS_1, CENTER_2[1] + RADIUS_2)]
                    
                    while (not (in_circle(rand, CENTER_1, RADIUS_1) and in_circle(rand, CENTER_2, RADIUS_2))) or collide(rand, diag_points):
                        rand = [random.uniform(CENTER_1[0], CENTER_2[0]), random.uniform(CENTER_1[1] - RADIUS_1, CENTER_2[1] + RADIUS_2)]
                        
                    point_x = rand[0]
                    point_y = rand[1]
                    diag_points.append(rand)
                if category == 'left':
                    rand = [random.uniform(CENTER_1[0] - RADIUS_1, CENTER_1[0] + RADIUS_1),
                            random.uniform(CENTER_1[1] - RADIUS_1, CENTER_1[1] + RADIUS_1)]
                    while (not (in_circle(rand, CENTER_1, RADIUS_1) and out_circle(rand, CENTER_2, RADIUS_2))) or collide(rand, diag_points):
                        rand = [random.uniform(CENTER_1[0] - RADIUS_1, CENTER_1[0] + RADIUS_1),
                                random.uniform(CENTER_1[1] - RADIUS_1, CENTER_1[1] + RADIUS_1)]
                    diag_points.append(rand)
                    point_x = rand[0]
                    point_y = rand[1]
                if category == 'right':
                    rand = [random.uniform(CENTER_2[0] - RADIUS_2, CENTER_2[0] + RADIUS_2),
                            random.uniform(CENTER_2[1] - RADIUS_2, CENTER_2[1] + RADIUS_2)]
                    while (not (in_circle(rand, CENTER_2, RADIUS_2) and out_circle(rand, CENTER_1, RADIUS_1))) or collide(rand, diag_points):
                        rand = [random.uniform(CENTER_2[0] - RADIUS_2, CENTER_2[0] + RADIUS_2),
                                random.uniform(CENTER_2[1] - RADIUS_2, CENTER_2[1] + RADIUS_2)]
                    diag_points.append(rand)
                    point_x = rand[0]
                    point_y = rand[1]

                twitter_table_writer.write_row([tweet, clean_tweet, length, category, point_x, point_y])



# Run the thread that writes to the table
thread = threading.Thread(target=write_live_data)
thread.start()

# Find TOP N most popular words in tweets
count = count_table.count_by("Number", by=["word"])
count_sorted = count.sort(order_by=["Number"], order=[SortDirection.DESCENDING])
count_sorted_top = count_sorted.head(TOP_N)

# Draw real-time bar chart with the frequency of the top N words
bar_fig, bar_fig_ax = plt.subplots()
plt.xticks(rotation=90)
rects = bar_fig_ax.bar(range(TOP_N), [0] * TOP_N)
def animate_bar_plot_fig(data, update):
    for rect, h in zip(rects, data["Number"]):
        rect.set_height(h)
    bar_fig_ax.set_xticklabels(data["word"])
    bar_fig_ax.relim()
    bar_fig_ax.autoscale_view(True, True, True)
bar_plot_ani = TableAnimation(bar_fig, count_sorted_top, animate_bar_plot_fig)

# Draw real-time word cloud for top N words
wordcloud_fig = plt.figure()
wordcloud_ax = wordcloud_fig.subplots()
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
def animate_wordcloud_fig(data, update):
    data_frame = dhpd.to_pandas(count_sorted_top)
    word_str = " ".join(data_frame["word"].tolist())
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          min_font_size=10, max_words=50).generate(word_str)
    wordcloud_ax.imshow(wordcloud)

wordclod_ani = TableAnimation(wordcloud_fig, count_sorted_top, animate_wordcloud_fig)


# Draw real-time box plot to show the distribution of the number of words in tweets
box_plot_fig = plt.figure()
box_plot_ax = box_plot_fig.subplots()
plt.gca().axes.get_xaxis().set_visible(False)
box_plot = box_plot_ax.boxplot([], patch_artist=True, labels=['Distribution of Tweet Word Counts'])

def animate_box_plot_fig(data, update):
    box_plot_ax.cla()
    box_plot_ax.boxplot(x=data['length'])
    box_plot_ax.relim()
    box_plot_ax.autoscale_view(True, True, True)

box_plot_ani = TableAnimation(box_plot_fig, tweet_table, animate_box_plot_fig)


## Draw Real-Time Twitter Venn diagram (https://github.com/anbarief/Blog)
venn_plot_fig = plt.figure()
venn_ax = venn_plot_fig.subplots()
venn_ax.plot(X_1, Y1_UPPER, color='k')
venn_ax.plot(X_1, Y1_LOWER, color='k')
venn_ax.plot(X_2, Y2_UPPER, color='k')
venn_ax.plot(X_2, Y2_LOWER, color='k')
venn_ax.text(SHIFT_1 - RADIUS_1 - 2, 0, TERM_1.upper(), ha='center', color='white')
venn_ax.text(SHIFT_2 + RADIUS_2 + 2, 0, TERM_2.upper(), ha='center', color='white')
plt.gca().axes.get_xaxis().set_visible(False)
plt.gca().axes.get_yaxis().set_visible(False)
plt.axis('equal')
venn_ax.set_title('Twitter Venn Diagram')

def animate_venn_fig(data, update):
    venn_ax.plot(data["point_x"], data["point_y"], marker='o', ms=10, color='green', lw=0)

tweet_table_with_category = tweet_table.where(filters=["category in `middle`, `left`, `right`"])
venn_ani = TableAnimation(venn_plot_fig, tweet_table_with_category, animate_venn_fig)

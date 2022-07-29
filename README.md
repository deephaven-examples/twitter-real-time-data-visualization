# twitter-real-time-data-visualization
This example pulls public tweets in real-time and builds various animated charts. 

The repository integrates [Deephaven](https://deephaven.io/) with [matplotlib](https://matplotlib.org/), a plotting library for the [Python](https://www.python.org/) programming language.

To get started, run:

```
git clone https://github.com/anastasia-si/twitter-real-time-data-visualization.git
cd twitter-real-time-data-visualization
docker-compose pull
docker-compose up  --build -d
```

Navigate to [http://localhost:10000/ide](http://localhost:10000/ide) and open the `main.py` notebook to see examples of real-time data visualizations.
Please note you might need to download corpus.words and stopwords from `nltk` library:
```
nltk.download('words')
nltk.download('stopwords')
```

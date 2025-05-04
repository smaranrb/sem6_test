import streamlit as st
import yaml
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from src.news.collector import NewsCollector
from src.news.summarizer import NewsSummarizer
from src.market.data_loader import MarketDataLoader
from src.models.sentiment import SentimentAnalyzer
from src.models.prediction import PricePredictor
from src.utils.visualization import create_candlestick_chart

# Load configuration
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set page config
st.set_page_config(page_title="Finance Agent", page_icon="📈", layout="wide")


# Initialize components
@st.cache_resource
def init_components():
    news_collector = NewsCollector(config)
    news_summarizer = NewsSummarizer(config)
    market_loader = MarketDataLoader(config)
    sentiment_analyzer = SentimentAnalyzer(config)
    price_predictor = PricePredictor(config)
    return (
        news_collector,
        news_summarizer,
        market_loader,
        sentiment_analyzer,
        price_predictor,
    )


# Sidebar
st.sidebar.title("Finance Agent")
symbol = st.sidebar.selectbox("Select Stock", config["market"]["symbols"])

# Main content
st.title(f"📈 {symbol} Analysis")

# Market Data Section
st.header("Market Data")
market_loader = MarketDataLoader(config)
data = market_loader.get_stock_data(symbol)

if data is not None:
    # Create candlestick chart
    fig = create_candlestick_chart(data, symbol)
    st.plotly_chart(fig, use_container_width=True)

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current Price", f"${data['Close'].iloc[-1]:.2f}")
    with col2:
        daily_change = (
            (data["Close"].iloc[-1] - data["Close"].iloc[-2]) / data["Close"].iloc[-2]
        ) * 100
        st.metric("Daily Change", f"{daily_change:.2f}%")
    with col3:
        st.metric("Volume", f"{data['Volume'].iloc[-1]:,}")
    with col4:
        st.metric("52-Week High", f"${data['High'].max():.2f}")

# News Section
st.header("Latest News")
news_collector = NewsCollector(config)
news_summarizer = NewsSummarizer(config)

news = news_collector.get_news(symbol)
if news:
    for article in news[:5]:  # Show top 5 news articles
        with st.expander(article["title"]):
            st.write(f"Source: {article['source']}")
            st.write(f"Published: {article['publishedAt']}")
            summary = news_summarizer.summarize(article["content"])
            st.write(summary)
            st.write(f"[Read more]({article['url']})")

# Sentiment Analysis
st.header("Market Sentiment")
sentiment_analyzer = SentimentAnalyzer(config)
sentiment = sentiment_analyzer.analyze_news(news)

if sentiment:
    st.write(f"Overall Sentiment: {sentiment['overall_sentiment']}")
    st.write(f"Confidence: {sentiment['confidence']:.2%}")

# Price Prediction
st.header("Price Prediction")
price_predictor = PricePredictor(config)
prediction = price_predictor.predict(symbol)

if prediction:
    st.write(f"Predicted Price: ${prediction['predicted_price']:.2f}")
    st.write(f"Confidence: {prediction['confidence']:.2%}")
    st.write(f"Time Horizon: {prediction['horizon']} days")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and LLMs")

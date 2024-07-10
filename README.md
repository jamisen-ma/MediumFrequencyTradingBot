# Energy Sector Sentiment Analysis Trading Bot

## Overview

This is a medium-frequency trading bot for stocks in the energy sector. It uses sentiment analysis from Twitter and Reddit to inform trading decisions. The bot collects real-time data, analyzes sentiment, and makes trade decisions based on the sentiment score. It also includes a machine learning model to backtest and refine trading strategies using historical financial data.

## Features

- **Sentiment Analysis**: Uses VADER and transformer-based models to analyze the sentiment of collected data.
- **Asynchronous Processing**: Utilizes `asyncio` and `aiohttp` for asynchronous HTTP requests to reduce waiting times and improve execution speed.
- **Parallel Processing**: Executes multiple trade orders in parallel using `asyncio.gather()` to enhance performance.
- **Medium/High-Frequency Capabilities**: Reduced the interval between data collection and trade execution to 1 second for higher frequency trading.

## Technology Stack

### Data Collection and Sentiment Analysis

- **Reddit API**: For collecting posts and comments related to energy sector companies.
- **Twitter API**: For collecting tweets related to energy sector companies.
- **VADER Sentiment Analysis**: A lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media.
- **HuggingFace Transformers**: For transformer-based sentiment analysis using pre-trained models.


### How to run the trading bot?
```sh
vcpkg install curl
vcpkg install nlohmann-json
python3 setup_database.py
python3 sentiment_analysis.py
python3 trading_bot.py

OR for C++ bot:

g++ trading_bot.cpp -o trading_bot -lcurl -ljsoncpp
./trading_bot
'''

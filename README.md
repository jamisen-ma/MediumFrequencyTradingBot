# Energy Sector Sentiment Analysis Trading Bot

## Overview

This is a medium-frequency trading bot for stocks in the energy sector. It uses sentiment analysis from Twitter and Reddit to inform trading decisions. The bot collects real-time data, analyzes sentiment using VADER, and executes trades using Alpaca based on the sentiment score. As the bot runs, it feeds data into a LSTM model to backtest and refine my trading algorthms.

## Features

- **Sentiment Analysis**: Uses VADER and transformer-based models to analyze the sentiment of collected data.
- **Asynchronous Processing**: Utilizes `asyncio` and `aiohttp` for asynchronous HTTP requests to reduce waiting times and improve execution speed.
- **Parallel Processing**: Executes multiple trade orders in parallel using `asyncio.gather()` to enhance performance.
- **Medium/High-Frequency Capabilities**: Reduced the interval between data collection and trade execution to 1 second for higher frequency trading.


## Tools and Technologies Used

- Reddit API
- Twitter API
- VADER Sentiment Analysis
- HuggingFace Transformers
- NLTK
- Alpaca API
- scikit-learn
- SQLite
- Yahoo Finance API
- aiohttp
- asyncio
- requests
- pandas
- nlohmann/json
  
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


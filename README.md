# Deal or No Deal Analysis
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/jstock29/dealnodeal/main/app.py)

This is a repository devoted to the beautiful, perfect game of Deal or No Deal. I've decided to collect game data from this show with the goal of analyzing it to figure out how the show works. This repo contains my code to gather data, build machine learning models to approximate the Banker (create RoboBankers), and play along with a RoboBanker to see how offers change over the course of a game. 

## Install
Create and activate a virtual environment:

`virtualenv venv`

`source venv/bin/activate`

Install Python requirements:

`pip install -r requirements.txt`

## Play Along
I tested a bunch of potential regression models and picked three of the better performing ones to be available for you to see how my robots work. 

You can play along or simulate your own games and offers with the Streamlit app deployed below. Disclaimer: I store anonymous data in my database for me to potentially do something with later. 

[https://share.streamlit.io/jstock29/dealnodeal/main/app.py](https://share.streamlit.io/jstock29/dealnodeal/main/app.py)

## Data Collection
I'll collect data using Streamlit. You can run the app (assuming you have a Postgres database and update the credentials) with:

`streamlit run main.py`

All data collected in the way assumes a local database so that I don't have people messing up my actual data. You can [see my actual data on Kaggle](https://www.kaggle.com/jaredstock/deal-or-no-deal-game-data).

## Data Analysis and Model Creation
Code to analyze the data in the database exists primarily in `main.py` and `visualization.py`. I use `model.py` to create and evaluate various regression models in my quest to create a RoboBanker.

I analyzed the data in a variety of ways (which you can [read more about here](https://towardsdatascience.com/i-figured-out-how-deal-or-no-deal-works-kind-of-875e63a8cef6)) but the highlight is that I determined that Banker doesn't use a strict algorithm, but rather works within a set of guidelines that define how offers can be made each round. I also looked at when contestants take offers, how offers change over the course of a game, and how much people win compared to their best offers.

To create my RoboBankers, I tested a variety of regression models from sckit-learn and elsewhere, but eventually landed on three top performers: A Decision Forest model, a LightGBM model, and an XGBoost model. 
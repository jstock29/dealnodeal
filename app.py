import datetime

import numpy as np
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import visualization
import joblib
import random
import SessionState

VALUES = [
    0,
    1,
    5,
    10,
    25,
    50,
    75,
    100,
    200,
    300,
    400,
    500,
    750,
    1_000,
    5_000,
    10_000,
    25_000,
    50_000,
    75_000,
    100_000,
    200_000,
    # 250_000,
    300_000,
    400_000,
    500_000,
    750_000,
    1_000_000,
    # 2_000_000,
    # 3_000_000,
    # 6_000_000,
]

BIG_VALUES = [val for val in VALUES if val >= 100_000]

L_SUM = sum([val for val in VALUES[:len(VALUES) // 2]])

R_SUM = sum([val for val in VALUES[len(VALUES) // 2:]])

engine = create_engine('postgresql://{user}:{pw}@{host}:{port}/{dbname}'.
                       format(user=st.secrets['username'], pw=st.secrets['password'], host=st.secrets['host'],
                              port=5432,
                              dbname=st.secrets['dbname']))

session = SessionState.get(run_id=0)


def generate(random_chars=12, alphabet="0123456789abcdef"):
    r = random.SystemRandom()
    return ''.join([r.choice(alphabet) for i in range(random_chars)])


query_params = st.experimental_get_query_params()
if not hasattr(st, 'game_id') or not query_params:
    st.game_id = generate(8)
    session.game_id = st.game_id
    st.experimental_set_query_params(round_number=1, game_id=st.game_id, prev_offer=0)


def main():
    st.set_page_config(
        page_title="DEAL OR NO DEAL",
        page_icon="🤑",
        initial_sidebar_state="expanded")
    st.sidebar.title('DEAL OR NO DEAL')
    selected_banker = st.sidebar.selectbox('Pick your Banker', ['Random Forest', 'LightGBM', 'XGBoost'], 0)
    if st.sidebar.button('New Game'):
        new_game()
    st.sidebar.markdown("""
    This is a simulation of the Deal or No Deal Banker's offers. The code for this project can be found at [my Github](https://github.com/jstock29/dealnodeal) and the data that I painstaking collected from over 100 episodes of the show is on [my Kaggle](https://www.kaggle.com/jaredstock/deal-or-no-deal-game-data).
    
    You can see what the RoboBanker will offer by simulating a board at various rounds. Each round you should pick the correct number of values from the board:
    
    1. Pick 6 - 6 Total
    2. Pick 5 - 11 Total
    3. Pick 4 - 15 Total
    4. Pick 3 - 18 Total
    5. Pick 2 - 20 Total
    6. Pick 1 - 21 Total
    7. Pick 1 - 22 Total
    8. Pick 1 - 23 Total
    9. Pick 1 - 24 Total
    10. Pick 1 -25 Total
    
    After each round you can see what my RoboBanker is offering you and decide if that's a deal you want to take or not. I will not give you that money though.
    
    FYI: Anonymous game data is sent to my database so I can maybe do stuff with it later. I don't know why that would sketch you out, this is all fake, but there you go.
    """)

    st.sidebar.caption('Jared Stock | NYC | 2021')


    app_state = st.experimental_get_query_params()
    game_id = app_state['game_id'][0]
    round_number = int(app_state['round_number'][0])
    prev_offer = float(app_state['prev_offer'][0])
    offer = 0.

    # st.write(app_state)
    st.header('Board')
    st.write('')
    col1, col2, col3 = st.beta_columns(3)
    l_cols = VALUES[:len(VALUES) // 2]
    r_cols = VALUES[len(VALUES) // 2:]
    model = joblib.load(f'bankers/{selected_banker}.pkl')

    with col1:
        values_1 = [st.checkbox(str(val), key=session.run_id) for val in VALUES[:len(VALUES) // 2]]
        left_sum = sum([val for i, val in enumerate(l_cols) if not values_1[i]])
    with col2:
        values_2 = [st.checkbox(str(val), key=session.run_id) for val in VALUES[len(VALUES) // 2:]]
        right_sum = sum([val for i, val in enumerate(r_cols) if not values_2[i]])
    values = values_1 + values_2
    choices = [val for i, val in enumerate(VALUES) if values[i]]
    remaining = [val for i, val in enumerate(VALUES) if not values[i]]
    remaining_bigs = [_ for _ in remaining if _ in BIG_VALUES]

    average = np.average(remaining)
    _max = max(remaining)

    if right_sum == 0:
        balance = (left_sum / L_SUM)
    elif left_sum == 0:
        balance = (right_sum / R_SUM)
    else:
        balance = (right_sum / R_SUM) / (left_sum / L_SUM)
    ev = expected_value(remaining)

    with col3:
        st.subheader('Info')
        st.write(f'Round: {round_number}')
        st.write(f'Picked: {len(choices)}')
        st.write(f'Previous Offer: {prev_offer}')
        st.write(f'Expected Value: {round(ev, 0)}')
        st.write(f'Probability of having a big value: {round(len(remaining_bigs) / len(remaining) * 100, 1)}%')

    st.subheader('Banker Offer')

    if len(choices) > 5:
        X = pd.DataFrame({'Round': [round_number], 'Board Average': [ev], 'Previous Offer': [prev_offer]})

        p = model.predict(X)
        offer = float(p[0])

        st.write(f'Offer: ${round(float(offer), 2)}')

        if offer / ev <= 1:
            st.progress(offer / ev)
        else:
            st.progress(1)
        st.caption(f'Offer % of Expected Value: {round((offer / ev) * 100, 2)}%')

    else:
        st.info('Pick values to see offers')

    col14, col15 = st.beta_columns(2)
    if len(choices) == 6 or len(choices) == 11 or len(choices) == 15 or len(choices) == 18 or len(choices) >= 20:
        with col14:
            if st.button('Deal!'):
                round_data = {
                    "Game ID": game_id,
                    "Round": round_number,
                    "Remaining Values": str(remaining),
                    "Board Value": sum(remaining),
                    "Board Average": round(average, 0),
                    "Board Balance": round(balance, 3),
                    "Probability of Big Value": round(len(remaining_bigs) / len(remaining), 3),
                    "Previous Offer": prev_offer,
                    "Offer": round(offer, 0),
                    "Offer Percent of Average": round(offer / average, 4),
                    "model":selected_banker,
                    "datetime":datetime.datetime.now(),
                    "Deal": True
                }
                df = pd.DataFrame(round_data, index=[0])
                populate_round(df, 'player_games')
        with col15:
            if st.button('No Deal!'):
                round_data = {
                    "Game ID": game_id,
                    "Round": round_number,
                    "Remaining Values": str(remaining),
                    "Board Value": sum(remaining),
                    "Board Average": round(average, 0),
                    "Board Balance": round(balance, 3),
                    "Probability of Big Value": round(len(remaining_bigs) / len(remaining), 3),
                    "Previous Offer": prev_offer,
                    "Offer": round(offer, 0),
                    "Offer Percent of Average": round(offer / average, 4),
                    "model":selected_banker,
                    "datetime":datetime.datetime.now(),
                    "Deal": False
                }
                round_number += 1
                st.experimental_set_query_params(round_number=round_number, game_id=game_id, prev_offer=round(offer, 0))
                df = pd.DataFrame(round_data, index=[0])
                populate_round(df, 'player_games')
    data = get_data('player_games')
    data = data.loc[data['Game ID'] == game_id]
    if st.checkbox('Show data'):
        st.write(data)

    visualization.single_line(data, game_id, width=600, height=400)


def expected_value(values: list):
    # variable prb is for probability of each element which is same for each element (uniform)
    prb = 1 / len(values)

    # calculating expectation overall
    sum = 0
    for i in range(0, len(values)):
        sum += (values[i] * prb)

    # returning expectation as sum
    return float(sum)


def get_data(table: str) -> pd.DataFrame:
    res = pd.read_sql_table(table, engine)
    return res


def populate_round(row: pd.DataFrame, table_name: str):
    row.to_sql(table_name, con=engine, if_exists='append', method='multi')
    row.to_pickle('latest_round.pkl')


def populate_player_round(row: pd.DataFrame, table_name: str):
    print(row)
    row.to_sql(table_name, con=engine, if_exists='replace', method='multi')
    # row.to_pickle('latest_round.pkl')


def calc_big_val_prob(df):
    remaining_list = df['Remaining Values'].apply(lambda x: x.replace('[', '').replace(']', '').split(','))
    remaining_list = remaining_list.apply(lambda x: [int(_) for _ in x])
    remaining_bigs = remaining_list.apply(lambda x: [_ for _ in x if _ in BIG_VALUES])
    df['Probability of Big Value'] = round(remaining_bigs.apply(len) / remaining_list.apply(len), 3)
    return df


def new_game():
    st.experimental_set_query_params()
    st.game_id = generate(8)
    st.experimental_set_query_params(round_number=1, game_id=st.game_id, prev_offer=0)
    session.run_id += 1


if __name__ == '__main__':
    main()

import json
import os
import random
import numpy as np
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import visualization
import joblib

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
                       format(user='postgres', pw='Password', host='localhost', port=5432,
                              dbname='dealnodeal'))


def main():
    st.set_page_config(
        page_title="DEAL OR NO DEAL",
        page_icon="🤑",
        initial_sidebar_state="expanded")
    st.sidebar.title('DEAL OR NO DEAL')
    game_id = st.sidebar.text_input('Game ID')
    contestant_name = st.sidebar.text_input('Contestant Name')
    contestant_gender = st.sidebar.selectbox('Contestant Gender', ['Male', 'Female', 'Non-Binary'])
    contestant_race = st.sidebar.selectbox('Contestant Race',
                                           ['White', 'Black', 'Hispanic', 'Asian', 'Hawaiian/Pacific Islander',
                                            'Native American'])
    if st.sidebar.button('Clear saved predictions'):
        with open('preds.json', 'w+') as f:
            json.dump([], f)
    st.header('Board')
    col1, col2, col3 = st.beta_columns(3)
    l_cols = VALUES[:len(VALUES) // 2]
    r_cols = VALUES[len(VALUES) // 2:]
    with col1:
        values_1 = [st.checkbox(str(val)) for val in VALUES[:len(VALUES) // 2]]
        left_sum = sum([val for i, val in enumerate(l_cols) if not values_1[i]])
        left_avg = np.average([val for i, val in enumerate(l_cols) if not values_1[i]])
        # st.write(f'{left_sum}/{L_SUM} | {round(left_sum / L_SUM, 3) * 100}%')
        # st.write(f'Avg: {round(left_avg, 0)}')
    with col2:
        values_2 = [st.checkbox(str(val)) for val in VALUES[len(VALUES) // 2:]]
        right_sum = sum([val for i, val in enumerate(r_cols) if not values_2[i]])
        right_avg = np.average([val for i, val in enumerate(r_cols) if not values_2[i]])
        # st.write(f'{right_sum}/{R_SUM} | {round(right_sum / R_SUM, 3) * 100}%')
        # st.write(f'Avg: {round(right_avg, 0)}')

    values = values_1 + values_2
    choices = [val for i, val in enumerate(VALUES) if values[i]]
    remaining = [val for i, val in enumerate(VALUES) if not values[i]]
    remaining_bigs = [_ for _ in remaining if _ in BIG_VALUES]

    average = np.average(remaining)
    _max = max(remaining)
    standard_deviation = float(np.std(remaining))

    if right_sum == 0:
        balance = (left_sum / L_SUM)
    elif left_sum == 0:
        balance = (right_sum / R_SUM)
    else:
        balance = (right_sum / R_SUM) / (left_sum / L_SUM)
    ev = expected_value(remaining)

    with col3:
        st.subheader('Stats')
        st.write(f'Picked: {len(choices)}')
        st.write(f'Expected Value: {round(ev, 0)}')
        st.write(f'Standard Deviation: {round(standard_deviation, 2)}')
        st.write(
            f'1 Sigma Range: [{max(round(ev - standard_deviation, 0), 0)}:{min(round(ev + standard_deviation, 0), 1_000_000)}]')
        st.write(
            f'1/2 Sigma Range: [{max(round(ev - (standard_deviation / 2), 0), 0)}:{min(round(ev + (standard_deviation / 2), 0), 1_000_000)}]')
        st.write(f'Balance: {round(balance, 2)}')
        st.write(f'Probability of having a big value: {round(len(remaining_bigs) / len(remaining) * 100, 1)}%')

    st.subheader('Offers')
    col11, col12, col13 = st.beta_columns(3)
    with col11:
        round_number = int(st.number_input('Round Number', min_value=1, max_value=15, step=1, format='%i'))
    with col12:
        prev_offer = st.number_input('Previous offer', min_value=0., max_value=5_000_000., step=1000.)
    with col13:
        offer = st.number_input('Current offer', min_value=0., max_value=5_000_000., step=1000.)
    st.write(f'% of Expected Value: {round((offer / ev) * 100, 2)}%')

    if offer / ev <= 1:
        st.progress(offer / ev)
    else:
        st.progress(1)
    col14, col15 = st.beta_columns(2)

    with col14:
        deal = st.checkbox('Deal?')
        amount_won = 0
        if deal:
            amount_won = st.number_input('Amount Won')
    with col15:
        postgame = st.checkbox('Postgame?')

    if st.button('Populate Round'):
        round_data = {
            "Game ID": game_id,
            "Round": round_number,
            "Contestant Name": contestant_name,
            "Contestant Gender": contestant_gender,
            "Contestant Race": contestant_race,
            "Remaining Values": str(remaining),
            "Board Value": sum(remaining),
            "Board Average": round(average, 0),
            "Board Balance": round(balance, 3),
            "Probability of Big Value": round(len(remaining_bigs) / len(remaining), 3),
            "Previous Offer": prev_offer,
            "Offer": offer,
            "Offer Percent of Average": round(offer / average, 4),
            "Deal": deal,
            "Amount Won": amount_won,
            "Postgame": postgame
        }
        df = pd.DataFrame(round_data, index=[0])
        populate_round(df, 'game_data')
    data = get_data('game_data')

    st.subheader('Predictions')
    manual_models = os.listdir('models')
    manual_models = [m for m in manual_models if '.pkl' in m]
    auto_models = os.listdir('models/auto')
    try:
        with open('preds.json') as f:
            hist_preds = json.load(f)
    except:
        hist_preds=[]
    selected_models = st.multiselect('Models', sorted(manual_models))
    if len(choices) > 5:
        X = pd.DataFrame({'Round': [round_number], 'Board Average': [ev], 'Previous Offer': [prev_offer]})
        for selected_model in selected_models:
            model = joblib.load('models/' + selected_model)
            p = model.predict(X)
            hist_preds.append({'Round': round_number, 'Prediction': int(p[0]), 'Model': selected_model})
            st.subheader(selected_model)
            col21, col22,col23 = st.beta_columns(3)
            with col21:
                st.write(f'Prediction: ${round(float(p[0]), 2)}')
            with col22:
                st.write(f'Prediction % of Expected Value: {round((float(p[0]) / ev) * 100, 2)}%')
            with col23:
                if offer != 0:
                    st.write(f'Error: {round((abs(offer - float(p[0])) / offer) * 100, 2)}%')

        if st.checkbox('Auto models'):
            for auto in auto_models:
                model = joblib.load('models/auto/' + auto)
                p = model.predict(X)
                if offer != 0:
                    hist_preds.append({auto: float(p[0]), 'error': round((abs(offer - float(p[0])) / offer) * 100, 2)})
                else:
                    hist_preds.append({auto: float(p[0]), 'error': None})

        if st.checkbox('Show all models'):
            st.write(hist_preds)

        if st.button('Save round'):
            with open('preds.json', 'w+') as f:
                json.dump(hist_preds, f)
    else:
        st.write('Make choices for round')

    # col4, col5 = st.beta_columns(2)
    # with col4:
    if len(hist_preds) > 0:
        visualization.single_line(data, game_id, hist_preds)
    else:
        visualization.single_line(data, game_id)

    # with col5:
    probability_data = data[['Game ID', 'Round', 'Offer', 'Offer Percent of Average', 'Probability of Big Value']]
    visualization.box_plot(probability_data, 'Offer Percent of Average')

    st.subheader('Database')
    st.dataframe(data)

    if st.checkbox('Viz'):
        visualization.offers(data)
    # visualization.profiling(data)


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
    engine = create_engine('postgresql://{user}:{pw}@{host}:{port}/{dbname}'.
                           format(user='postgres', pw='Password', host='localhost', port=5432,
                                  dbname='dealnodeal'))
    res = pd.read_sql_table(table, engine)
    return res


def populate_round(row: pd.DataFrame, table_name: str):
    row.to_sql(table_name, con=engine, if_exists='append', method='multi')
    row.to_pickle('latest_round.pkl')


def overwrite_table(table: pd.DataFrame, table_name: str):
    table.to_sql(table_name, con=engine, if_exists='replace', method='multi')


def calc_big_val_prob(df):
    remaining_list = df['Remaining Values'].apply(lambda x: x.replace('[', '').replace(']', '').split(','))
    remaining_list = remaining_list.apply(lambda x: [int(_) for _ in x])
    remaining_bigs = remaining_list.apply(lambda x: [_ for _ in x if _ in BIG_VALUES])
    df['Probability of Big Value'] = round(remaining_bigs.apply(len) / remaining_list.apply(len), 3)
    return df


if __name__ == '__main__':
    main()

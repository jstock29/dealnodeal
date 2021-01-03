import random

import numpy as np
import streamlit as st
import pandas as pd
from sqlalchemy import create_engine

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
    300_000,
    400_000,
    500_000,
    750_000,
    1_000_000
]

BIG_VALUES = [
    100_000,
    200_000,
    300_000,
    400_000,
    500_000,
    750_000,
    1_000_000]

L_SUM = sum([val for val in VALUES[:len(VALUES) // 2]])
R_SUM = sum([val for val in VALUES[len(VALUES) // 2:]])

engine = create_engine('postgresql://{user}:{pw}@{host}:{port}/{dbname}'.
                       format(user='postgres', pw='Password', host='localhost', port=5432,
                              dbname='dealnodeal'))


def main():
    st.sidebar.title('DEAL OR NO DEAL')
    game_id = st.sidebar.text_input('Game ID')
    contestant_name = st.sidebar.text_input('Contestant Name')
    contestant_gender = st.sidebar.selectbox('Contestant Gender', ['Male', 'Female', 'Non-Binary'])
    contestant_race = st.sidebar.selectbox('Contestant Race',
                                           ['White', 'Black', 'Hispanic', 'Asian', 'Hawaiian/Pacific Islander',
                                            'Native American'])
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
        st.write(f'Balance: {round(balance, 2)}')
        st.write(f'Standard Deviation: {round(standard_deviation, 2)}')
        st.write(f'Probability of having a big value: {round(len(remaining_bigs) / len(remaining) * 100, 1)}%')

    st.subheader('Offers')
    round_number = int(st.number_input('Round Number', min_value=1., max_value=15., step=1.))
    prev_offer = st.number_input('Previous offer', min_value=0., max_value=1_000_000., step=1000.)
    offer = st.number_input('Current offer', min_value=0., max_value=1_000_000., step=1000.)

    col4, col5 = st.beta_columns(2)
    with col4:
        st.write(f'Delta to Expected Value: {round(offer - ev, 0)}')
        st.write(f'Offer % of Expected Value: {round((offer / ev) * 100, 2)}%')
        if offer / ev <= 1:
            st.progress(offer / ev)
        else:
            st.progress(1)
    with col5:
        st.write(f'Delta to Best Remaining: {round((offer - _max), 0)}')
        st.write(f'Offer % of Best Remaining: {round((offer / _max * 100), 2)}%')
        if offer / ev <= 1:
            st.progress(offer / _max)
        else:
            st.progress(1)
    deal = st.checkbox('Deal?')
    amount_won = 0
    if deal:
        amount_won = st.number_input('Amount Won')

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
        st.write(df)
        populate_round(df, 'game_data')

    st.subheader('Database')
    data = get_data('game_data')
    st.dataframe(data)


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


def overwrite_table(table: pd.DataFrame, table_name: str):
    table.to_sql(table_name, con=engine, if_exists='replace', method='multi')


# def calc_big_val_prob(df):
#     remaining_list = df['Remaining Values'].apply(lambda x: x.replace('[', '').replace(']', '').split(','))
#     remaining_list = remaining_list.apply(lambda x: [int(_) for _ in x])
#     remaining_bigs = remaining_list.apply(lambda x: [_ for _ in x if _ in BIG_VALUES])
#     df['Probability of Big Value'] = round(remaining_bigs.apply(len) / remaining_list.apply(len), 3)
#     return df


if __name__ == '__main__':
    main()

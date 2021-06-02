import streamlit as st
from main import get_data
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt


# import pydeck as pdk
# import altair as alt
# from sklearn import preprocessing as pre

def make_correlation_plot(df: pd.DataFrame):
    st.subheader('Correlation Plot')
    fig, ax = plt.subplots(figsize=(10, 10))
    st.write(sns.heatmap(df.corr(), annot=True, linewidths=0.5))
    st.pyplot(fig)


def multi_line(data: pd.DataFrame, averages: pd.DataFrame):
    if data.shape[0] > 10:
        ticks = 10
    else:
        ticks = data.shape[0]
    deals = data.loc[data['Deal']]
    averages.reset_index(inplace=True)
    line_chart = alt.Chart(data).mark_line().encode(
        x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
        y=f'Offer:Q',
        color='Game ID:N',
        opacity=alt.value(0.5)
    )
    line_chart_ev = None
    if st.checkbox('Show Expected Value'):
        line_chart_ev = alt.Chart(data).mark_line().encode(
            x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
            y=f'Board Average:Q',
            color='Game ID:N',
            opacity=alt.value(0.2)
        )

    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['Round'], empty='none')
    selectors = alt.Chart(data).mark_point().encode(
        x='Round:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    points = line_chart.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    text = line_chart.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'Game ID:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(data).mark_rule(color='gray').encode(
        x='Round:Q',
    ).transform_filter(
        nearest
    )
    average_line = alt.Chart(averages).mark_line().encode(
        x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
        y=f'Offer:Q',
        color=alt.value('lightgray'),
        opacity=alt.value(0)
    )
    deals_series = alt.Chart(deals).mark_circle().encode(
        x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
        y=f'Amount Won:Q',
        color=alt.value('red'),
        opacity=alt.value(0)
    )

    if st.checkbox('Show Deals'):
        deals_series = alt.Chart(deals).mark_circle().encode(
            x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
            y=f'Amount Won:Q',
            color='Game ID:N',
            size=alt.value(60),
            opacity=alt.value(1)
        )
    if st.checkbox('Show Average'):
        average_line = alt.Chart(averages).mark_line().encode(
            x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
            y=f'Offer:Q',
            color='Game ID:N',
            opacity=alt.value(1)
        )
    if line_chart_ev:
        layers = alt.layer(line_chart, line_chart_ev, selectors, points, rules, text, average_line,
                           deals_series).properties(
            width=900,
            height=600
        )
    else:
        layers = alt.layer(line_chart, selectors, points, rules, text, average_line, deals_series).properties(
            width=900,
            height=600
        )
    st.altair_chart(layers)


def probability_multi_line(data: pd.DataFrame, averages: pd.DataFrame):
    if data.shape[0] > 10:
        ticks = 10
    else:
        ticks = data.shape[0]
    averages.reset_index(inplace=True)
    line_chart = alt.Chart(data).mark_line().encode(
        x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
        y=f'Offer Percent of Average:Q',
        color='Game ID:N',
        opacity=alt.value(0.5)
    )

    nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['Round'], empty='none')
    selectors = alt.Chart(data).mark_point().encode(
        x='Round:Q',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )
    points = line_chart.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    text = line_chart.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'Game ID:Q', alt.value(' '))
    )

    # Draw a rule at the location of the selection
    rules = alt.Chart(data).mark_rule(color='gray').encode(
        x='Round:Q',
    ).transform_filter(
        nearest
    )
    average_line = alt.Chart(averages).mark_line().encode(
        x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
        y=f'Offer Percent of Average:Q',
        color=alt.value('red'),
        opacity=alt.value(1)
    )
    layers = alt.layer(line_chart, selectors, points, rules, text, average_line).properties(
        width=900,
        height=600
    )
    st.altair_chart(layers)


def box_plot(data: pd.DataFrame, feature: str):
    boxplot = alt.Chart(data).mark_boxplot().encode(
        x='Round:O',
        y=f'{feature}:Q'
    ).properties(
        width=800,
        height=400
    )
    st.altair_chart(boxplot)


def single_line(data: pd.DataFrame, game_id: str, preds: list = []):
    pred_df = pd.DataFrame(preds)
    line = data[data['Game ID'] == game_id]
    if line.shape[0] > 10:
        ticks = 10
    else:
        ticks = line.shape[0]
    offers = alt.Chart(line).mark_line().encode(
        x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
        y=f'Offer:Q',
        color=alt.value('red'),
        opacity=alt.value(1)
    )
    expected = alt.Chart(line).mark_line().encode(
        x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
        y=f'Board Average:Q',
        color=alt.value('gray'),
        opacity=alt.value(.75)
    )
    if len(preds) > 0:
        models = list(set(val for dic in preds for val in dic.values()))
        predictions = alt.Chart(pred_df).mark_line().encode(
            x=alt.X('Round:Q', axis=alt.Axis(tickCount=ticks, grid=False)),
            y=f'Prediction:Q',
            color='Model',
            opacity=alt.value(0.4)
        )
        nearest = alt.selection(type='single', nearest=True, on='mouseover', fields=['Round'], empty='none')
        selectors = alt.Chart(data).mark_point().encode(
            x='Round:Q',
            opacity=alt.value(0),
        ).add_selection(
            nearest
        )
        points = predictions.mark_point().encode(
            opacity=alt.condition(nearest, alt.value(1), alt.value(0))
        )
        text = predictions.mark_text(align='left', dx=5, dy=-5).encode(
            text=alt.condition(nearest, 'Model', alt.value(' '))
        )
        rules = alt.Chart(data).mark_rule(color='gray').encode(
            x='Round:Q',
        ).transform_filter(
            nearest
        )
        layers = alt.layer(offers, expected, predictions,selectors,points, text,rules).properties(
            width=800,
            height=400
        )
    else:
        layers = alt.layer(offers, expected).properties(
            width=800,
            height=400
        )
    st.altair_chart(layers)


def profiling(data: pd.DataFrame):
    data.reset_index(inplace=True)
    # data.drop(['index', 'Game ID'], axis=1, inplace=True)
    pr = ProfileReport(data, explorative=True)
    st_profile_report(pr)


def offers_vs_winnings(data: pd.DataFrame):
    best_offers = data.groupby('Game ID')
    game_best_offers = best_offers['Offer'].max()
    game_amount_won = best_offers['Amount Won'].max()

    potential_loss = game_amount_won - game_best_offers
    games = pd.concat([game_best_offers, game_amount_won, potential_loss], axis=1)
    games.rename(columns={0: "Unrealized Winnings"}, inplace=True, index=str)
    games.reset_index(inplace=True)
    # games.merge(ids)
    st.write(games)
    st.bar_chart(games['Unrealized Winnings'])
    st.write(games['Amount Won'].mean())
    # st.write(games['Unrealized Winnings'].mean())
    # st.write(games['Unrealized Winnings'].median())

    # for i, row, in data.iterrows():
    #     print(row)
    # if row['Amount Won'] != 0:
    #     potential_loss =


def offers(data: pd.DataFrame):
    offer_data = data[['Game ID', 'Round', 'Offer', 'Board Average', 'Deal', 'Amount Won']]
    averages = offer_data.groupby('Round').mean()
    offers_vs_winnings(data)

    if st.checkbox('Show Averages'):
        deals = offer_data[offer_data["Deal"]]
        # offer_data.drop(['Deal'], inplace=True, axis=1)
        # averages.drop(['Deal'], inplace=True, axis=1)
        st.subheader('When do people take deals?')
        deals_rounds = np.histogram(
            deals['Round'], bins=11, range=(0, 10))[0]
        st.bar_chart(deals_rounds)
        st.subheader('Average Offers by Round')
        st.line_chart(averages)
        averages['% of Expected Value'] = averages['Offer'] / averages['Board Average']
        st.line_chart(averages['% of Expected Value'])
        if st.checkbox('Show Averages Data'):
            st.table(averages)
    final_round = st.slider('Final Round', min_value=1, max_value=10, step=1, value=9)
    min_offer, max_offer = st.slider('Offer Range', min_value=0, max_value=2000000, step=1000, value=[0, 2000000])
    game = str(int(st.number_input('Game', min_value=0, max_value=1000, step=1)))

    if len(game) == 1:
        filter_game_id = '000' + game
    elif len(game) == 2:
        filter_game_id = '00' + game
    else:
        filter_game_id = '0' + game

    if filter_game_id == '0000':
        offer_data = offer_data.loc[((offer_data['Round'] <= final_round) &
                                     (offer_data['Offer'] <= max_offer) &
                                     (offer_data['Offer'] >= min_offer))]
    else:
        offer_data = offer_data.loc[offer_data['Game ID'] == filter_game_id]
    multi_line(offer_data, averages)

    probability_data = data[
        ['Game ID', 'Round', 'Offer', 'Offer Percent of Average', 'Probability of Big Value', 'Amount Won']]
    probability_averages = offer_data.groupby('Round').mean()

    probability_multi_line(probability_data, probability_averages)
    # box_plot(probability_data, 'Offer Percent of Average')
    # box_plot(probability_data, 'Offer')


if __name__ == '__main__':
    # offers()
    pass

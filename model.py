import eli5
import joblib
import streamlit as st
import pandas as pd
from sklearn.svm import LinearSVC
from sqlalchemy import create_engine

from sklearn import preprocessing as pre
import sklearn.model_selection as select
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import f_classif, mutual_info_classif, SelectFromModel
from xgboost import XGBRegressor
import lightgbm as lgb

import visualization as viz


engine = create_engine('postgresql://{user}:{pw}@{host}:{port}/{dbname}'.
                       format(user=st.secrets['username'], pw=st.secrets['password'], host=st.secrets['host'], port=5432,
                              dbname=st.secrets['dbname']))


MODELS = {
    "Linear OLS": linear_model.LinearRegression(),
    # "Logistic Regression": linear_model.LogisticRegression(),
    # "Stochastic Gradient Descent": linear_model.SGDRegressor(),
    "Lasso": linear_model.Lasso(),
    "ElasticNet CV": linear_model.ElasticNetCV(),
    "Decision Tree": DecisionTreeRegressor(random_state=69),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=69),
    "XGBoost": XGBRegressor(),
    "LightGBM": lgb.LGBMRegressor(),
    # "MLP Regressor": MLPRegressor()
}


def split_data(X, y):
    test_size = st.slider('Test Split Percentage', 0, 100, 20, 1) / 100
    X_train, X_test, y_train, y_test = select.train_test_split(
        X, y, test_size=test_size, random_state=69)
    st.write(f'Training: {X_train.shape[0]} | Test: {X_test.shape[0]}')

    return X_train, X_test, y_train, y_test


def equity_check(df: pd.DataFrame):
    white = df[df['Contestant Race'] == 'White']
    not_white = df[df['Contestant Race'] != 'White']
    male = df[df['Contestant Gender'] == 'Male']
    not_male = df[df['Contestant Gender'] != 'Male']

    white_avg = white[white['Amount Won'] != 0]['Amount Won'].mean()
    not_white_avg = not_white[not_white['Amount Won'] != 0]['Amount Won'].mean()
    male_avg = male[male['Amount Won'] != 0]['Amount Won'].mean()
    not_male_avg = not_male[not_male['Amount Won'] != 0]['Amount Won'].mean()
    st.write(white_avg)
    st.write(not_white_avg)
    st.write(male_avg)
    st.write(not_male_avg)


def preprocess_data(categorical_cols: list, numeric_cols: list):
    numerical_transformer = SimpleImputer(strategy='median')
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    return preprocessor


def normalize_column(df: pd.DataFrame, col: str) -> pd.DataFrame:
    scaler = pre.StandardScaler()
    df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    return df


def feature_selection(X_train, y_train):
    f = f_classif(X_train, y_train)
    st.write('f values')
    st.write(f[0])

    mi = mutual_info_classif(X_train, y_train)
    st.write('Estimated mutual information between each feature and the target')
    st.write(mi)


def exclude_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[(df['Board Value'] <= 3418416)]
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['Remaining Values'] = df['Remaining Values'].astype(str).str.replace('[', '')
    df['Remaining Values'] = df['Remaining Values'].astype(str).str.replace(']', '')
    df['Remaining Values'] = df['Remaining Values'].astype(str).str.split(',')
    df['Board Median'] = df['Remaining Values'].median()

    return df


def train_model(data: pd.DataFrame):
    selected_model = st.sidebar.selectbox('Model', list(MODELS.keys()))
    model = MODELS[selected_model]
    if st.checkbox('Exclude outliers'):
        data = exclude_outliers(data)
    if 'RandomForest' in str(model):
        model.n_estimators = st.number_input('Number of Random Forest Estimators', 1, 10000, 100)
    y = data['Offer']
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [str(cname) for cname in data.columns if
                        # data[cname].nunique() < 10 and
                        data[cname].dtype == "object"]
    # data = feature_engineering(data)

    # Select numerical columns
    numerical_cols = [str(cname) for cname in data.columns if
                      data[cname].dtype in ['int64', 'float64']]
    st.subheader('Data')
    categorical_feature_cols = st.multiselect('Categorical Features', categorical_cols, [])
    numeric_feature_cols = st.multiselect('Numeric Features', numerical_cols,
                                          ['Round', 'Board Value', 'Board Average', 'Previous Offer',
                                           'Probability of Big Value'])
    all_cols = list(data.columns)
    drop_cols = list(set(all_cols) - set(categorical_feature_cols) - set(numeric_feature_cols))
    data = data.drop(drop_cols, axis=1)
    X = data

    if st.checkbox('Proflie?'):
        viz.profiling(X)

    X_train, X_valid, y_train, y_valid = split_data(X, y)
    #
    # if st.checkbox('Lazy Predict'):
    #     reg = LazyRegressor(ignore_warnings=False, custom_metric=None)
    #     models, predictions = reg.fit(X_train, X_valid, y_train, y_valid)
    #     st.table(models)

    st.subheader('Feature Selection')
    if st.checkbox('Show'):
        feature_selection(X_train, y_train)

    preprocessor = preprocess_data(categorical_feature_cols, numeric_feature_cols)

    st.subheader('Model Training')
    feature_cols, features = None, None
    if selected_model == 'MLP Regressor':
        model.activation = st.selectbox('Activation Function', ['identity', 'logistic', 'tanh', 'relu'], 3)
        model.solver = st.selectbox('Solver', ['adam', 'lbfgs', 'sgd'], 0)
        if model.solver == 'sgd':
            model.learning_rate = st.selectbox('Learning Rate', ['constant', 'invscaling', 'adaptive'], 0)
            model.power_t = st.number_input('Power T', 0., 1., 0.5, 0.01)
        if model.solver == 'adam':
            model.epsilon = st.number_input('Epsilon', 0., 1., 1e-8, 1e-8)
        model.alpha = st.number_input('Alpha', 0., 1., 0.0001, 0.0001)

    feature_selector = st.selectbox('Feature Selection Type', ['Linear SVC', 'None'], 1)
    model_pipeline = Pipeline([('preprocessor', preprocessor)])
    if feature_selector == "Linear SVC":
        feature_selection_model = SelectFromModel(LinearSVC())
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('scaler', pre.StandardScaler(with_std=False, with_mean=False)),
                                         ('feature_selection', feature_selection_model),
                                         ('model', model)
                                         ])
        features = model_pipeline.named_steps['feature_selection']

    elif feature_selector == 'None':
        model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                         ('scaler', pre.StandardScaler(with_std=False, with_mean=False)),
                                         ('model', model)
                                         ])

    # Preprocessing of training data, fit model
    model_pipeline.fit(X_train, y_train)
    if features:
        feature_cols = list(X.columns[features.get_support()])
        st.write(feature_cols)

    # Preprocessing of validation data, get predictions
    preds = model_pipeline.predict(X_valid)

    numeric_features_list = list(numeric_feature_cols)
    if (len(categorical_feature_cols) > 0):
        onehot_columns = list(model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps[
                                  'onehot'].get_feature_names(input_features=categorical_feature_cols))
        numeric_features_list.extend(onehot_columns)

    # Evaluate the model
    mae = mean_absolute_error(y_valid, preds)
    r2 = r2_score(y_valid, preds)
    st.write(f'Single Mean Absolute Error: {round(mae, 3)}')
    st.write(f'Single R^2: {round(r2, 3)}')
    joblib.dump(model_pipeline, f'models/{selected_model}_{round(mae, 3)}_{round(r2, 3)}.pkl')

    st.subheader('Model Training - Cross Validation')
    n_folds = st.number_input('N Folds', 1, 50, 5)
    mae_scores = -1 * cross_val_score(model_pipeline, X, y, cv=n_folds, scoring='neg_mean_absolute_error')

    st.write(f"MAE scores:\n", mae_scores)
    st.write(f"Average MAE: {round(mae_scores.mean(), 3)}")
    st.write(f"MAE Spread: {round(mae_scores.max() - mae_scores.min(), 3)}")
    if not features:
        exp = eli5.explain_weights(model_pipeline.named_steps['model'], top=50, feature_names=numeric_features_list)
    else:
        exp = eli5.explain_weights(model_pipeline.named_steps['model'], top=50, feature_names=feature_cols)
    exp_df = eli5.formatters.format_as_dataframe(exp)
    st.dataframe(exp_df)
    joblib.dump(model_pipeline, f'models/{selected_model}_{n_folds}folds_{round(mae_scores.mean(), 3)}.pkl')
    return model_pipeline


def find_best_model(data: pd.DataFrame):
    performances = {}
    y = data['Offer']
    # Select categorical columns with relatively low cardinality (convenient but arbitrary)
    categorical_cols = [str(cname) for cname in data.columns if
                        data[cname].nunique() < 10 and
                        data[cname].dtype == "object"]

    # Select numerical columns
    numerical_cols = [str(cname) for cname in data.columns if
                      data[cname].dtype in ['int64', 'float64']]
    st.subheader('Data')
    categorical_feature_cols = st.multiselect('Categorical Features', categorical_cols, [])

    # for num_feature in numerical_cols:
    numeric_feature_cols = st.multiselect('Numeric Features', numerical_cols,
                                          ['Round', 'Board Value', 'Board Average', 'Previous Offer',
                                           'Probability of Big Value'])

    all_cols = list(data.columns)
    drop_cols = list(set(all_cols) - set(categorical_feature_cols) - set(numeric_feature_cols))
    data = data.drop(drop_cols, axis=1)
    X = data

    X_train, X_valid, y_train, y_valid = split_data(X, y)

    n_folds = st.number_input('N Folds', 1, 50, 5)

    for selected_model in list(MODELS.keys()):
        st.write(selected_model)
        model = MODELS[selected_model]
        performances[selected_model] = {}
        preprocessor = preprocess_data(categorical_feature_cols, numeric_feature_cols)
        if selected_model == 'Lasso':
            alphas = [1, 5, 10, 15, 25, 35, 50, 69, 75, 90, 95, 99]
            best = 1_000_000
            for alpha in alphas:
                model.alpha = alpha / 100
                model.max_iter = 25_000
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                 ('scaler', pre.StandardScaler(with_std=False, with_mean=False)),
                                                 ('model', model)
                                                 ])
                model_pipeline.fit(X_train, y_train)
                mae_scores = -1 * cross_val_score(model_pipeline, X, y, cv=n_folds, scoring='neg_mean_absolute_error')
                if mae_scores.mean() < best:
                    best = mae_scores.mean()
                    performances[selected_model]['scores'] = list(mae_scores)
                    performances[selected_model]['avg_score'] = mae_scores.mean()
                    performances[selected_model]['model'] = model_pipeline
                    performances[selected_model]['alpha'] = alpha
                    joblib.dump(model_pipeline, f'models/auto/{selected_model}.pkl')

        elif selected_model == 'Random Forest':
            print('Random Forest Auto')
            estimators = [
                5,
                10,
                15,
                25,
                50,
                100,
                200,
                250,
                500
            ]
            best = 1_000_000
            for estimator in range(1,500):
                model.n_estimators = estimator
                model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                                 ('scaler', pre.StandardScaler(with_std=False, with_mean=False)),
                                                 ('model', model)
                                                 ])
                model_pipeline.fit(X_train, y_train)

                mae_scores = -1 * cross_val_score(model_pipeline, X, y, cv=n_folds, scoring='neg_mean_absolute_error')

                if mae_scores.mean() < best:
                    best = mae_scores.mean()
                    print(best, estimator)
                    performances[selected_model]['scores'] = list(mae_scores)
                    performances[selected_model]['avg_score'] = mae_scores.mean()
                    performances[selected_model]['model'] = model_pipeline
                    performances[selected_model]['estimators'] = estimator
                    joblib.dump(model_pipeline, f'models/auto/{selected_model}.pkl')

        else:
            model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                             ('scaler', pre.StandardScaler(with_std=False, with_mean=False)),
                                             ('model', model)
                                             ])
            model_pipeline.fit(X_train, y_train)

            mae_scores = -1 * cross_val_score(model_pipeline, X, y, cv=n_folds, scoring='neg_mean_absolute_error')
            performances[selected_model]['scores'] = list(mae_scores)
            performances[selected_model]['avg_score'] = mae_scores.mean()
            performances[selected_model]['model'] = model_pipeline

            joblib.dump(model_pipeline, f'models/auto/{selected_model}.pkl')
    if st.checkbox('Show All'):
        st.write(performances)
    print(performances.items())
    avgs = [x[1]['avg_score'] for x in performances.items()]
    best = min(avgs)
    best_obj = [x for x in performances.items() if x[1]['avg_score'] == best]
    st.write(best_obj)


@st.cache(ttl=3600)
def get_data():
    data = pd.read_sql_table('game_data', engine)
    data.drop(['index', 'Game ID', 'Contestant Name'], inplace=True, axis=1)
    return data


def main():
    st.set_page_config(
        page_title="Robo Banker",
        page_icon="🤖",
        initial_sidebar_state="expanded")
    st.sidebar.write("""
        # Robo Banker
        """)

    mode = st.sidebar.selectbox('Training Mode', ['Manual', 'Auto'], 0)
    data = get_data()
    if mode == 'Manual':
        train_model(data)
    else:
        find_best_model(data)


if __name__ == '__main__':
    main()

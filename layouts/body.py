from dash import html, dash_table, dcc
import pandas as pd
from algorithms import KNN_MLP, GA
import split_data


data = pd.read_csv("data/water_potability.csv")
std_data = pd.read_csv("data/standardized_water_potability.csv")


def missing_data_table():
    x = data.drop('Potability', axis=1)
    missing_values = pd.DataFrame(x.isnull().sum(), columns=["Missing Values Count"])
    missing_values.index.name = "Values"

    missing_data_table = dash_table.DataTable(
        data=missing_values.reset_index().to_dict("records"),
        columns=[{"name": i, "id": i} for i in missing_values.reset_index().columns],
        style_cell={
            "textAlign": "left",
            "whiteSpace": "normal",
            "height": "auto",
            "weight": "px",
            "textOverflow": "ellipsis",
            "pointerEvents": "none",
        },
        style_header={
            "textAlign": "center",
            "backgroundColor": "rgba(14, 186, 199, 0.25)",
            "fontWeight": "bold",
            "color": "rgb(35, 20, 39)",
            "border": "2px solid white",
        },
        style_data={
            "border": "2px solid white",
            "fontWeight": "lighter",
            "backgroundColor": "transparent",
            "color": "white",
        },
    )
    return missing_data_table

test_value = 0.5

std_data_impt_knn = KNN_MLP.imputation(std_data, 5, "Standardized_Data")
X_train_std_knn, y_train_std_knn, X_test_std_knn, y_test_std_knn = split_data.split(pd.read_csv(std_data_impt_knn), test_value)

data_impt_knn = KNN_MLP.imputation(data, 5, 'Data')
X_train_knn, y_train_knn, X_test_knn, y_test_knn = split_data.split(pd.read_csv(data_impt_knn), test_value)

std_data_impt_ga = GA.imputation_std_data(std_data, "Standardized_Data", test_value)
X_train_std_ga, y_train_std_ga, X_test_std_ga, y_test_std_ga = split_data.split(pd.read_csv(std_data_impt_ga), test_value)

data_impt_ga = GA.imputation_data(data, 'Data', test_value)
X_train_ga, y_train_ga, X_test_ga, y_test_ga = split_data.split(pd.read_csv(data_impt_ga), test_value)

def get_content():
    content = html.Div([html.Div(children=[html.P(["Using 'water potability' dataset to study the different types of optimisation and classification methods,",html.Br(),
                                                    "and see its performance in imputing missing values."],
                                            className="description"),
                                           html.P(["The following table shows the number of missing values of each variable:"],
                                            className="desc"),
                                            missing_data_table()],
                                className="describe"),
                        html.Div(children=[html.Label(children=["Standardized Data"], id="std-label", className="title"),
                                            html.Div(children=[html.Label(["KNN Imputation"]),
                                                                html.Div(children=[dcc.Graph(figure=KNN_MLP.classifying_data_KNN_bar(X_train_std_knn, y_train_std_knn, X_test_std_knn, y_test_std_knn, test_value)),
                                                                                    dcc.Graph(figure=KNN_MLP.classifying_data_KNN_pie(X_train_std_knn, y_train_std_knn, X_test_std_knn, y_test_std_knn))]),
                                                                html.Div(children=[dcc.Graph(figure=KNN_MLP.classifying_data_MLP_bar(X_train_std_knn, y_train_std_knn, X_test_std_knn, y_test_std_knn, test_value)),
                                                                                    dcc.Graph(figure=KNN_MLP.classifying_data_MLP_pie(X_train_std_knn, y_train_std_knn, X_test_std_knn, y_test_std_knn))])],
                                                    className="mixedTest"),
                                            html.Div(children=[html.Label(["Genetic Algorithm"]),
                                                               html.Div(children=[dcc.Graph(figure=KNN_MLP.classifying_data_KNN_bar(X_train_std_ga, y_train_std_ga, X_test_std_ga, y_test_std_ga, test_value)),
                                                                                    dcc.Graph(figure=KNN_MLP.classifying_data_KNN_pie(X_train_std_ga, y_train_std_ga, X_test_std_ga, y_test_std_ga))]),
                                                                html.Div(children=[dcc.Graph(figure=KNN_MLP.classifying_data_MLP_bar(X_train_std_ga, y_train_std_ga, X_test_std_ga, y_test_std_ga, test_value)),
                                                                                    dcc.Graph(figure=KNN_MLP.classifying_data_MLP_pie(X_train_std_ga, y_train_std_ga, X_test_std_ga, y_test_std_ga))])],
                                                    className="cleanTest"),
                                            html.Div(children=[html.Label(["MICE Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["Escape Optimization"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["HHO Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["SMA Algorithm"])], className="cleanTest")],
                                className="standardized-data"),
                        html.Div(children=[html.Label(children=["Normal Data"], id="nrml-label", className="title"),
                                            html.Div(children=[html.Label(["KNN Imputation"]),
                                                                html.Div(children=[dcc.Graph(figure=KNN_MLP.classifying_data_KNN_bar(X_train_knn, y_train_knn, X_test_knn, y_test_knn, test_value)),
                                                                                    dcc.Graph(figure=KNN_MLP.classifying_data_KNN_pie(X_train_knn, y_train_knn, X_test_knn, y_test_knn))]),
                                                                html.Div(children=[dcc.Graph(figure=KNN_MLP.classifying_data_MLP_bar(X_train_knn, y_train_knn, X_test_knn, y_test_knn, test_value)),
                                                                                    dcc.Graph(figure=KNN_MLP.classifying_data_MLP_pie(X_train_knn, y_train_knn, X_test_knn, y_test_knn))])],
                                                    className="mixedTest"),
                                            html.Div(children=[html.Label(["ESC Optimization"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["MICE Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["GA Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["fourth Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["fifth Algorithm"])], className="cleanTest")],
                                className="normal-data")],
            className="main-content")

    return content

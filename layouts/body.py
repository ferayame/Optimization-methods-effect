from dash import html, dash_table, dcc
import pandas as pd
import algorithms.KNN_MLP as aknn
import split_data


data = pd.read_csv("webapp/data/water_potability.csv")
std_data = pd.read_csv("webapp/data/standardized_water_potability.csv")


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

std_data_impt = aknn.imputation(std_data, 5, "Standardized_Data")
X_train_std, y_train_std, X_test_std, y_test_std = split_data.split(pd.read_csv(std_data_impt), test_value)

data_impt = aknn.imputation(data, 5, 'Data')
X_train, y_train, X_test, y_test = split_data.split(pd.read_csv(data_impt), test_value)

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
                                                                html.Div(children=[dcc.Graph(figure=aknn.classifying_data_KNN_bar(X_train_std, y_train_std, X_test_std, y_test_std, test_value)),
                                                                                    dcc.Graph(figure=aknn.classifying_data_KNN_pie(X_train_std, y_train_std, X_test_std, y_test_std))]),
                                                                html.Div(children=[dcc.Graph(figure=aknn.classifying_data_MLP_bar(X_train_std, y_train_std, X_test_std, y_test_std, test_value)),
                                                                                    dcc.Graph(figure=aknn.classifying_data_MLP_pie(X_train_std, y_train_std, X_test_std, y_test_std))])],
                                                    className="mixedTest"),
                                            html.Div(children=[html.Label(["Escape Optimization"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["MICE Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["Genetic Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["HHO Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["SMA Algorithm"])], className="cleanTest")],
                                className="standardized-data"),
                        html.Div(children=[html.Label(children=["Normal Data"], id="nrml-label", className="title"),
                                            html.Div(children=[html.Label(["KNN Imputation"]),
                                                                html.Div(children=[dcc.Graph(figure=aknn.classifying_data_KNN_bar(X_train, y_train, X_test, y_test, test_value)),
                                                                                    dcc.Graph(figure=aknn.classifying_data_KNN_pie(X_train, y_train, X_test, y_test))]),
                                                                html.Div(children=[dcc.Graph(figure=aknn.classifying_data_MLP_bar(X_train, y_train, X_test, y_test, test_value)),
                                                                                    dcc.Graph(figure=aknn.classifying_data_MLP_pie(X_train, y_train, X_test, y_test))])],
                                                    className="mixedTest"),
                                            html.Div(children=[html.Label(["ESC Optimization"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["MICE Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["GA Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["fourth Algorithm"])], className="cleanTest"),
                                            html.Div(children=[html.Label(["fifth Algorithm"])], className="cleanTest")],
                                className="normal-data")],
            className="main-content")

    return content

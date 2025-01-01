import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.impute import KNNImputer
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def imputation(data, k, filename):
    imputer = KNNImputer(n_neighbors=k)
    imputed_data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    fileDirectory = f"output/Imputed_{filename}_using_KNN.csv"
    imputed_data.to_csv(fileDirectory, index=False)
    return fileDirectory

def evaluate_performance(y_true, y_pred):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred, average="macro"),
        "Precision": precision_score(y_true, y_pred, average="macro"),
        "F1 Score": f1_score(y_true, y_pred, average="macro"),
    }
    return metrics

def visualize_performance_pie(metrics, title):
    fig = make_subplots(rows=2, cols=2, specs=[[{"type": "Pie"}]*2,[{"type": "Pie"}]*2],
                        horizontal_spacing=0.2)
    pie_colors = ['#468dff', ' #344fff']
    pie_labels = ['Correct', 'Incorrect']
    i = 0
    j = 0
    annotations = []
    for (metric, value) in metrics.items():
        sizes = [value, 1 - value]
        if i < 2:
            fig.add_trace(go.Pie(labels=pie_labels, values=sizes, marker_colors=pie_colors, textinfo='percent', hole=.3), row=1, col=i+1)
            annotations.append(dict(x=(i+1)/2, y=0.7, text=metric, showarrow=False, xanchor='center', yanchor='bottom'))
            i += 1
        else:
            fig.add_trace(go.Pie(labels=pie_labels, values=sizes, marker_colors=pie_colors, textinfo='percent', hole=.3), row=2, col=j+1)
            annotations.append(dict(x=(j+1)/2, y=0.2, text=metric, showarrow=False, xanchor='center', yanchor='bottom'))
            j += 1
            
    fig.update_layout(title_text=f'{title} - Performance',annotations=annotations,
                      paper_bgcolor = 'rgba(0,0,0,0)',
                      plot_bgcolor = 'rgba(0,0,0,0)',
                      font=dict(color="white"))
    return fig

def visualize_performance_bar(metrics, title, test_size):
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())

    fig = make_subplots(rows=1, cols=1, specs=[[{"type": "bar"}]])
    fig.add_trace(go.Bar(x=metric_names, y=metric_values, marker={'color':'#344fff'},
                         text=[f'{v:.4f}' for v in metric_values], textposition='auto'), row=1, col=1)
    fig.update_yaxes(range=[0, 1], row=1, col=1, showline=True, linewidth=2, linecolor='#5ee1ff', mirror=True, gridcolor='#5ee1ff', gridwidth=0.5)
    fig.update_xaxes(showline=True, linewidth=2, linecolor='#5ee1ff', mirror=True)
    fig.update_layout(title_text=f'{title} - Performance Metrics with a Test of size {test_size}', title_x=0.5,
                    paper_bgcolor = 'rgba(0,0,0,0)',
                    plot_bgcolor = 'rgba(0,0,0,0)',
                    font=dict(color="white"))

    return fig
    
def test_bar(X_train, X_test, y_train, y_test, classifier, title, test_size):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metrics = evaluate_performance(y_test, y_pred)

    return visualize_performance_bar(metrics, title, test_size)

def test_pie(X_train, X_test, y_train, y_test, classifier, title):
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    metrics = evaluate_performance(y_test, y_pred)

    return visualize_performance_pie(metrics, title)

def classifying_data_KNN_bar(X_train, y_train, X_test, y_test, test_size):
    fig = test_bar(X_train, y_train, X_test, y_test, KNeighborsClassifier(n_neighbors=5), 'KNN Classifier', test_size)
    return fig

def classifying_data_KNN_pie(X_train, y_train, X_test, y_test):
    fig = test_pie(X_train, y_train, X_test, y_test, KNeighborsClassifier(n_neighbors=5), 'KNN Classifier')
    return fig

def classifying_data_MLP_bar(X_train, y_train, X_test, y_test, test_size):
    fig = test_bar(X_train, y_train, X_test, y_test, MLPClassifier(random_state=42, max_iter=1000), 'MLP Classifier', test_size)
    return fig

def classifying_data_MLP_pie(X_train, y_train, X_test, y_test):
    fig = test_pie(X_train, y_train, X_test, y_test, MLPClassifier(random_state=42, max_iter=1000), 'MLP Classifier')
    return fig
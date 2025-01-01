import dash
from dash import html
from layouts import footer, body


app = dash.Dash(__name__, external_stylesheets=[
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css"
])

app.title = "Imputation and Classification"

def layout():
    return html.Div([body.get_content(), footer.get_footer()], className="container")

app.layout = layout()

if __name__ == "__main__":
    app.run_server(debug=True, use_reloader=True, reloader_type='watchdog')

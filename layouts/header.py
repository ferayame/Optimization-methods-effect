from dash import html, dcc
from flask import session

def get_header():
    # dad : Data Analysis Dashboard
    # tc : Text Container
    header = html.Header(id='dad', children=[
                html.Img(src='../assets/images/groupname/name.png', alt='ferayame', className='logo'),
            html.Div(id='tc', children=[
                html.H1("Main Title", className="Main-title"),
                html.P(["subtitle",html.Br(),"discreption",html.Br(),"or information"],
                                className="Sub-title")],
                className='container'),
            html.Div(id = 'btn', children =
                     dcc.Upload(id='upload-data', children=
                                html.Button(children=["Upload File", html.Img(src='../assets/images/icons/icons8-excel-50.png',
                                                              alt='excel', className='excelogo')], className="uploadexl"),
            accept=".xlsx"))],
        className='header')
    return header
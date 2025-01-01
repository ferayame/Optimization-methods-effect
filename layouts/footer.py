from dash import html

# The organization github link
github_rul = 'https://github.com/ferayame'

def get_footer():
    footer = html.Footer(children=[
                html.Div(children = [html.Ul([html.Li(html.P("© 2024 ferayame. All Rights Reserved."),
                                                      className="nested-list1"),
                                      html.Li(html.A(html.Img(src='../assets/images/icons/github-mark-white.png',
                                                              alt='github', className='githublogo'), href=github_rul, target='_blank', className='footerlink'),
                                              className="nested-list2")],
                                             className="list")],
                className='footer-stf')])
    return footer
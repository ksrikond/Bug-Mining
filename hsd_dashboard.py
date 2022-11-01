import os
import dash
import dash_table
import json

import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import src.utils.hsd_util as _util

from dash.dependencies import Output, Input, State
from wordcloud import WordCloud
from flask import Flask

from main import HsdMiningAnalysis
from src.utils.logger_config import api_logger

cfg = _util.load_configuration('config.yaml')

raw_data_file = cfg['dashboard']['input']['rawDataFile']
algorithm = cfg['dashboard']['algorithm']
tableDataFile = cfg['dashboard']['output']['tableDataFile']

root_path = os.getcwd()
src_path = os.path.join(root_path, 'src')
datafiles_path = os.path.join(src_path, 'data', 'data_files')
model_dir = os.path.join(src_path, 'model')

stopwords_exceptions_filename = os.path.join(datafiles_path, 'raw_data', cfg['dashboard']['stopwordsAndExceptions'])
tf_idf_filename = os.path.join(datafiles_path, 'tf_idf_data', cfg['dashboard']['input']['tfIdfFile'])
df_tf_idf = pd.read_csv(tf_idf_filename)
vocab = []

if algorithm == "co-occurrence":
    co_occurrence_path = os.path.join(model_dir, cfg['dashboard']['input']['wordEmbeddingModelFile'][algorithm])
    co_occurrence = _util.load_co_occurrence_matrix(co_occurrence_path)
    vocab = list(co_occurrence.keys())
    lastmt = os.stat(co_occurrence_path).st_mtime
    word_clusters_path = os.path.join(datafiles_path, 'co_occurrence_data',
                                      cfg['dashboard']['output']['wordClustersFile'][algorithm])

if algorithm == "word2vec":
    word2vec_model_path = os.path.join(model_dir, cfg['dashboard']['input']['wordEmbeddingModelFile'][algorithm])
    word2vec_model = _util.load_word2vec_model(word2vec_model_path)
    vocab = list(word2vec_model.wv.vocab)
    lastmt = os.stat(word2vec_model_path).st_mtime
    word_clusters_path = os.path.join(datafiles_path, 'word2_vec_data',
                                      cfg['dashboard']['output']['wordClustersFile'][algorithm])

local_df = df_tf_idf
words, similar_words = [], []

"""
#
# Page layout and contents starts here
#
"""

NAVBAR = dbc.Navbar(
    children=[
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src='static/es-logo.png', height="30px")),
                    dbc.Col(
                        dbc.NavbarBrand("HSD MINING ANALYSIS", className="ml-4")
                    ),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://hsdes.intel.com/appstore/article/",
        )
    ],
    color="dark",
    dark=True,
    sticky="top",
)

WORDCLOUD_PLOTS = [
    dbc.CardHeader(
        [
            html.H5("Most Frequently Occurring Themes from HSD Data",
                    style={'display': 'inline-block', 'margin-right': '5px'}),
            html.I(className="fas fa-question-circle fa-lg", id="target-info-first-section",
                   style={'float': 'right', 'margin': '5px'}),
            dbc.Tooltip(
                "This section presents the top recurrent themes from HSD data you processed. By default, it shows top 25 themes."
                " You can view more themes using select number of features option. Unwanted themes can be removed using 'Modify feature list' section.",
                target="target-info-first-section"),
        ]),
    dbc.Alert(
        "Not enough data to render these plots, please adjust the filters",
        id="no-data-alert-wordcloud",
        color="warning",
        style={"display": "none"},
    ),
    dbc.Alert(
        "Please select year first to filter data by month",
        id="select-year-alert",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            dbc.Row(
                [
                    html.Div(id='hidden-div', style={"display": "none"}),
                    dbc.Col(html.P(["Select features, month, year to filter results: "]), md=4),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="number-features",
                                placeholder="Select number of features",
                                clearable=True,
                                options=[
                                    {"label": int(i), "value": int(i)}
                                    for i in range(25, 201)
                                    # replace 201 with len(df_tf_idf.columns[:-4]) to enable all the features
                                ]
                            )
                        ],
                        md=3
                    ),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="year-dropdown",
                                placeholder="Select year",
                                multi=True,
                                clearable=True,
                                options=[
                                    {"label": int(year), "value": int(year)}  # complete this part
                                    for year in df_tf_idf['Year'].unique()
                                ]
                            )
                        ],
                        md=2
                    ),
                    dbc.Col(
                        [
                            dcc.Dropdown(
                                id="month-dropdown",
                                placeholder="Select month",
                                multi=True,
                                clearable=True,
                                options=[
                                    {"label": "January", "value": 1},
                                    {"label": "February", "value": 2},
                                    {"label": "March", "value": 3},
                                    {"label": "April", "value": 4},
                                    {"label": "May", "value": 5},
                                    {"label": "June", "value": 6},
                                    {"label": "July", "value": 7},
                                    {"label": "August", "value": 8},
                                    {"label": "September", "value": 9},
                                    {"label": "October", "value": 10},
                                    {"label": "November", "value": 11},
                                    {"label": "December", "value": 12},
                                ]
                            )
                        ],
                        md=2
                    ),
                    dbc.Col(
                        [
                            html.Button('Submit',
                                        id='tfidf-button',
                                        type='submit',
                                        n_clicks=0)
                        ]
                    )
                ],
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Loading(
                            id="loading-frequencies",
                            children=[dcc.Graph(id="hsd-frequency_figure")],
                            type="default",
                        )
                    ),
                    dbc.Col(
                        [
                            dcc.Tabs(
                                id="tabs",
                                children=[
                                    dcc.Tab(
                                        label="Treemap",
                                        children=[
                                            dcc.Loading(
                                                id="loading-treemap",
                                                children=[dcc.Graph(id="hsd-treemap")],
                                                type="default",
                                            )
                                        ],
                                    ),
                                    dcc.Tab(
                                        label="Wordcloud",
                                        children=[
                                            dcc.Loading(
                                                id="loading-wordcloud",
                                                children=[
                                                    dcc.Graph(id="hsd-wordcloud")
                                                ],
                                                type="default",
                                            )
                                        ],
                                    ),
                                ],
                            )
                        ], md=8,
                    ),
                ], className="mt-2"
            )
        ],
        style={"marginTop": 0, "marginBottom": 0},
    ),
]

LEFT_CARD = dbc.Jumbotron(
    [
        html.H4(children="Select parameters to filter results", className="display-5"),
        html.Hr(className="my-2"),
        html.Label("Select number of words(n)", className="lead"),
        html.P(
            "(If not selected, takes default: 25)",
            style={"fontSize": 12, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="n-dropdown",
            clearable=True,
            placeholder="Select n value",
            style={"font-size": 12},
            options=[
                {"label": i, "value": i}  # complete this part
                for i in range(1, 201)
            ]
        ),
        dcc.Checklist(
            options=[],

        ),
        html.Label("Select number of similar words(top_n)", style={"marginTop": 50}, className="lead"),
        html.P(
            "(If not selected, takes default: 10)",
            style={"fontSize": 12, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="top-n-dropdown",
            clearable=True,
            placeholder="Select top_n value",
            style={"marginBottom": 0, "font-size": 12},
            options=[
                {"label": i, "value": i}  # complete this part
                for i in range(1, 26)
            ]
        ),
        html.Label("Select a depth of connected graph(depth)", style={"marginTop": 50}, className="lead"),
        html.P(
            "(If not selected, takes default: 2)",
            style={"fontSize": 12, "font-weight": "lighter"},
        ),
        dcc.Dropdown(
            id="depth-dropdown",
            clearable=True,
            placeholder="Select depth value",
            style={"marginBottom": 0, "font-size": 12},
            options=[
                {"label": i, "value": i}  # complete this part
                for i in range(0, 11)
            ]
        ),
        html.Button('Submit',
                    id='wordvec-button',
                    type='submit',
                    n_clicks=0,
                    style={
                        "marginTop": 20
                    })
    ]
)

WORD2VEC_PLOTS = [
    dbc.CardHeader(
        [
            html.H5("Context Graph of Recurrent Themes from HSD Data",
                    style={'display': 'inline-block', 'margin-right': '5px'}),
            html.I(className="fas fa-question-circle fa-lg", id="target-info-third-section",
                   style={'float': 'right', 'margin': '5px'}),
            dbc.Tooltip(
                "This section presents context graph for the top recurrent themes. By default, it shows context graph for top 25 themes."
                "You can view for more themes using 'select number of features' option from the first section and 'Select number of words' option from the left panel. For example, If you want to see data from top 30 recurrent themes, select 30 in both number of features and select number of words.",
                target="target-info-third-section"),
        ]),
    dbc.Alert(
        "Not enough data to render plots, please adjust the filters",
        id="no-data-alert-word2vec",
        color="warning",
        style={"display": "none"},
    ),
    dbc.CardBody(
        [
            html.P(
                "Click on a word in the legend to explore that specific word",
                className="mb-0",
            ),
            html.P(
                "(affected by year and month selection)",
                style={"fontSize": 12, "font-weight": "lighter"},
            ),
            dcc.Loading(
                id="loading-word2vec-plot",
                children=[
                    dcc.Graph(id="hsd-word2vec")
                ],
                type="default"
            )
        ]
    ),
]

BODY = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Card(WORDCLOUD_PLOTS)),
            ], style={"marginTop": 30}
        ),
        dcc.Interval(id='interval', interval=10000, n_intervals=0),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Alert(
                            "Successfully added data to stopwords and exceptions list. Refresh dashboard after you see the alert message.",
                            id="modify-feature-msg",
                            color="info",
                            dismissable=True,
                            style={"display": "none"},
                        ),
                        dbc.Alert(
                            "Please refresh dashboard to see latest data.",
                            id="refresh-dash-msg",
                            color="success",
                            dismissable=True,
                            style={"display": "none"},
                        ),
                        dbc.Jumbotron(
                            [
                                html.H4(children="Modify feature list (Enter comma separated words)",
                                        className="display-5",
                                        style={'display': 'inline-block', 'margin-right': '5px'}),
                                html.I(className="fas fa-question-circle fa-lg", id="target-info-second-section",
                                       style={'float': 'right', 'margin': '5px'}),
                                dbc.Tooltip(
                                    "This section can be used to tune the results. Unwanted features can be removed using 'Enter stopwords' option, start typing the word you wish to remove and there will be suggestions to select words. Important keywords can also be added using 'Enter features to be selected' option",
                                    target="target-info-second-section"),
                                html.Hr(className="my-2"),
                                html.P(
                                    "(Note: Please modify the features set carefully. Words once added can be deleted "
                                    "manually from '" + stopwords_exceptions_filename + "')",
                                    style={"fontSize": 12, "font-weight": "lighter"},
                                ),
                                html.Div(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(html.P("Enter stopwords: ", className="lead"), md=4),
                                                dbc.Col([
                                                    dcc.Dropdown(
                                                        id="id-stopwords",
                                                        multi=True,
                                                        placeholder="Example: server, bugeco, sprsp",
                                                        options=[
                                                            {"label": word, "value": word}  # complete this part
                                                            for word in vocab
                                                        ],
                                                    ),
                                                ], md=8)
                                            ], style={"margin-top": 20}
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(html.Label("Enter features to be selected: ", className="lead"),
                                                        md=4),
                                                dbc.Col(
                                                    dcc.Input(
                                                        id="id-important-features",
                                                        type="text",
                                                        placeholder="Example: snr, upihp, rtl",
                                                        style={"width": "inherit"}
                                                    ),
                                                    md=8
                                                )
                                            ]
                                        )
                                    ]
                                ),
                                html.Button('Submit',
                                            id='modify-feature-button',
                                            type='submit',
                                            n_clicks=0,
                                            className='mt-2')
                            ]
                        )
                    ], md=12
                ),
            ], style={"marginTop": 50}
        ),
        dbc.Row(
            [
                dbc.Col(LEFT_CARD, md=4, align="center"),
                dbc.Col(
                    [
                        dbc.Card(WORD2VEC_PLOTS)
                    ],
                    md=8
                )
            ],
            style={"marginTop": 35}
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Alert(
                            "Not enough data to display in table, please adjust the filters",
                            id="no-data-alert-datatable",
                            color="warning",
                            style={"display": "none"},
                        ),
                        dbc.Alert(
                            "Successfully exported data to '" + os.path.join(root_path, tableDataFile),
                            id="export-msg",
                            color="success",
                            dismissable=True,
                            style={"display": "none"},
                        ),
                        html.Div(
                            [
                                dcc.Loading(
                                    id="loading-datatable",
                                    children=[
                                        html.Button('Export',
                                                    id='table-button',
                                                    type='submit',
                                                    n_clicks=0,
                                                    style={
                                                        'margin-bottom': '2px'
                                                    }),
                                        dash_table.DataTable(
                                            id="datatable-interactivity",
                                            columns=[
                                                {'name': 'Feature', 'id': 'feature'},
                                                {'name': 'Cluster of similar words', 'id': 'similar-words'},
                                                {'name': 'HSD IDs', 'id': 'HSD-IDS'}
                                            ],
                                            style_header={
                                                'text-align': 'left'
                                            },
                                            style_cell={
                                                'text-align': 'left'
                                            },
                                            style_table={
                                                'border': '1px solid lightgrey',
                                                'overflow': 'auto'
                                            },
                                            editable=False,
                                            row_selectable="multi",
                                            row_deletable=False,
                                            selected_rows=[],
                                            selected_row_ids=[],
                                            page_action="native",
                                            page_current=0,
                                            page_size=10,
                                            # style_table={"width": "97%", "margin": 15}
                                        )
                                    ],
                                )
                            ],
                            id="datatable",
                            style={"display": "none"},
                        )
                    ],
                    md=12,
                    align="center"
                )
            ],
            style={"marginTop": 10}
        )
    ],
    className="mt-12 mb-4",
    style={
        "max-width": "1240px"
    }
)

server = Flask(__name__)
app = dash.Dash(__name__,
                server=server,
                external_stylesheets=[dbc.themes.BOOTSTRAP,
                                      "https://use.fontawesome.com/releases/v5.10.2/css/all.css"],
                meta_tags=[
                    {
                        "name": "viewport",
                        "content": "width=device-width"
                    }
                ])
app.layout = html.Div(children=[NAVBAR, BODY])

"""
#
# Page layout and contents end here
#
#
# callbacks start here
#
"""


@app.callback(
    [
        Output("hidden-div", "style"),
    ],
    [
        Input("interval", "n_intervals")
    ]
)
def detect_file_update(n_intervals):
    """Callback for every n_interval to detect file changes

    Parameters
    ----------
    n_intervals: int
        n_intervals for the element with id interval

    Returns
    ----------
    Empty style for a hidden div
    """
    global df_tf_idf, lastmt

    if algorithm == "word2vec" and (os.stat(word2vec_model_path).st_atime > lastmt):
        global word2vec_model

        api_logger.info("Detected Changes to files, reload dash!")
        lastmt = os.stat(word2vec_model_path).st_atime
        df_tf_idf = pd.read_csv(tf_idf_filename)
        word2vec_model = _util.load_word2vec_model(word2vec_model_path)
        return [{"display": "none"}]

    elif algorithm == "co-occurrence" and (os.stat(co_occurrence_path).st_atime > lastmt):
        global co_occurrence

        api_logger.info("Detected Changes to files, reload dash!")
        lastmt = os.stat(co_occurrence_path).st_atime
        df_tf_idf = pd.read_csv(tf_idf_filename)
        co_occurrence = _util.load_co_occurrence_matrix(co_occurrence_path)
        return [{"display": "none"}]

    return [{"display": "none"}]


@app.callback(
    [
        Output("hsd-wordcloud", "figure"),
        Output("hsd-frequency_figure", "figure"),
        Output("hsd-treemap", "figure"),
        Output("no-data-alert-wordcloud", "style"),
        Output("select-year-alert", "style"),
    ],
    [
        Input("tfidf-button", "n_clicks"),

    ],
    [
        State("number-features", "value"),
        State("year-dropdown", "value"),
        State("month-dropdown", "value")
    ]
)
def update_wordcloud_plot(n_clicks, n_features, year, month):
    """Callback when button 'tfidf-button' is clicked

    Parameters
    ----------
    n_clicks: int
        Number of clicks on 'tfidf-button'

    n_features: int
        Number of features dropdown value

    year: list
        Year(s) selected

    month: list
        month(s) selected

    Returns
    ----------
    figure: WordCloud figure
    figure: Frequency figure
    figure: TreeMap figure
    style: display if there is no data for wordcloud/frequency/TreeMap
    style: display if month is selected and year is not
    """
    global local_df

    if not n_features:
        n_features = 25

    api_logger.info("Parameters selected: n_features:- {0}".format(n_features))

    local_df = df_tf_idf
    if n_clicks:
        if month and not year:
            return [{}, {}, {}, {"display": "none"}, {"display": "block"}]

        if year or month:
            api_logger.info("Filtering Data by year: {0} and month: {1}".format(year, month))
            local_df = filter_dataframe(year, month)

        if local_df is None:
            return [{}, {}, {}, {"display": "block"}, {"display": "none"}]

    features = _util.get_features_list(local_df)  # getting features and frequencies normalized across feature space
    features = features[:n_features]  # only taking n number of features as selected, default: 25

    wordcloud_data = get_wordcloud_plotly(features)
    freq_plot_data = get_frequency_figure(features)
    treemap_data = get_treemap_figure(features)

    return [wordcloud_data, freq_plot_data, treemap_data, {"display": "none"}, {"display": "none"}]


@app.callback(
    [
        Output("hsd-word2vec", "figure"),
        Output("no-data-alert-word2vec", "style"),
    ],
    [
        Input("hsd-frequency_figure", "figure"),
        Input("wordvec-button", "n_clicks"),

    ],
    [
        State("n-dropdown", "value"),
        State("top-n-dropdown", "value"),
        State("depth-dropdown", "value")
    ]
)
def update_word2vec_plot(figure, n_clicks, n_words, top_n, depth):
    """Callback when button 'wordvec-button' is clicked or frequency figure is changed

    Parameters
    ----------
    figure: dict
        Figure object from frequency figure

    n_clicks: int
        Number of times 'wordvec-button' is clicked

    n_words: int
        Number of words to build context plot for

    top_n: int
        Number of similar words

    depth: int
        Depth parameter

    Returns
    ----------
    figure: plotly figure of words and context
    style: display if there is no data
    """
    if n_words is None:
        n_words = 25
    if top_n is None:
        top_n = 10
    if depth is None:
        depth = 2

    api_logger.info("Parameters selected: n_word:- {0}, top_n:- {1}, depth:- {2} ".format(n_words, top_n, depth))

    if not figure:
        return [{}, {"display": "block"}]

    word2vec_nodes = figure['data'][0]['y'][::-1]
    if not word2vec_nodes:
        return [{}, {"display": "block"}]

    word_clusters, embedding_clusters, labels = [], [], []
    fig = {}
    if algorithm == "word2vec":
        word_clusters, embedding_clusters, labels = _util.produce_word_clusters_from_word2vec(word2vec_model,
                                                                                              n=n_words,
                                                                                              n_similar=top_n,
                                                                                              depth=depth,
                                                                                              top_vocab_words=word2vec_nodes)

        fig = _util.tsne_plot_similar_words_plotly(labels, word_clusters, embedding_clusters)
        layout = go.Layout(
            {
                "xaxis": {
                    "showgrid": False,
                    "showticklabels": False,
                    "zeroline": False,
                    "automargin": True,
                    # "range": [-100, 250],
                },
                "yaxis": {
                    "showgrid": False,
                    "showticklabels": False,
                    "zeroline": False,
                    "automargin": True,
                    # "range": [-100, 450],
                },
                "height": 540,
                "margin": dict(t=10, b=10, l=0, r=0, pad=4),
                "hovermode": "closest",
            }
        )
        fig.layout = layout

    elif algorithm == "co-occurrence":
        word_clusters, embedding_clusters, labels = _util.produce_word_clusters_from_co_occurrence(co_occurrence,
                                                                                                   n=n_words,
                                                                                                   n_similar=top_n,
                                                                                                   depth=depth,
                                                                                                   top_vocab_words=word2vec_nodes)

        fig = _util.plot_co_occurrence_matrix(labels, word_clusters, embedding_clusters)
        fig.update_layout(dict1={"title": "", "height": 540, "margin": dict(t=10, b=10, l=0, r=0, pad=4)})

    global words, similar_words

    words, similar_words = labels, word_clusters
    hsd_ids = []
    for elem in words:
        if elem in local_df.columns:
            hsd_ids.append(local_df[local_df[elem] > 0].sort_values(by=[elem], ascending=False)['hsd_id'].tolist())
        else:
            hsd_ids.append([])

    _util.save_word_clusters_with_ids(word_clusters_path, labels, word_clusters, hsd_ids)

    return [fig, {"display": "none"}]


@app.callback(
    [
        Output("datatable-interactivity", "data"),
        Output("datatable", "style"),
        Output("no-data-alert-datatable", "style"),
    ],
    [
        Input("hsd-word2vec", "figure")
    ]
)
def update_datatable(figure):
    """Callback when 'hsd-word2vec' is changed

    Parameters
    ----------
    figure: dict
        Figure object from frequency figure

    Returns
    ----------
    data: data to be populated in a table
    style: display if there is data to display
    style: display if no data
    """
    api_logger.info("loading data table")

    if not figure:
        return [None, {"display": "none"}, {"display": "block"}]

    if words:
        hsd_ids = []
        for elem in words:
            hsd_ids.append(local_df[local_df[elem] > 0].sort_values(by=[elem], ascending=False)['hsd_id'].tolist()
                           if elem in local_df.columns else [])

        df_datatable = pd.DataFrame(
            list(zip(words, [', '.join(i) for i in similar_words], [', '.join(map(str, i)) for i in hsd_ids])),
            columns=['feature', 'similar-words', 'HSD-IDS'])
        return [df_datatable.to_dict('records'), {"display": "block"}, {"display": "none"}]

    return [None, {"display": "none"}, {"display": "block"}]


@app.callback(
    [
        Output("export-msg", "style"),
    ],
    [
        Input("table-button", "n_clicks"),
    ],
    [
        State("datatable-interactivity", "data"),
        State("datatable-interactivity", "selected_rows")
    ]
)
def export_selected(n_clicks, rows, selected_row_indices):
    """Callback when button 'table-button' is clicked

    Parameters
    ----------
    n_clicks: int
        Number of times Export button is clicked

    rows: dict
        Data from datatable

    selected_row_indices: list
        Iterator of selected row indices

    Returns
    ----------
    style: display if data is exported
    """
    if n_clicks:
        api_logger.info("Exporting table data to csv")
        df_table = pd.DataFrame(rows)

        if selected_row_indices:
            df_table = df_table.loc[selected_row_indices]

        df_table.to_csv(tableDataFile, index=False)
        api_logger.info("Completed exporting table data to csv")
        return [{"display": "block"}]

    return [{"display": "none"}]


@app.callback(
    [
        Output("modify-feature-msg", "style"),
    ],
    [
        Input("modify-feature-button", "n_clicks"),
    ],
    [
        State("id-stopwords", "value"),
        State("id-important-features", "value")
    ]
)
def update_features(n_clicks, custom_stopwords, exceptions):
    """Callback when button 'modify-feature-button' is clicked

    Parameters
    ----------
    n_clicks: int
        Number of times 'modify-feature-button' button is clicked

    custom_stopwords: list
        Stopwords selected by user

    exceptions: str
        Exception words entered by user

    Returns
    ----------
    style: display if stopwords/exception words are added
    """
    if n_clicks and (custom_stopwords or exceptions):  # check if either of input is not empty
        api_logger.info("Updating stopwords and exception list")
        with open(stopwords_exceptions_filename, mode='r+') as json_file:
            data = json.load(json_file)

            if custom_stopwords:
                data['custom_stopwords'].extend(custom_stopwords)

            if exceptions:
                exceptions = list(filter(None, exceptions.split(',')))  # removing empty tokens
                data['exceptions'].extend(
                    [i.strip() for i in exceptions if i.strip() not in data[
                        'exceptions']])  # removing extra leading and trailing spaces and adding it to json

            json_file.seek(0)
            json.dump(data, json_file)
            json_file.truncate()

        api_logger.info("Stopwords and Exceptions: {0}".format(data))
        return [{"display": "block"}]

    return [{"display": "none"}]


@app.callback(
    [
        Output("refresh-dash-msg", "style")
    ],
    [
        Input("modify-feature-msg", "style"),
    ]
)
def run_main(input_style):
    """Trigger HSD Mining Analysis again to re-train model

    Parameters
    ----------
    input_style: dict
        Updated style if feature list was modified i.e. stopwords/exceptions were added.

    Returns
    ----------
    style: display if model re-trained successfully and dashboard needs to be refreshed.
    """
    if input_style['display'] == 'block':
        api_logger.info("Re-running main.py")
        hsd_mining = HsdMiningAnalysis(raw_data=raw_data_file, dash=True, algorithm=algorithm)
        hsd_mining.run_hsd_mining()
        api_logger.info("Completed running main.py")
        return [{"display": "block"}]

    return [{"display": "none"}]


# # @app.callback(
# #     [
#
#     ],
#     [
#         Input("hsd-word2vec", "clickData"),
#     ]
# )
# def upadate_data_table(click_event):
#     print(click_event)


"""
#
# Callbacks end here
#
# supporting functions start here
#
"""


def filter_dataframe(year, month):
    """Trigger HSD Mining Analysis again to re-train model

    Parameters
    ----------
    year: list
        Year(s) to filter results

    month: list
        Month(s) to filter results

    Returns
    ----------
    DataFrame with filtered data
    """
    filtered_df = df_tf_idf
    if year and month:
        filtered_df = df_tf_idf.loc[(df_tf_idf['Year'].isin(year)) & (df_tf_idf['Month'].isin(month))]
    elif year is not None:
        filtered_df = df_tf_idf.loc[df_tf_idf['Year'].isin(year)]

    return None if filtered_df.empty else filtered_df


def get_wordcloud_plotly(features):
    """Build WordCloud using plotly

    Parameters
    ----------
    features: DataFrame
        words and their frequencies

    Returns
    ----------
    WordCloud figure using plotly
    """
    api_logger.info("Loading wordcloud figure")
    wordcloud = WordCloud(collocations=False,
                          width=1600, height=1600,
                          background_color='white',
                          max_words=100,
                          random_state=42
                          )
    wordcloud.generate_from_frequencies(features)

    word_list, freq_list, fontsize_list, position_list, orientation_list, color_list = [], [], [], [], [], []

    for (word, freq), fontsize, position, orientation, color in wordcloud.layout_[:]:
        word_list.append(word)
        freq_list.append(freq)
        fontsize_list.append(fontsize)
        position_list.append(position)
        orientation_list.append(orientation)
        color_list.append(color)

    scatter_points = list(zip(*position_list))
    new_freq_list = [i * 300 for i in freq_list]

    trace = go.Scatter(
        x=list(scatter_points[0]),
        y=list(scatter_points[1]),
        textfont=dict(size=new_freq_list, color=color_list),
        hoverinfo="text",
        textposition="top center",
        hovertext=["{0} - {1}".format(w, f) for w, f in zip(word_list, freq_list)],
        mode="text",
        text=word_list,
    )

    layout = go.Layout(
        {
            "xaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                # "range": [-100, 250],
            },
            "yaxis": {
                "showgrid": False,
                "showticklabels": False,
                "zeroline": False,
                "automargin": True,
                # "range": [-100, 450],
            },
            "margin": dict(t=10, b=10, l=5, r=5, pad=4),
            "hovermode": "closest",
        }
    )

    wordcloud_figure = {
        "data": [trace],
        "layout": layout,
    }

    return wordcloud_figure


def get_frequency_figure(features):
    """Build Frequency figure using plotly

    Parameters
    ----------
    features: DataFrame
        words and their frequencies

    Returns
    ----------
    Frequency figure using plotly
    """
    api_logger.info("Loading frequency figure")

    word_list_top, freq_list_top = features.index.values, features.values

    freq_figure_data = {
        "data": [
            {
                "y": word_list_top[::-1],
                "x": freq_list_top[::-1],
                "type": "bar",
                "name": "",
                "orientation": "h",
            }
        ],
        "layout": {
            "height": "550",
            "margin": dict(t=20, b=20, l=100, r=20, pad=4)
        },
    }
    return freq_figure_data


def get_treemap_figure(features):
    """Build TreeMap figure using plotly

    Parameters
    ----------
    features: DataFrame
        words and their frequencies

    Returns
    ----------
    TreeMap figure using plotly
    """
    api_logger.info("Loading treemap figure")
    word_list_top, freq_list_top = features.index.values, features.values

    treemap_data = go.Treemap(
        labels=word_list_top, parents=[""] * len(word_list_top), values=freq_list_top
    )
    treemap_layout = go.Layout(
        {
            "margin": dict(t=10, b=10, l=5, r=5, pad=4)
        }
    )
    treemap_figure = {
        "data": [treemap_data],
        "layout": treemap_layout
    }
    return treemap_figure


"""
#
# supporting functions end here
#
"""

# if __name__ == "__main__":
#     app.run_server(debug=True)

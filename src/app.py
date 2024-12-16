import dash
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from dash import Input, Output, dcc, html
from dash.dependencies import ALL, State

from myfunc import (
    prepare_recommendation_system,
    myIBCF,
    get_recommended_movies,
    get_displayed_movies
)

# Initialize the Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.YETI,
        dbc.icons.BOOTSTRAP
    ],
    suppress_callback_exceptions=True
)
server = app.server

# Load and prepare data
ratings, movies, S_top30, top_10_popular = prepare_recommendation_system()

# Define styles for the sidebar and content
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "18rem",
    "padding": "2rem 1rem",
    "backgroundColor": "#f8f9fa",
}

CONTENT_STYLE = {
    "marginLeft": "20rem",
    "marginRight": "2rem",
    "padding": "2rem 1rem",
}

# Define the sidebar layout
sidebar = html.Div(
    [
        html.H3("Movie Recommender", className="display-6"),
        html.Hr(),
        dbc.Nav(
            [
                dbc.NavLink("System I - Popularity", href="/", active="exact"),
                dbc.NavLink("System II - Collaborative", href="/system-2", active="exact"),
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

# Define the main content layout
content = html.Div(id="page-content", style=CONTENT_STYLE)

# Set the app layout with a Store component and hidden Divs for scrolling
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    content,
    dcc.Store(id='recommended-movies-store'),  # Data Store for recommendations
    # html.Div(id='scroll-trigger', style={'display': 'none'}), 
    html.Div(id='scroll-action', style={'display': 'none'})     # Hidden action Div for client-side callback
])

### ==========================
###     Helper Functions
### ==========================

def get_movie_card(movie, with_rating=False):
    """
    Generate a movie card component.
    
    Inputs:
    - movie: Series containing movie details
    - with_rating: Boolean indicating if rating inputs should be included
    
    Output:
    - HTML Div containing the movie card
    """
    poster_url = movie['PosterURL'] if pd.notna(movie['PosterURL']) else "https://via.placeholder.com/300x450?text=No+Image"

    # Define the RadioItems options with star icons (Unicode stars)
    star_options = [
        {"label": "★☆☆☆☆", "value": "1"},
        {"label": "★★☆☆☆", "value": "2"},
        {"label": "★★★☆☆", "value": "3"},
        {"label": "★★★★☆", "value": "4"},
        {"label": "★★★★★", "value": "5"},
    ]

    return html.Div(
        dbc.Card(
            [
                dbc.CardImg(
                    src=poster_url,
                    top=True,
                    style={"height": "300px", "objectFit": "cover"},
                ),
                dbc.CardBody(
                    [
                        html.H6(movie['Title'], className="card-title text-center"),
                        # Include RadioItems if with_rating is True
                        dbc.RadioItems(
                            options=star_options,
                            id={"type": "movie_rating", "movie_id": movie['movieID']},
                            inline=True,
                            className="text-center",
                            inputClassName="m-1",
                            labelClassName="px-1",
                            labelStyle={"cursor": "pointer"},
                            persistence=True,
                            persistence_type="memory",
                        ) if with_rating else None,
                    ]
                ),
            ],
            className="h-100",
        ),
        className="col mb-4",
    )

def display_recommendations_with_posters(recommended_df):
    """
    Display recommended movies with posters in a responsive grid.
    
    Inputs:
    - recommended_df: DataFrame containing recommended movies with 'PrefixedMovieID', 'Title', 'PredictedRating', 'PosterURL'
    
    Output:
    - List of HTML Divs containing movie cards
    """
    cards = []
    for idx, movie in recommended_df.iterrows():
        cards.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardImg(
                            src=movie['PosterURL'],
                            top=True,
                            style={"height": "150px", "object-fit": "cover"},
                        ),
                        dbc.CardBody(
                            [
                                html.H6(movie['Title'], className="card-title text-center"),
                                html.P(
                                    f"Predicted Rating: {movie['PredictedRating']:.2f}" if not pd.isna(movie['PredictedRating']) else "Predicted Rating: N/A",
                                    className="card-text text-center"
                                ),
                            ]
                        ),
                    ],
                    className="h-100",
                ),
                className="col mb-4",
            )
        )
    return cards

### ==========================
### Dash Callbacks for System I and System II
### ==========================

@app.callback(
    Output("page-content", "children"),
    [Input("url", "pathname")]
)
def render_page_content(pathname):
    if pathname == "/":
        # System I: Popularity-Based Recommendations
        popular_movies = top_10_popular.head(10)
        popular_cards = [get_movie_card(movie, with_rating=False) for idx, movie in popular_movies.iterrows()]
        return html.Div(
            [
                html.H1("Top 10 Popular Movies"),
                html.Hr(),
                dbc.Row(
                    popular_cards,
                    className="row row-cols-1 row-cols-md-2 row-cols-lg-3 row-cols-xl-5 g-4",
                ),
            ]
        )
    elif pathname == "/system-2":
        # System II: Collaborative Filtering
        # Select a subset of movies for user to rate (e.g., 100 movies)
        sample_movies = get_displayed_movies(movies_df=movies, n=100)
        
        if sample_movies.empty:
            return html.Div(
                [
                    html.H3("No movies available to display."),
                    html.P("Please check your data or try again later."),
                ]
            )
        
        movie_cards = [get_movie_card(movie, with_rating=True) for idx, movie in sample_movies.iterrows()]
        return html.Div(
            [
                html.H1("Rate As Many Movies to Get Recommendations"),
                html.Hr(),
                dbc.Row(
                    movie_cards,
                    className="row row-cols-1 row-cols-md-2 row-cols-lg-3 row-cols-xl-5 g-4",
                ),
                dbc.Button("Get Recommendations", color="success", size="lg", id="button-recommend", className="mt-4"),
                html.Hr(),
                html.Div(id="recommendation-alert"),
                html.H1("Your Recommendations", id="your-recommendation", style={"display": "none"}),
                dbc.Spinner(children=[
                    dbc.Row(
                        [],
                        className="row row-cols-1 row-cols-md-2 row-cols-lg-3 row-cols-xl-5 g-4",
                        id="recommended-movies",
                    )
                ], type="circle"),
            ]
        )
    # Fallback for unknown paths
    return html.Div(
        [
            html.H1("404: Not found"),
            html.P("The pathname you entered was not recognized..."),
        ]
    )

@app.callback(
    Output("button-recommend", "disabled"),
    Input({"type": "movie_rating", "movie_id": ALL}, "value"),
)
def update_button_recommend_visibility(values):
    """
    Enable the 'Get Recommendations' button only if the user has rated at least one movie.
    
    Inputs:
    - values: List of ratings from RadioItems
    
    Output:
    - Boolean indicating whether the button should be disabled
    """
    # Enable the button if at least one rating is provided
    #return not any(filter(None, values))
    return False

@app.callback(
    [
        Output("recommended-movies", "children"),
        Output("your-recommendation", "style"),
        Output("recommendation-alert", "children")  
    ],
    [Input("button-recommend", "n_clicks")],
    [
        State({"type": "movie_rating", "movie_id": ALL}, "value"),
        State({"type": "movie_rating", "movie_id": ALL}, "id"),
    ],
    prevent_initial_call=True,
)
def on_getting_recommendations(n_clicks, ratings, ids):
    """
    Generate and display movie recommendations based on user ratings.
    
    Inputs:
    - n_clicks: Number of times the 'Get Recommendations' button was clicked
    - ratings: List of user ratings
    - ids: List of RadioItems IDs corresponding to movies
    
    Outputs:
    - List of recommended movie cards
    - Style dictionary to make the recommendations section visible
    - Update the scroll-trigger Div to initiate scrolling
    """
    # Extract user ratings into a dictionary
    rating_input = {
        'm' + str(id['movie_id']): int(rating) 
        for id, rating in zip(ids, ratings) 
        if rating is not None
    }

    # if not rating_input:
    #     # No ratings provided; return empty recommendations and keep the section hidden
    #     return [], {"display": "none"}, None
    if not rating_input:
        # default to top 10 popular movies
        recommended_movie_ids = top_10_popular['PrefixedMovieID'].tolist()[:10]
        recommended_df = get_recommended_movies(recommended_movie_ids, movies_df=movies)
        recommended_cards = [get_movie_card(movie, with_rating=False) for idx, movie in recommended_df.iterrows()]
        
        # Create an alert
        alert = dbc.Alert(
            "No ratings provided. Showing top 10 popular movies.",
            color="info",
            dismissable=True,
            is_open=True
        )
        
        return recommended_cards, {"display": "block"}, alert
    
    # Create a user ratings Series indexed by 'm1', 'm2', ..., 'mXXXX'
    user_ratings = pd.Series(data=np.nan, index=['m' + str(mid) for mid in movies['movieID']])
    for movie_id, rating in rating_input.items():
        user_ratings[movie_id] = rating

    # Generate recommendations using myIBCF
    # Ensure that myIBCF is correctly implemented and imported
    recommended_movie_ids = myIBCF(
        newuser=user_ratings,
        S_top30_df=S_top30,
        popularity_ranking_df=top_10_popular,
        n_recommend=10
    )

    # Get recommended movies DataFrame
    recommended_df = get_recommended_movies(recommended_movie_ids, movies_df=movies)

    # Generate movie cards for recommendations
    recommended_cards = [get_movie_card(movie, with_rating=False) for idx, movie in recommended_df.iterrows()]

    # Return the recommended cards, make the section visible, and trigger scrolling
    return recommended_cards, {"display": "block"}, None

## ==========================
## Client-Side Callback for Scrolling
## ==========================

app.clientside_callback(
    """
    function(trigger) {
        if (trigger) {
            var element = document.getElementById('your-recommendation');
            if (element) {
                element.scrollIntoView({behavior: 'smooth'});
                // Optionally, add a temporary highlight effect
                element.style.transition = "background-color 0.5s ease";
                element.style.backgroundColor = "#ffff99";  // Light yellow highlight
                setTimeout(function(){
                    element.style.backgroundColor = "";
                }, 1000);  // Remove highlight after 1 second
            }
        }
        return '';
    }
    """,
    Output('scroll-action', 'children'), 
    #Input('scroll-trigger', 'children')    
    Input('button-recommend', 'n_clicks')
)



if __name__ == "__main__":
    app.run_server(port=8080, debug=True)

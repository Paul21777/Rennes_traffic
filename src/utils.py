import plotly.express as px
import numpy as np



def create_figure(data):

    fig_map = px.scatter_mapbox(
            data,
            title="Traffic en temps réel",
            color="traffic",
            lat="lat",
            lon="lon",
            color_discrete_map={'freeFlow':'green', 'heavy':'orange', 'congested':'red'},
            zoom=10,
            height=500,
            mapbox_style="carto-positron"
    )

    return fig_map

def prediction_from_model(model, hour_to_predict):
    hour_to_predict = int(hour_to_predict) % 24
    input_pred = np.zeros(24)
    input_pred[hour_to_predict] = 1
    cat_predict = np.argmax(model.predict(np.array([input_pred])))
    return cat_predict
map_token = 'pk.eyJ1Ijoic2h0cmF1c3NhcnQiLCJhIjoiY2tqcDU2dW56MDVkNjJ6angydDF3NXVvbyJ9.nx2c5XzUH9MwIv4KcWVGLA'

import plotly.express as px
import plotly.graph_objects as go
'''
> Compare Two DataFrames w/ GPS coordinates (longitude & Latitude)
> Input DataFrames with lat,lon coordinates
> Uses Plotly Go & Mapbox Integration (token above)
'''
# compare two paths
def plot_geo(df1,df2):

    fig = go.Figure()

    if(df1 is not None):
        fig.add_trace(go.Scattermapbox(
                lat=data1['latDeg'],
                lon=data1['lngDeg'],
                mode='markers',
                name = 'PATH #1',
                marker=go.scattermapbox.Marker(
                    size=17,
                        color=lst_color[0],
                    opacity=0.7)))

    if(df2 is not None):
        fig.add_trace(go.Scattermapbox(
                lat=data2['latDeg'],
                lon=data2['lngDeg'],
                mode='markers',
                name = 'PATH #2',
                marker=go.scattermapbox.Marker(
                    size=17,
                    color=lst_color[4],
                    opacity=0.7)))

    # Plot Aesthetics
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        hovermode='closest',
        mapbox=dict(
            accesstoken=map_token,
            center=dict(lon=-122.23,lat=37.5),
            pitch=0,      # Up Down
            bearing=45,   # Sideways
            zoom=10.2))   # 
    fig.update_layout(margin={"r":10,"t":10,"l":10,"b":10},mapbox_style="light",height=600,width=1200)
    fig.update_layout(font=dict(family='sans-serif',size=16))
    fig.show()

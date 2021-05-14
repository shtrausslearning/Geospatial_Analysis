import plotly.express as px
import plotly.graph_objects as go

# compare two paths
def plot_geo(df1,df2):
    # Figure factory requires physical file with mapbox code
    px.set_mapbox_access_token(open(".\\kaggle\\working\\mapbox_token").read())

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

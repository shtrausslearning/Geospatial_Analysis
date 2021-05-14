map_token = 'pk.eyJ1Ijoic2h0cmF1c3NhcnQiLCJhIjoiY2tqcDU2dW56MDVkNjJ6angydDF3NXVvbyJ9.nx2c5XzUH9MwIv4KcWVGLA'

import plotly.figure_factory as ff
'''
Create a Discretised Hexbin Plot from Scattered Data
Input lat,lon coordinate data from DataFrame
'''
# Hexbin Plot
fig = ff.create_hexbin_mapbox(data_frame=df_perth,  # df
                              lat="LATITUDE",       # latitude
                              lon="LONGITUDE",      # longitude
                              nx_hexagon=200,       # hexagon discretisation
                              opacity=0.5,
                              min_count=1,          # minimum point in hexbin required 
                              labels={"color": "Point Count"},
                              range_color = [0,20], 
                              color_continuous_scale="mint",
                              show_original_data=True, # show point data
                              original_data_marker=dict(size=4, opacity=0.25,color='black'), # point data options
                              zoom = 11,
                              height=500)

# Plot Aesthetics
fig.update_layout(
    hovermode='closest',
    mapbox=dict(
        accesstoken=map_token,
        bearing=90,
        center=dict(lat=-32, lon=115.9),
        pitch=0,
        zoom=11
    )
)
fig.update_layout(margin={"r":0,"t":80,"l":0,"b":0},mapbox_style="light")
fig.show()

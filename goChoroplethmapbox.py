import plotly.graph_objects as go

''' 
Plot Choropleth Map using go.Choroplethmapbox
import via geoJSON format (Create shp_to_JSON_example.py)
'''

# define colourbar bounds from DataFrame
zmin = df_merged0['Percent_Unemployed'].min()
zmax = df_merged0['Percent_Unemployed'].max()

# Set the data for the map
data = go.Choroplethmapbox(
        geojson = lga_json,             #this is your GeoJSON
        locations = df_merged0.index,    #the index of this dataframe should align with the 'id' element in your geojson
        z = df_merged0['Percent_Unemployed'], #sets the color value
        text = df_merged0.LGA_NAME20,    #sets text for each shape
        colorbar=dict(thickness=20, ticklen=3, tickformat='%',outlinewidth=0), #adjusts the format of the colorbar
        marker_line_width=1, marker_opacity=0.8, colorscale="viridis", #adjust format of the plot
        zmin=zmin, zmax=zmax,           #sets min and max of the colourbar
        hovertemplate = "<b>%{text}</b><br>" + "%{z:.0%}<br>" + "<extra></extra>")  # sets the format of the text shown when you hover over each shape

# Mapbox included Layout Aesthetics
layout = go.Layout(mapbox1 = dict(center = dict(lat= -37 , lon=145),zoom = 6,accesstoken = map_token),                      
                   autosize=True,height=600,width=1000)
fig=go.Figure(data=data, layout=layout)
fig.update_layout(margin=dict(l=30, r=30, t=70, b=30))
fig.update_layout(font=dict(family='sans-serif',size=16),title=f"<b>VIC</b> | % UNEMPLOYED FEMALES | [20-24]")
fig.show()

import geopandas as gpd

''' 
> Read SHP File containing boundary data (w/ bounday identifier eg.LGA_CODE20) 
> Combine w/ statistics data w/ matching/overlapping boundary identifiers 
'''

# 1) Read SHP file containing boundary segment data
lga_gdf = gpd.read_file('/kaggle/input/LGA2020AUST/LGA_2020_AUST.shp')   # read SHP format Dataset
lga_gdf = lga_gdf[lga_gdf['STE_NAME16']=='Victoria']                     # Get Interested Data Subset 
lga_gdf['LGA_CODE20'] = lga_gdf['LGA_CODE20'].astype('str')              # Make sure all df use same type upon merging (a)

# 2) Loading the statistics data visualisation dataframe
df = pd.read_csv('/kaggle/input/ABSC16/ABS_C16_G43_LGA_09012021123023131.csv')  # read dataframe w/ stats
emp_df = df[(df['Age'] == '20 - 24') & (df['Sex'] == 'Females')]                # Get Interested Data Subset 
emp_df = emp_df[['LGA_2016','Labour force status', 'Region', 'Value']]          # Limit DataFrame Columns to those of interest
emp_df['LGA_2016'] = emp_df['LGA_2016'].astype('str')  # Make sure all df use same type upon merging (b)
emp_df = emp_df.pivot(index='LGA_2016', columns='Labour force status', values='Value').reset_index().rename_axis(None, axis=1) # Use PivotTable to rearrange data

# Create some new relations from available data
emp_df['Percent_Unemployed'] = emp_df['Total Unemployed']/(emp_df['Total Unemployed']+emp_df['Total Employed']) 

# Merge dataframes (1) & (2) & adjust merged dataframe
df_merged0 = pd.merge(lga_gdf[['LGA_CODE20', 'geometry', 'LGA_NAME20']], 
                      emp_df[['LGA_2016', 'Percent_Unemployed']], 
                      left_on='LGA_CODE20', right_on='LGA_2016', how='outer')
df_merged0 = df_merged0.dropna(subset=['Percent_Unemployed', 'LGA_CODE20', 'geometry'])
df_merged0.index = df_merged0.LGA_CODE20
df_merged0.info()

# Convert to JSON
df_merged0 = df_merged0.to_crs(epsg=4327)
lga_json = df_merged0.__geo_interface__

import pandas as pd


pst_data = pd.read_excel('Position_Data_0_0_image.xlsx')            
df = pd.DataFrame(pst_data)
new_x = df['loc_x'][861]
new_y = df['loc_y'][861]
new_z = df['loc_z'][861]
df['loc_x'] = df['loc_x'].apply(lambda x: -new_x + x)
df['loc_y'] = df['loc_y'].apply(lambda x: -new_y + x)
df['loc_z'] = df['loc_y'].apply(lambda x: -new_z + x)
df.to_excel('Origin_Moved.xlsx', sheet_name= 'CSS in m')
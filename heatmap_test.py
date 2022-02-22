import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

image_coordinates = pd.read_excel('Image.xlsx')            
df = pd.DataFrame(image_coordinates)
im = plt.imread('left000862.png')

result = df.pivot(index='v', columns='u', values='LAeq')

ax = sns.heatmap(result,cmap='rainbow', vmax=80, vmin=70)

ax.imshow(im)

plt.show()
import pandas as pd
import numpy as np
encoding='utf8'

print("Spotify's Top 50 Songs")

df = pd.read_csv(r'/Users/Home/PycharmProjects/DSDJ/Top Spotify/top50.csv') #, encoding='utf8')
#df.apply(lambda x: pd.lib.infer_dtype(x.values))
print(df)

# df = pd.read_csv('1459966468_324.csv', dtype={'text': unicode})





# with open('r', 'top50.csv') as file:
#    dataframe = pd.read_csv('/Users/Home/PycharmProjects/DSDJ/Top Spotify/top50.csv')
#    print(dataframe)




# with open('PC/Folder/file.txt', 'r') as file1:
#    FileContent = file1.read()
#    print(FileContent)


import pandas as pd
df = pd.read_csv('main/val.csv')
df.to_parquet('main/val.parquet',index=False, engine='fastparquet')
#print(pd.read_parquet("main/test1.parquet", engine='fastparquet'))
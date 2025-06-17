import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src.dataloader import Metadata
from matplotlib import rc
import pandas as pd
import logging
import os
rc('font',**{'family':'serif','serif':['Helvetica']})


metadataset = Metadata(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'allsup.xlsx')))
print(metadataset.__len__())
sample, idx, metadata = metadataset.__getitem__(1)
print(f'Index: {idx}, Subject path: {metadataset.files[idx]}, Metadata: {metadata}')
df = metadataset.metadata

logging.info(f"Dataframe shape: {df.shape}")
logging.info(f"Dataframe columns: {df.columns}")
logging.info(f"Dataframe head: {df.head()}")
logging.info(f"Dataframe info: {df.info()}")
logging.info(f"Dataframe describe: {df.describe()}")

plt.hist(df['age'])
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.grid()
plt.savefig('figures/age_distribution.png')
plt.show()


plt.figure(figsize=(15, 8))
columns_to_plot = ['sex', 'diag', 'hand']

for idx, col in enumerate(columns_to_plot):
    plt.subplot(1, 3, idx + 1)
    value_counts = df[col].value_counts()
    plt.pie(value_counts, labels=value_counts.index, autopct='%1.1f%%') 
    plt.title(f'Distribution of {col.capitalize()}') 

plt.tight_layout()
plt.savefig('figures/distribution.png') 
plt.show()


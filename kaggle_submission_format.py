import pandas as pd

df = pd.DataFrame(np.arange(label_vals.shape[0]), columns=['Id'])
df['Labels'] = label_vals
df.to_csv("a.csv', index=False)

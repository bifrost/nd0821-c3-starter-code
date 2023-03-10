# %%
import pandas as pd
import pandas_profiling

# %%
df = pd.read_csv('./data/census.csv')
# %%

# %%
df.describe()
# %%
profile = pandas_profiling.ProfileReport(df)
# %%
profile

# %%

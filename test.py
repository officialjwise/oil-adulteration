import pandas as pd
import numpy as np

df = pd.DataFrame(np.random.rand(5, 50), columns=[f'feature_{i}' for i in range(50)])
df.to_csv('test_features.csv', index=False)
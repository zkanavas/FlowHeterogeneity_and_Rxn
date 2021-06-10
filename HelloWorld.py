import numpy as np
import pandas as pd
a = np.array([[[1, 2, 3],
               [1, 2, 3],
               [1, 2, 3]],
              [[2, 2, 2],
               [2, 2, 2],
               [2, 2, 2]],
              [[3, 3, 3],
               [3, 3, 3],
               [3, 3, 3, ]]])
x, y, z = a.shape

out_arr = np.column_stack((np.repeat(np.arange(x), y*z),  # x-column
                           np.tile(np.repeat(np.arange(x), y), z),  # y-column
                           np.tile(np.tile(np.arange(x), y), z),  # z-column
                           a.reshape(x*y*z, -1)))  # velocity

out_df = pd.DataFrame(out_arr, columns=['x', 'y', 'z', 'u'])
print(out_df)

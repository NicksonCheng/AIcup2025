import pandas as pd
import numpy as np
import os

output_dir = "./"
submission_path = [ 
    '/home/syung/Project/AICUP/output_pr/class_vote1.csv',

    '/home/syung/Project/AICUP/output_pu/class_vote1.csv',
]



output = None
for path in submission_path:
    df_test = pd.read_csv(path)
    print(f'讀取 {path} 成功')
    print(df_test.shape)
    if output is None:
        output = df_test
    else:
        output = pd.concat([output, df_test], ignore_index=True, axis=0)
print(output.shape)

# 輸出

output_file = os.path.join(output_dir, "submission.csv")
output.to_csv(output_file, index=False)
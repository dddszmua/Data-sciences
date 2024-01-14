import pandas as pd
import jieba

import Utils

df = pd.read_csv('./low data.csv', encoding='utf-8').astype(str)

# Data preprocessing
df['类别'] = df['类别'].str.replace('_x0000_', '')
df['描述'] = df['描述'].str.replace('_x0000_', '')
df.to_csv('./low data.csv', encoding='utf-8', index=False)

# Analysis

groups = df.groupby('机型')

S1_df = groups.get_group('T2x')
S2_df = groups.get_group('Y33e')
S3_df = groups.get_group('Y77')

# Emotion analyze
t_df = Utils.getEmotion(S3_df)
Utils.getRate(t_df)

# Model training for prediction
#words = Utils.devide(XF_df)
#Utils.train(XF_df, words)

# WordCloud
Utils.transform(S3_df)
Utils.drawCloud('./comments.txt')

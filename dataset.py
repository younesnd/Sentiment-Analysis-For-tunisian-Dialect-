import pandas as pd 
df= pd.read_csv("path")
df["kfold"]= -1
df = df.sample(frac=1).reset_index(drop=True)
y = df.label.values
kf = model_selection.StratifiedKFold(n_splits=6)
for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
	df.loc[v_, 'kfold'] = f
df.to_csv("train_folds.csv", index=False)

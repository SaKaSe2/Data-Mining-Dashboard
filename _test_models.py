"""Test: buang sampel ambigu tengah, hanya Rendah vs Tinggi."""
from pipeline import load_and_preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import numpy as np
import os

csv = os.path.abspath(os.path.join('.', '..', 'climate_change_impact_on_agriculture_2024 (1).csv'))
print('Loading...')
df, log = load_and_preprocess(csv)

yield_col = 'Crop_Yield_MT_per_HA'

# Strategi: bottom 30% = Rendah, top 30% = Tinggi, buang 40% tengah
q_low = df[yield_col].quantile(0.30)
q_high = df[yield_col].quantile(0.70)
df_filtered = df[(df[yield_col] <= q_low) | (df[yield_col] >= q_high)].copy()
y = (df_filtered[yield_col] >= q_high).map({True: 'Tinggi', False: 'Rendah'})
print('Filtered shape:', df_filtered.shape, '| Distribution:', y.value_counts().to_dict())

nf = df_filtered.select_dtypes(include=np.number).drop(columns=[yield_col], errors='ignore').columns.tolist()
nf = [c for c in nf if not c.startswith('tfidf_')]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# NB
acc_nb = cross_val_score(GaussianNB(), df_filtered[nf], y, cv=cv, scoring='accuracy').mean()
print('NB  Acc:', round(acc_nb, 4))

# DT
dt = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
acc_dt = cross_val_score(dt, df_filtered[nf], y, cv=cv, scoring='accuracy').mean()
print('DT  Acc:', round(acc_dt, 4))

# Voting
vt = VotingClassifier(estimators=[('nb', GaussianNB()), ('dt', DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced'))], voting='soft')
acc_vt = cross_val_score(vt, df_filtered[nf], y, cv=cv, scoring='accuracy').mean()
print('VT  Acc:', round(acc_vt, 4))

# GB
gb = GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42)
acc_gb = cross_val_score(gb, df_filtered[nf], y, cv=cv, scoring='accuracy').mean()
print('GB  Acc:', round(acc_gb, 4))

print('DONE')

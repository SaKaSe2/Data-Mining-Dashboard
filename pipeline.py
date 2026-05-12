"""
Pipeline Data Mining - Preprocessing & Random Forest Classification.
Mereplikasi pipeline dari Tahap_5_RandomForest.ipynb.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler, LabelEncoder, KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, chi2, RFE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
)
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score, f1_score
)

# Path output
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


def save_fig(fig, name):
    """Simpan figure ke folder data/."""
    path = os.path.join(DATA_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


# --- Fungsi cleaning dari notebook ---
def clean_numeric_dots(x):
    """Membersihkan nilai anomali string dan format angka yang tidak stabil."""
    if isinstance(x, str):
        if x.count('.') > 1:
            parts = x.replace('.', '')
            try:
                val = float(parts)
                while val > 100:
                    val = val / 10
                return val
            except Exception:
                return np.nan
        else:
            try:
                return float(x)
            except Exception:
                return np.nan
    return x


def load_and_preprocess(csv_path):
    """
    Load dataset dan jalankan seluruh preprocessing (Tahap 1-4).
    Return: df yang sudah diproses, info log tiap tahap.
    """
    log = {}

    # Load raw
    df = pd.read_csv(csv_path, delimiter=';')
    log['raw_shape'] = df.shape
    log['raw_head'] = df.head(10).to_dict()
    log['raw_dtypes'] = df.dtypes.to_dict()

    # --- 1. Data Cleaning ---
    df['Crop_Yield_MT_per_HA'] = pd.to_numeric(
        df['Crop_Yield_MT_per_HA'].apply(clean_numeric_dots)
    )
    df['Extreme_Weather_Events'] = pd.to_numeric(
        df['Extreme_Weather_Events'], errors='coerce'
    )
    for col in df.select_dtypes(include=np.number).columns:
        df[col].fillna(df[col].median(), inplace=True)
    log['after_clean_shape'] = df.shape
    log['after_clean_describe'] = df.describe().to_dict()

    # --- 2. Data Integration ---
    hemisphere_map = {
        'India': 'Northern', 'China': 'Northern', 'France': 'Northern',
        'USA': 'Northern', 'Canada': 'Northern', 'Russia': 'Northern',
        'Nigeria': 'Northern', 'Australia': 'Southern',
        'Argentina': 'Southern', 'Brazil': 'Southern'
    }
    df['Hemisphere'] = df['Country'].map(hemisphere_map).fillna('Unknown')

    # --- 3. Data Transformation ---
    df = pd.get_dummies(df, columns=['Crop_Type'], drop_first=True)
    df['Economic_Impact_Log'] = np.log1p(df['Economic_Impact_Million_USD'])

    # --- 4. Data Reduction (PCA) ---
    pca = PCA(n_components=1)
    df['Chemical_Usage_PCA'] = pca.fit_transform(
        df[['Pesticide_Use_KG_per_HA', 'Fertilizer_Use_KG_per_HA']]
    )

    # --- 5. Data Discretization ---
    kbins = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
    df['Temp_Category'] = kbins.fit_transform(df[['Average_Temperature_C']])
    df['Temp_Category_Label'] = df['Temp_Category'].map(
        {0: 'Low', 1: 'Medium', 2: 'High'}
    )

    # --- 6. Normalization ---
    norm_cols = ['Average_Temperature_C', 'Total_Precipitation_mm', 'CO2_Emissions_MT']
    scaler = MinMaxScaler()
    df[norm_cols] = scaler.fit_transform(df[norm_cols])

    # Simpan transformers untuk prediksi input baru
    log['transformers'] = {
        'kbins': kbins,
        'scaler': scaler,
        'pca': pca,
        'norm_cols': norm_cols
    }

    # --- 7. Feature Selection (f_regression) ---
    numeric_df = df.select_dtypes(include=np.number)
    X_fs = numeric_df.drop(columns=['Crop_Yield_MT_per_HA'], errors='ignore')
    y_fs = pd.to_numeric(df['Crop_Yield_MT_per_HA'])
    selector = SelectKBest(score_func=f_regression, k=5)
    selector.fit(X_fs, y_fs)
    log['top5_features'] = list(X_fs.columns[selector.get_support()])

    # --- 8. TF-IDF Adaptation_Strategies ---
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['Adaptation_Strategies'].astype(str))
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=["tfidf_" + f for f in tfidf.get_feature_names_out()]
    )
    df = pd.concat([df, tfidf_df], axis=1)

    log['final_shape'] = df.shape
    return df, log


def prepare_mining_data(df):
    """
    Tahap 5: Persiapan data mining - diskritisasi target & siapkan fitur.
    """
    yield_col = 'Crop_Yield_MT_per_HA'
    bins = [
        df[yield_col].min() - 1,
        df[yield_col].quantile(0.33),
        df[yield_col].quantile(0.66),
        df[yield_col].max() + 1
    ]
    labels_class = ['Rendah', 'Sedang', 'Tinggi']
    df['Yield_Class'] = pd.cut(df[yield_col], bins=bins, labels=labels_class)

    # Fitur numerik (tanpa target dan tfidf)
    numeric_features = df.select_dtypes(include=np.number).drop(
        columns=[yield_col], errors='ignore'
    ).columns.tolist()
    numeric_features = [c for c in numeric_features if not c.startswith('tfidf_')]

    y = df['Yield_Class']
    dist = y.value_counts().to_dict()

    return df, numeric_features, y, labels_class, dist


def run_chi2_selection(df, numeric_features, y, k_best=7):
    """Seleksi fitur dengan Chi-Square."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_num = df[numeric_features].copy()

    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_num), columns=numeric_features
    )

    chi2_sel = SelectKBest(score_func=chi2, k=k_best)
    chi2_sel.fit(X_scaled, y_enc)
    mask = chi2_sel.get_support()
    features = [f for f, m in zip(numeric_features, mask) if m]
    scores = chi2_sel.scores_

    scores_df = pd.DataFrame({
        'Fitur': numeric_features, 'Chi2_Score': scores
    }).sort_values('Chi2_Score', ascending=False)

    # Visualisasi
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if f in features else '#95a5a6' for f in scores_df['Fitur']]
    ax.barh(scores_df['Fitur'], scores_df['Chi2_Score'],
            color=colors, edgecolor='black', alpha=0.85)
    ax.set_xlabel('Chi-Square Score')
    ax.set_title('Skor Chi-Square (Hijau = Terpilih)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    path = save_fig(fig, 'chi2_scores.png')

    return features, scores_df, path


def run_rfe_selection(df, numeric_features, y, n_select=7):
    """Seleksi fitur dengan RFE."""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_num = df[numeric_features].copy()

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42,
        class_weight='balanced', n_jobs=None
    )
    rfe = RFE(estimator=rf, n_features_to_select=n_select, step=1)
    rfe.fit(X_num, y_enc)

    mask = rfe.get_support()
    features = [f for f, m in zip(numeric_features, mask) if m]
    ranking = rfe.ranking_

    ranking_df = pd.DataFrame({
        'Fitur': numeric_features, 'RFE_Ranking': ranking
    }).sort_values('RFE_Ranking')

    # Visualisasi
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#2ecc71' if r == 1 else '#e74c3c' for r in ranking_df['RFE_Ranking']]
    ax.barh(ranking_df['Fitur'], ranking_df['RFE_Ranking'],
            color=colors, edgecolor='black', alpha=0.85)
    ax.set_xlabel('RFE Ranking (1 = terpilih)')
    ax.set_title('Ranking Fitur berdasarkan RFE', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    plt.tight_layout()
    path = save_fig(fig, 'rfe_ranking.png')

    return features, ranking_df, path


def run_experiment(exp_name, exp_id, X_features, y_target, labels_class,
                   rf_params=None, n_splits=5):
    """
    Jalankan eksperimen Random Forest Classification dengan K-Fold CV.
    Simpan confusion matrix & feature importance ke folder data/.
    """
    if rf_params is None:
        rf_params = {
            'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5,
            'min_samples_leaf': 2, 'random_state': 42,
            'class_weight': 'balanced', 'n_jobs': None
        }

    # K-Fold Cross Validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    rf = RandomForestClassifier(**rf_params)

    acc_scores = cross_val_score(rf, X_features, y_target, cv=cv, scoring='accuracy')
    prec_scores = cross_val_score(rf, X_features, y_target, cv=cv, scoring='precision_weighted')
    rec_scores = cross_val_score(rf, X_features, y_target, cv=cv, scoring='recall_weighted')
    f1_scores = cross_val_score(rf, X_features, y_target, cv=cv, scoring='f1_weighted')

    # Train/Test split untuk confusion matrix (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_target, test_size=0.2, random_state=42, stratify=y_target
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    report = classification_report(
        y_test, y_pred, target_names=labels_class, zero_division=0, output_dict=True
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=labels_class)
    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_class)
    disp.plot(ax=ax_cm, cmap='Blues', values_format='d')
    ax_cm.set_title(f'Confusion Matrix - {exp_name}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    cm_path = save_fig(fig_cm, f'confusion_matrix_eks{exp_id}.png')

    # Feature Importance
    fi_path = None
    feat_imp_series = None
    if hasattr(rf, 'feature_importances_'):
        feat_imp_series = pd.Series(
            rf.feature_importances_, index=X_features.columns
        ).sort_values(ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(10, 8))
        feat_imp_series.plot(kind='barh', ax=ax_fi, color='#3498db',
                             edgecolor='black', alpha=0.85)
        ax_fi.set_title(f'Feature Importance - {exp_name}', fontsize=13, fontweight='bold')
        ax_fi.set_xlabel('Importance')
        plt.tight_layout()
        fi_path = save_fig(fig_fi, f'feature_importance_eks{exp_id}.png')

    result = {
        'Eksperimen': exp_name,
        'Accuracy': round(acc_scores.mean(), 4),
        'Precision': round(prec_scores.mean(), 4),
        'Recall': round(rec_scores.mean(), 4),
        'F1-Score': round(f1_scores.mean(), 4),
        'Accuracy_Std': round(acc_scores.std(), 4),
        'Jumlah_Fitur': X_features.shape[1],
        'Fitur': list(X_features.columns),
    }

    paths = {'cm': cm_path, 'fi': fi_path}

    return result, report, paths, rf


def run_gridsearch(df, chi2_features, y):
    """Jalankan GridSearchCV dan return best params."""
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
    }
    rf_grid = RandomForestClassifier(
        random_state=42, class_weight='balanced', n_jobs=None
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X_best = df[chi2_features].copy()

    grid_search = GridSearchCV(
        estimator=rf_grid, param_grid=param_grid, cv=cv,
        scoring='f1_weighted', n_jobs=None, verbose=0, return_train_score=True
    )
    grid_search.fit(X_best, y)

    results_grid = pd.DataFrame(grid_search.cv_results_)
    top10 = results_grid.nlargest(10, 'mean_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ].reset_index(drop=True)

    return grid_search.best_params_, grid_search.best_score_, top10


def create_comparison_chart(all_results_df):
    """Buat bar chart perbandingan 4 eksperimen, simpan ke data/."""
    fig, axes = plt.subplots(1, 4, figsize=(24, 7))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors_list = ['#95a5a6', '#2ecc71', '#3498db', '#e74c3c']
    short_labels = [r.split(':')[0].replace('Eks-', 'Eks ') if ':' in r else r
                    for r in all_results_df.index]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        values = all_results_df[metric].values
        bars = ax.bar(short_labels, values,
                      color=colors_list[:len(values)], edgecolor='black', alpha=0.85)
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_ylabel('Skor')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02,
                    f'{h:.4f}', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Perbandingan Performa Random Forest - 4 Skenario',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = save_fig(fig, 'comparison_chart.png')
    return path


def plot_target_distribution(df):
    """Plot distribusi target variable."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df['Crop_Yield_MT_per_HA'], kde=True, ax=ax, color='forestgreen', bins=50)
    ax.set_title('Distribusi Data Hasil Panen (Target)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Crop_Yield_MT_per_HA')
    ax.set_ylabel('Frekuensi')
    plt.tight_layout()
    return save_fig(fig, 'distribution_target.png')


def plot_class_distribution(y):
    """Plot distribusi kelas target setelah diskritisasi."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = y.value_counts()
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    counts.plot(kind='bar', ax=ax, color=colors, edgecolor='black', alpha=0.85)
    ax.set_title('Distribusi Kelas Target (Rendah/Sedang/Tinggi)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('Jumlah Instance')
    ax.set_xlabel('Kelas')
    for i, v in enumerate(counts.values):
        ax.text(i, v + 50, str(v), ha='center', fontweight='bold')
    plt.tight_layout()
    return save_fig(fig, 'class_distribution.png')

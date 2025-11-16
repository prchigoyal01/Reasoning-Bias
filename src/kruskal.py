import pandas as pd
import numpy as np
from scipy.stats import kruskal
import scikit_posthocs as sp
import warnings
warnings.filterwarnings("ignore")


# 1. LOAD DATA
CSV_PATH = "kruskal_input_predictions.csv"

df = pd.read_csv(CSV_PATH)

# Standardize strings
for col in ["model", "language", "category", "context", "category_type"]:
    df[col] = df[col].astype(str).str.lower().str.strip()

def fmt(items):
    return ",".join(items)

# COMPUTE PER-ITEM BIAS SCORES
def compute_bias_amb(row):
    """Ambiguous context: biasAmb = 1 if pred=0 (counter-stereo), 0 if pred=2 (stereo), NaN otherwise."""
    if row["context"] != "ambig":
        return np.nan
    if row["pred"] == 0:
        return 1
    elif row["pred"] == 2:
        return 0
    else:  # pred == 1 (unknown)
        return np.nan


def compute_bias_dis(row):
    """Disambiguated context: biasDis = 1 if pred==gold, 0 otherwise. Unknown gold entries excluded."""
    if row["context"] != "disambig":
        return np.nan
    if row["gold"] == 1:
        return np.nan
    return 1 if row["pred"] == row["gold"] else 0


df["bias_amb"] = df.apply(compute_bias_amb, axis=1)
df["bias_dis"] = df.apply(compute_bias_dis, axis=1)


# FILTER MODELS FOR AMBIG ACCURACY (Wu et al.)
def ambiguous_accuracy(d):
    """Percent of ambiguous samples where model chooses gold=1"""
    amb = d[d["context"] == "ambig"]
    if len(amb) == 0:
        return 0
    correct = (amb["pred"] == 1).mean()
    return correct

model_ambig_acc = df.groupby("model").apply(ambiguous_accuracy).sort_values(ascending=False)
print("\n=== AMBIGUOUS ACCURACY (for filtering) ===")
print(model_ambig_acc)

# Keep only models above chance (33%)
valid_models = model_ambig_acc[model_ambig_acc >= 0.33].index.tolist()

df_valid = df[df["model"].isin(valid_models)].copy()
print("\nModels retained for ambiguous bias testing:", valid_models)


# KRUSKAL-WALLIS HELPERS
def run_kw_test(groups_dict):
    """Run Kruskal–Wallis and return H, p."""
    vectors = [v.dropna().values for v in groups_dict.values()]
    if any(len(v) == 0 for v in vectors):
        return None, None  # Not enough data
    H, p = kruskal(*vectors)
    return H, p

def dunn_posthoc(df_local, value_col, group_col):
    """Run Dunn with Bonferroni correction."""
    try:
        result = sp.posthoc_dunn(
            df_local, val_col=value_col, group_col=group_col, p_adjust="bonferroni"
        )
        return result
    except:
        return None

def valid_for_kruskal(groups_dict):
    """Check if at least two groups have >= 2 non-NaN values."""
    counts = [len(g.dropna()) for g in groups_dict.values()]
    non_empty = sum(c >= 2 for c in counts)
    return non_empty >= 2



results = []

languages = df_valid["language"].unique()
models = df_valid["model"].unique()
categories = df_valid["category"].unique()


# CROSS-LINGUAL TESTS
for model in models:
    for metric in ["bias_amb", "bias_dis"]:
        groups = {
            lang: df_valid[(df_valid.model == model) & (df_valid.language == lang)][metric]
            for lang in languages
        }
        if valid_for_kruskal(groups):
            H, p = run_kw_test(groups)
            lang_list = fmt(list(groups.keys()))
            results.append([
                "cross-lingual",
                model,
                metric,
                H,
                p,
                lang_list  
            ])



# CROSS-CATEGORY TESTS
for model in models:
    for metric in ["bias_amb", "bias_dis"]:

        # Pool all languages together
        groups = {
            cat: df_valid[(df_valid.model == model) &
                          (df_valid.category == cat)][metric].dropna()
            for cat in categories
        }

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if len(v) > 0}

        if valid_for_kruskal(groups):
            H, p = run_kw_test(groups)
            cat_list = ",".join(list(groups.keys()))
            results.append([
                "cross-category",
                model,
                metric,
                H,
                p,
                cat_list     # record which categories were tested
            ])

results_df = pd.DataFrame(results, columns=["test_type", "group", "metric", "H", "p", "category"])
results_df.to_csv("kruskal_results.csv", index=False)

print("\n=== SAVED: kruskal_results.csv ===")

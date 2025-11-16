import matplotlib.pyplot as plt
import numpy as np

# ================================================================
# CATEGORY ORDER (NO CONTROL ONES)
# ================================================================
categories = ["Age", "Disability", "Gender", "Physical", "SES", "Sexual"]
y_pos = np.arange(len(categories))

# ================================================================
# MODEL × MODE COLOR MAP
# ================================================================
model_colors = {
    "llama2_short": "blue",
    "llama31_short": "red",
    "llama31_cot": "green",
    "deepseek_reasoning": "purple",
}

# ================================================================
# BIASAmb DATA (NON-CONTROL ONLY)
# ================================================================
bias_data = {
    "EN": {
        "llama2_short": {
            "Age":0.835, "Disability":0.865, "Gender":0.821,
            "Physical":0.872, "SES":0.849, "Sexual":0.839
        },
        "llama31_short": {
            "Age":0.720, "Disability":0.655, "Gender":0.641,
            "Physical":0.697, "SES":0.633, "Sexual":0.694
        },
        "llama31_cot": {
            "Age":0.818, "Disability":0.824, "Gender":0.832,
            "Physical":0.786, "SES":0.789, "Sexual":0.839
        },
        "deepseek_reasoning": {
            "Age":0.723, "Disability":0.735, "Gender":0.674,
            "Physical":0.714, "SES":0.695, "Sexual":0.694
        }
    },

    "ES": {
        "llama2_short": {
            "Age":0.862, "Disability":0.831, "Gender":0.837,
            "Physical":0.903, "SES":0.878, "Sexual":0.887
        },
        "llama31_short": {
            "Age":0.729, "Disability":0.687, "Gender":0.690,
            "Physical":0.675, "SES":0.649, "Sexual":0.710
        },
        "llama31_cot": {
            "Age":0.730, "Disability":0.712, "Gender":0.712,
            "Physical":0.689, "SES":0.722, "Sexual":0.742
        },
        "deepseek_reasoning": {
            "Age":0.634, "Disability":0.600, "Gender":0.712,
            "Physical":0.642, "SES":0.598, "Sexual":0.645
        }
    },

    "TR": {
        "llama2_short": {
            "Age":0.976, "Disability":0.986, "Gender":0.984,
            "Physical":0.972, "SES":0.982, "Sexual":0.952
        },
        "llama31_short": {
            "Age":0.759, "Disability":0.739, "Gender":0.763,
            "Physical":0.804, "SES":0.698, "Sexual":0.613
        },
        "llama31_cot": {
            "Age":0.728, "Disability":0.677, "Gender":0.726,
            "Physical":0.688, "SES":0.702, "Sexual":0.677
        },
        "deepseek_reasoning": {
            "Age":0.670, "Disability":0.729, "Gender":0.694,
            "Physical":0.749, "SES":0.713, "Sexual":0.710
        }
    },

    "NL": {
        "llama2_short": {
            "Age":0.878, "Disability":0.858, "Gender":0.853,
            "Physical":0.900, "SES":0.876, "Sexual":0.855
        },
        "llama31_short": {
            "Age":0.641, "Disability":0.653, "Gender":0.641,
            "Physical":0.681, "SES":0.576, "Sexual":0.597
        },
        "llama31_cot": {
            "Age":0.711, "Disability":0.715, "Gender":0.663,
            "Physical":0.739, "SES":0.703, "Sexual":0.710
        },
        "deepseek_reasoning": {
            "Age":0.679, "Disability":0.680, "Gender":0.625,
            "Physical":0.639, "SES":0.650, "Sexual":0.597
        }
    }
}

GLOBAL_MIN = 0.476
GLOBAL_MAX = 1.086


# ================================================================
# PLOT FUNCTION FOR ONE LANGUAGE
# ================================================================
def plot_language(lang):
    plt.figure(figsize=(10, 7))

    for model_key, model_data in bias_data[lang].items():
        color = model_colors[model_key]

        for i, c in enumerate(categories):
            value = model_data[c]
            significant = abs(value) > 0.75

            if significant:
                plt.text(value, i, "⋆", fontsize=18, color=color,
                         ha="center", va="center")
            else:
                plt.plot(value, i, "o", color=color, markersize=8)

    plt.yticks(y_pos, categories, fontsize=11)
    plt.axvline(0, linestyle="--", color="black")

    plt.title(f"BiasAmb Across Models — {lang}", fontsize=16)
    plt.xlabel("BiasAmb Score")

    # same scale across all languages
    plt.xlim(GLOBAL_MIN, GLOBAL_MAX)

    # custom legend
    handles = [
        plt.Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=col, markersize=10, label=name)
        for name, col in model_colors.items()
    ]
    plt.legend(handles=handles, title="Model × Eval Mode")

    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()




# ================================================================
# GENERATE ALL 4 FIGURES
# ================================================================
plot_language("EN")
plot_language("ES")
plot_language("TR")
plot_language("NL")

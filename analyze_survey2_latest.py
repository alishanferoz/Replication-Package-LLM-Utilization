import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import re
import textwrap

# --- USER CONTROL: VISUAL SETTINGS ---
BAR_WIDTH = 0.6
BAR_SPACING = 1.4
BAR_LABEL_SIZE = 35
LABEL_WRAP_WIDTH = 12
Y_LABEL_FONT_SIZE = 18  # Font size for the question text (ticks)
X_TICK_FONT_SIZE = 18  # Font size for the numbers on the X-axis
Y_TITLE_FONT_SIZE = 22  # Font size for the Y-axis title (e.g., "Access Methods")

# --- MANUAL LABEL MAPPING ---
MANUAL_MAP = {
    "Via a standalone web-based chat interface": "Web-based chat interface",
    "As an IDE/editor plugin or extension": "IDE/editor plugin",
    "As a browser extension or add-on for coding tasks": "Browser add-on",
    "Through a command-line tool or terminal interface": "CLI or Terminal",
    "Via an API in custom scripts or applications": "API or applications",
    "Integrated into continuous integration (CI) or code-review tools": "CI or code-review tools",
    "Syntax errors in generated code": "Syntax errors",
    "Semantic errors in generated code": "Semantic errors",
    "Incorrect or misleading answers": "Incorrect answers",
    "Outdated or incomplete knowledge": "Outdated knowledge",
    "Context lacking": "Context lacking",
    "Inability to provide creative answers": "Uncreative answers",
    "Inability of the LLM to execute or verify code it generates": "Verify code it generates",
    "Privacy or confidentiality concerns": "Privacy concerns",
    "Technical issues due to integration": "Integration issues",
    "Inconsistencies in answers": "Inconsistent answers",
    "Struggles with complex or unclear prompts": "Complex prompts",
    "Seamless sharing": "Seamless sharing"
}

# Standardized Scales
FREQ_SCALE_USAGE = ["Never", "Rarely", "Sometimes", "Often", "Always"]
FREQ_SCALE_LIMITS = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
FREQ_SCALE_BENEFIT = ["No benefit", "Low benefit", "Moderate benefit", "High benefit", "Very high benefit"]

# Colors
COLORS_LIKERT_5 = ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac']


def get_text_color(hex_color):
    try:
        rgb = mcolors.hex2color(hex_color)
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return 'black' if luminance > 0.5 else 'white'
    except:
        return 'black'


def clean_text_value(val):
    if pd.isna(val) or val == "": return np.nan
    val = str(val).strip()
    val = re.sub(r'\s*\(.*?\)', '', val)
    return val


def wrap_labels(labels, width=LABEL_WRAP_WIDTH):
    return [textwrap.fill(str(l), width, break_long_words=False) for l in labels]


def shorten_label_manual(text):
    if not text: return ""
    match = re.search(r'\[(.*?)\]', text)
    clean_text = match.group(1).replace('[?', '').strip() if match else text
    for original, short in MANUAL_MAP.items():
        if original.lower() in clean_text.lower(): return short
    return clean_text[:30]


def process_and_get_data(df, columns, scale):
    data = []
    for col in columns:
        q_label = shorten_label_manual(col)
        if not q_label: continue
        counts = df[col].astype(str).str.strip().str.lower().value_counts()
        row = {s: counts.get(s.lower(), 0) for s in scale}
        row['Question'] = q_label
        row['Total'] = sum(row[s] for s in scale)
        row['SortWeight'] = row[scale[-2]] + row[scale[-1]]
        if row['Total'] > 0: data.append(row)
    return data


def plot_diverging_likert_limitations(df_subset, title, filename, colors, y_label="Limitations"):
    scale = FREQ_SCALE_LIMITS
    data = process_and_get_data(df_subset, df_subset.columns, scale)
    if not data: return

    plot_df = pd.DataFrame(data).set_index('Question').sort_values('SortWeight', ascending=True)
    questions = plot_df.index.tolist()

    fig_height = len(questions) * BAR_SPACING + 2
    fig, ax = plt.subplots(figsize=(16, fig_height))
    plot_df[scale].plot(kind='barh', stacked=True, ax=ax, color=colors, edgecolor='none', width=BAR_WIDTH)

    ax.set_xlim(0, plot_df['Total'].max())
    ax.margins(x=0)

    # --- STYLE UPDATES ---
    ax.set_title(title, pad=120, fontsize=24, fontweight='bold')
    ax.set_xlabel('Count', fontsize=20, fontweight='bold')

    # 1. Update Y-axis title and font size
    ax.set_ylabel(y_label, fontsize=Y_TITLE_FONT_SIZE, fontweight='bold')

    # 2. Update X-axis tick (numbers) font size
    ax.tick_params(axis='x', labelsize=X_TICK_FONT_SIZE)

    for i, c in enumerate(ax.containers):
        segment_color = colors[i % len(colors)]
        txt_color = get_text_color(segment_color)
        labels = [f"{int(v.get_width())}" if v.get_width() >= 1 else "" for v in c]
        ax.bar_label(c, labels=labels, label_type='center', fontsize=BAR_LABEL_SIZE, color=txt_color, fontweight='bold')

    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(wrap_labels(questions), fontsize=Y_LABEL_FONT_SIZE, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=len(scale), fontsize=18, frameon=False)

    plt.subplots_adjust(top=0.85)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def plot_diverging_likert(df_subset, title, filename, scale, colors, show_percentage=False, y_label="Category"):
    data = process_and_get_data(df_subset, df_subset.columns, scale)
    if not data: return

    plot_df = pd.DataFrame(data).set_index('Question').sort_values('SortWeight', ascending=True)
    questions = plot_df.index.tolist()
    final_df = plot_df[scale]

    if show_percentage:
        final_df = final_df.div(plot_df['Total'], axis=0) * 100
        x_max = 100
    else:
        x_max = plot_df['Total'].max()

    fig_height = len(questions) * BAR_SPACING + 2
    fig, ax = plt.subplots(figsize=(16, fig_height))
    final_df.plot(kind='barh', stacked=True, ax=ax, color=colors, edgecolor='none', width=BAR_WIDTH)

    ax.set_xlim(0, x_max)
    ax.margins(x=0)

    # --- STYLE UPDATES ---
    ax.set_title(title, pad=120, fontsize=24, fontweight='bold')
    ax.set_xlabel('Percentage (%)' if show_percentage else 'Count', fontsize=20, fontweight='bold')

    # 1. Update Y-axis title and font size
    ax.set_ylabel(y_label, fontsize=Y_TITLE_FONT_SIZE, fontweight='bold')

    # 2. Update X-axis tick (numbers) font size
    ax.tick_params(axis='x', labelsize=X_TICK_FONT_SIZE)

    for i, c in enumerate(ax.containers):
        segment_color = colors[i % len(colors)]
        txt_color = get_text_color(segment_color)
        labels = []
        for v in c:
            w = v.get_width()
            threshold = 7.0 if show_percentage else 0.5
            labels.append(
                f"{w:.0f}%" if show_percentage and w >= threshold else (f"{int(w)}" if w >= threshold else ""))
        ax.bar_label(c, labels=labels, label_type='center', fontsize=BAR_LABEL_SIZE, color=txt_color, fontweight='bold')

    ax.set_yticks(range(len(questions)))
    ax.set_yticklabels(wrap_labels(questions), fontsize=Y_LABEL_FONT_SIZE, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=len(scale), fontsize=18, frameon=False)

    plt.subplots_adjust(top=0.85)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    try:
        df = pd.read_excel('results-survey797953.xlsx')
    except:
        df = pd.read_csv('results-survey797953.xlsx')

    for col in df.columns: df[col] = df[col].apply(clean_text_value)

    limit_cols = [c for c in df.columns if "limitations" in c.lower() and "face" in c.lower()]
    if limit_cols:
        plot_diverging_likert_limitations(df[limit_cols], "",
                                          # "Limitations Encountered",
                                          "chart_F_limits.png",
                                          COLORS_LIKERT_5, y_label="Perceived Limitations")

    access_cols = [c for c in df.columns if "How often do you use the following methods" in c]
    if access_cols:
        plot_diverging_likert(df[access_cols], "",
                              # "LLM Integration Methods",
                              "chart_F_access.png", FREQ_SCALE_USAGE,
                              COLORS_LIKERT_5, y_label="Access Method")

    ben_int_cols = [c for c in df.columns if "How much benefit do you get from the integration of LLM" in c]
    if ben_int_cols:
        plot_diverging_likert(df[ben_int_cols], "",
                              # "Benefit from LLM Integration",
                              "chart_H_benefit_int.png",
                              FREQ_SCALE_BENEFIT, COLORS_LIKERT_5, show_percentage=True, y_label="LLM Integration Benefit")


if __name__ == "__main__":
    main()
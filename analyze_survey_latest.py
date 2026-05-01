import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors  # Added for color processing
from matplotlib.patches import Patch  # Added for custom legends
import re
from wordcloud import WordCloud
import io
import time

# ---------------------------------------------------------
# 1. CONFIGURATION & MAPPING
# ---------------------------------------------------------

# Frequency Scales mappings
FREQ_SCALE_USAGE = ["Never", "Rarely", "Sometimes", "Often", "Always"]
FREQ_SCALE_LIMITS = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
FREQ_SCALE_BENEFIT = ["No benefit", "Low benefit", "Moderate benefit", "High benefit", "Very high benefit"]

# Colors (Formal palette)
COLORS_LIKERT_5 = ['#F0F6FB', '#BDD7E7', '#6BAED6', '#2171B5','#08306B' ] # Brown to Teal
COLORS_BENEFIT_5 = ['#F2F0F7', '#CBC9E2', '#9E9AC8', '#6A51A3', '#3F007D']
COLORS_BINARY = '#4c72b0' # Standard Seaborn Blue



# ---------------------------------------------------------
# 2. TEXT UTILS
# ---------------------------------------------------------

def get_text_color(hex_color):
    """Returns 'black' for light backgrounds and 'white' for dark backgrounds."""
    try:
        rgb = mcolors.hex2color(hex_color)
        # Luminance formula: 0.299*R + 0.587*G + 0.114*B
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return 'black' if luminance > 0.5 else 'white'
    except:
        return 'black'


def clean_label_simple(text):
    """
    Simple regex-based cleaner to remove brackets and descriptions.
    """
    if not text:
        return ""

    # Remove newlines
    text = text.replace("\n", " ").strip()

    # Remove [Option] or [?] parts if they exist (though matching logic handles extraction usually)
    # This is a fallback to clean up any remaining noise
    text = re.sub(r'\[.*?\]', '', text)

    return text.strip()


def clean_text_value(val):
    if pd.isna(val) or val == "":
        return np.nan
    val = str(val).strip()
    # Remove parenthetical explanations
    val = re.sub(r'\s*\(.*?\)', '', val)

    # Standardize Yes/No
    if val.lower() == 'yes': return 'Yes'
    if val.lower() == 'no': return 'No'

    # Standardize Frequencies
    val_lower = val.lower()
    # Handle "Very  Often" (double space)
    val_lower = " ".join(val_lower.split())

    if val_lower == 'very often': return 'Very Often'
    if val_lower == 'always': return 'Always'
    if val_lower == 'often': return 'Often'
    if val_lower == 'sometimes': return 'Sometimes'
    if val_lower == 'rarely': return 'Rarely'
    if val_lower == 'never': return 'Never'

    # Standardize Benefits (Sentence case based on FREQ_SCALE_BENEFIT)
    if 'benefit' in val_lower:
        if val_lower == 'no benefit': return 'No benefit'
        if val_lower == 'low benefit': return 'Low benefit'
        if val_lower == 'moderate benefit': return 'Moderate benefit'
        if val_lower == 'high benefit': return 'High benefit'
        if val_lower == 'very high benefit': return 'Very high benefit'

    return val


# ---------------------------------------------------------
# 3. DATA PROCESSING
# ---------------------------------------------------------

def process_data(filepath):
    try:
        df = pd.read_csv(filepath)
    except:
        df = pd.read_excel(filepath)

    for col in df.columns:
        clean_val = df[col].apply(clean_text_value)
        df[col] = clean_val

    df.dropna(how='all', inplace=True)
    return df


# ---------------------------------------------------------
# 4. CHART GENERATION HELPERS
# ---------------------------------------------------------

def wrap_labels(labels, width=30):
    return ['\n'.join(re.findall(f'.{{1,{width}}}(?:\\s+|$)', l)) for l in labels]


def format_task_label(text):
    """
    Formats the task label to be essentially one word per line,
    BUT keeps short words attached to their context based on the user's preference:
    - 3-letter words (e.g. 'the', 'and') attach to the PREVIOUS word.
    - 2-letter words (e.g. 'of', 'in') attach to the NEXT word.
    """
    words = text.split()
    if not words:
        return ""
    res = [words[0]]
    for i in range(1, len(words)):
        word = words[i]
        prev = words[i - 1]

        # Default delimiter is newline (one word per line)
        delim = "\n"

        # Rule 1: 3-letter words attach to previous (e.g. "Capture the")
        if len(word) == 3:
            delim = " "

        # Rule 2: 2-letter words attach to next (so the PREVIOUS word must have been 2 letters to glue to current)
        # i.e., if prev was 'of', delim is space.
        if len(prev) <= 2:
            delim = " "

        res.append(delim + word)
    return "".join(res)


def plot_combined_usage_benefit_for_requirement(df, usage_cols, all_benefit_cols, title, filename):
    """
    Generates a combined chart where each task has two stacked bars:
    1. Usage Frequency
    2. Benefit Level
    """
    # 1. Match columns based on text inside []
    matched_data = []

    # Helper to clean label for matching
    def get_core_label(col_name):
        match = re.search(r'\[(.*?)\]', col_name)
        if match:
            # Remove " [?]" descriptions and trim
            text = match.group(1).replace('[?', '').split('?')[0]  # Heuristic to clean
            text = re.sub(r"\s*\(.*?\)", "", text)
            return text.strip()
        return None

    # Create map for benefits
    benefit_map = {}
    for b_col in all_benefit_cols:
        lbl = get_core_label(b_col)
        if lbl:
            benefit_map[lbl] = b_col

    for u_col in usage_cols:
        u_lbl = get_core_label(u_col)
        if not u_lbl: continue

        # Try to find match
        b_col = benefit_map.get(u_lbl)

        # Fuzzy fallback if exact match fails
        if not b_col:
            u_clean = re.sub(r'\W+', '', u_lbl).lower()
            for b_lbl, col in benefit_map.items():
                b_clean = re.sub(r'\W+', '', b_lbl).lower()
                if u_clean in b_clean or b_clean in u_clean:
                    b_col = col
                    break

        if b_col:
            short_name = clean_label_simple(u_lbl.replace("Other", "").strip())
            matched_data.append({
                'task': short_name,
                'usage_col': u_col,
                'benefit_col': b_col
            })

    if not matched_data:
        print(f"No matched usage/benefit data for {title}")
        return

    # 2. Setup Plot
    n_tasks = len(matched_data)

    # --- UPDATED PARAMETERS FOR TIGHTER SPACING ---
    step = 0.45  # Decreased from 0.65 (brings groups closer)
    bar_height = 0.18  # Increased from 0.15 (makes bars thicker)
    gap = 0.02  # Decreased from 0.05 (tighter space between Usage/Benefit)

    y_indices = np.arange(n_tasks) * step

    # Adjusted fig_height to match the smaller step
    fig_height = n_tasks * 1.8 + 2
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Calculate Y positions
    usage_y = y_indices - (bar_height / 2 + gap / 2)
    benefit_y = y_indices + (bar_height / 2 + gap / 2)

    # 3. Plotting Helper
    def plot_stacked_bars(ax, y_pos, data_list, col_key, scale, colors):
        lefts = np.zeros(len(data_list))
        for i, category in enumerate(scale):
            widths = []
            counts_list = []
            for item in data_list:
                counts = df[item[col_key]].value_counts()
                c = counts.get(category, 0)
                total = sum([counts.get(k, 0) for k in scale])
                pct = (c / total * 100) if total > 0 else 0
                widths.append(pct)
                counts_list.append(c)

            bars = ax.barh(y_pos, widths, left=lefts, height=bar_height, color=colors[i],
                           edgecolor='white', linewidth=0.5)
            lefts += np.array(widths)

            for j, rect in enumerate(bars):
                w = rect.get_width()
                if w > 5:
                    txt_color = get_text_color(colors[i])
                    count_val = counts_list[j]
                    # Reduced font size slightly to fit in tighter bars if needed
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                            f"{w:.0f}%\n({count_val})", ha='center', va='center',
                            color=txt_color, fontsize=18, fontweight='bold')

    # Plot Bars
    plot_stacked_bars(ax, usage_y, matched_data, 'usage_col', FREQ_SCALE_USAGE, COLORS_LIKERT_5)
    plot_stacked_bars(ax, benefit_y, matched_data, 'benefit_col', FREQ_SCALE_BENEFIT, COLORS_BENEFIT_5)

    # 4. Labels and Legends
    ax.set_yticks(y_indices)
    labels_text = [m['task'] for m in matched_data]
    smart_labels = [format_task_label(l) for l in labels_text]
    ax.set_yticklabels(smart_labels, fontsize=16, fontweight='bold')

    # Adjusted padding for tighter look
    ax.tick_params(axis='y', which='major', pad=55)
    ax.tick_params(axis='x', labelsize=18)
    ax.margins(x=0)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=40)
    ax.invert_yaxis()

    # Add text labels "Usage" and "Benefit"
    for y in y_indices:
        ax.text(-1, y - (bar_height / 2 + gap / 2), "Usage", ha='right', va='center',
                fontsize=14, fontstyle='italic', color='#444')
        ax.text(-1, y + (bar_height / 2 + gap / 2), "Benefit", ha='right', va='center',
                fontsize=14, fontstyle='italic', color='#444')

    # Legends (Adjust bbox_to_anchor if height changed significantly)
    leg1 = ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(COLORS_LIKERT_5, FREQ_SCALE_USAGE)],
                     title="Usage Frequency", loc='upper left', bbox_to_anchor=(1.02, 1.0),
                     frameon=True, fontsize=16, title_fontsize=18)
    ax.add_artist(leg1)

    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(COLORS_BENEFIT_5, FREQ_SCALE_BENEFIT)],
              title="Benefit Level", loc='upper left', bbox_to_anchor=(1.02, 0.45),
              frameon=True, fontsize=16, title_fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_combined_usage_benefit(df, usage_cols, all_benefit_cols, title, filename):
    """
    Generates a combined chart where each task has two stacked bars:
    1. Usage Frequency
    2. Benefit Level
    """
    # 1. Match columns based on text inside []
    matched_data = []

    # Helper to clean label for matching
    def get_core_label(col_name):
        match = re.search(r'\[(.*?)\]', col_name)
        if match:
            # Remove " [?]" descriptions and trim
            text = match.group(1).replace('[?', '').split('?')[0]  # Heuristic to clean
            text = re.sub(r"\s*\(.*?\)", "", text)
            return text.strip()
        return None

    # Create map for benefits
    benefit_map = {}
    for b_col in all_benefit_cols:
        lbl = get_core_label(b_col)
        if lbl:
            benefit_map[lbl] = b_col

    for u_col in usage_cols:
        u_lbl = get_core_label(u_col)
        if not u_lbl: continue

        # Try to find match
        b_col = benefit_map.get(u_lbl)

        # Fuzzy fallback if exact match fails
        if not b_col:
            u_clean = re.sub(r'\W+', '', u_lbl).lower()
            for b_lbl, col in benefit_map.items():
                b_clean = re.sub(r'\W+', '', b_lbl).lower()
                if u_clean in b_clean or b_clean in u_clean:
                    b_col = col
                    break

        if b_col:
            short_name = clean_label_simple(u_lbl.replace("Other", "").strip())
            matched_data.append({
                'task': short_name,
                'usage_col': u_col,
                'benefit_col': b_col
            })

    if not matched_data:
        print(f"No matched usage/benefit data for {title}")
        return

    # --- CHANGE: Reduced spacing between groups ---
    # step=0.65 reduces gap significantly (was 0.75)
    # --- UPDATED PARAMETERS FOR TIGHTER SPACING ---
    # 2. Setup Plot
    n_tasks = len(matched_data)

    # --- UPDATED PARAMETERS FOR TIGHTER SPACING ---
    step = 0.45  # Decreased from 0.65 (brings groups closer)
    bar_height = 0.18  # Increased from 0.15 (makes bars thicker)
    gap = 0.02  # Decreased from 0.05 (tighter space between Usage/Benefit)

    y_indices = np.arange(n_tasks) * step

    # Adjusted fig_height to match the smaller step
    fig_height = n_tasks * 1.8 + 2
    fig, ax = plt.subplots(figsize=(14, fig_height))

    # Calculate Y positions
    usage_y = y_indices - (bar_height / 2 + gap / 2)
    benefit_y = y_indices + (bar_height / 2 + gap / 2)

    # 3. Plotting Helper
    def plot_stacked_bars(ax, y_pos, data_list, col_key, scale, colors):
        lefts = np.zeros(len(data_list))
        for i, category in enumerate(scale):
            widths = []
            counts_list = []
            for item in data_list:
                counts = df[item[col_key]].value_counts()
                c = counts.get(category, 0)
                total = sum([counts.get(k, 0) for k in scale])
                pct = (c / total * 100) if total > 0 else 0
                widths.append(pct)
                counts_list.append(c)

            bars = ax.barh(y_pos, widths, left=lefts, height=bar_height, color=colors[i],
                           edgecolor='white', linewidth=0.5)
            lefts += np.array(widths)

            for j, rect in enumerate(bars):
                w = rect.get_width()
                if w > 5:
                    txt_color = get_text_color(colors[i])
                    count_val = counts_list[j]
                    # Reduced font size slightly to fit in tighter bars if needed
                    ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + rect.get_height() / 2,
                            f"{w:.0f}%\n({count_val})", ha='center', va='center',
                            color=txt_color, fontsize=18, fontweight='bold')

    # Plot Bars
    plot_stacked_bars(ax, usage_y, matched_data, 'usage_col', FREQ_SCALE_USAGE, COLORS_LIKERT_5)
    plot_stacked_bars(ax, benefit_y, matched_data, 'benefit_col', FREQ_SCALE_BENEFIT, COLORS_BENEFIT_5)

    # 4. Labels and Legends
    ax.set_yticks(y_indices)
    labels_text = [m['task'] for m in matched_data]
    smart_labels = [format_task_label(l) for l in labels_text]
    ax.set_yticklabels(smart_labels, fontsize=16, fontweight='bold')

    # Adjusted padding for tighter look
    ax.tick_params(axis='y', which='major', pad=55)
    ax.tick_params(axis='x', labelsize=18)
    ax.margins(x=0)
    ax.set_title(title, fontsize=18, fontweight='bold', pad=40)
    ax.invert_yaxis()

    # Add text labels "Usage" and "Benefit"
    for y in y_indices:
        ax.text(-1, y - (bar_height / 2 + gap / 2), "Usage", ha='right', va='center',
                fontsize=14, fontstyle='italic', color='#444')
        ax.text(-1, y + (bar_height / 2 + gap / 2), "Benefit", ha='right', va='center',
                fontsize=14, fontstyle='italic', color='#444')

    # Legends (Adjust bbox_to_anchor if height changed significantly)
    leg1 = ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(COLORS_LIKERT_5, FREQ_SCALE_USAGE)],
                     title="Usage Frequency", loc='upper left', bbox_to_anchor=(1.02, 1.0),
                     frameon=True, fontsize=16, title_fontsize=18)
    ax.add_artist(leg1)

    ax.legend(handles=[Patch(facecolor=c, label=l) for c, l in zip(COLORS_BENEFIT_5, FREQ_SCALE_BENEFIT)],
              title="Benefit Level", loc='upper left', bbox_to_anchor=(1.02, 0.45),
              frameon=True, fontsize=16, title_fontsize=18)

    plt.tight_layout()
    plt.subplots_adjust(right=0.75)
    plt.savefig(filename, dpi=300)
    plt.close()
# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    file_path = 'results-survey797953.xlsx'
    df = process_data(file_path)

    # --- PREPARE FOR COMBINED CHARTS ---
    # We need to fetch all benefit columns first to pass them to the combined plotter
    ben_task_cols = [c for c in df.columns if
                     "How much do you benefit from the outputs of LLMs for the following tasks" in c]

    print("Generating Combined Usage and Benefit Charts...")

    # --- E) Requirement Gathering (Combined Usage + Benefit) ---
    req_cols = [c for c in df.columns if "In which of the following requirement-gathering tasks" in c]
    if req_cols:
        plot_combined_usage_benefit_for_requirement(df, req_cols, ben_task_cols,"",
                                    # "Frequency of LLM Usage and Benefits in Requirement Gathering Tasks",
                                    "chart_E_req_combined.png")

    # --- F) Other Frequency Charts (Combined) ---

    # Design
    design_cols = [c for c in df.columns if "In which of the following software designing tasks" in c]
    if design_cols:
        plot_combined_usage_benefit(df, design_cols, ben_task_cols,"",
                                    # "Frequency of LLM Usage and Benefits in Design Tasks",
                                    "chart_F_design_combined.png")

    # Development
    dev_cols = [c for c in df.columns if "In which of the following development tasks" in c]
    if dev_cols:
        plot_combined_usage_benefit(df, dev_cols, ben_task_cols,"",
                                    # "Frequency of LLM Usage and Benefits in Development Tasks",
                                    "chart_F_dev_combined.png")

    # Testing
    test_cols = [c for c in df.columns if "In which of the following software testing tasks" in c]
    if test_cols:
        plot_combined_usage_benefit(df, test_cols, ben_task_cols,"",
                                    # "Frequency of LLM Usage and Benefits in Testing Tasks",
                                    "chart_F_test_combined.png")

    print("Usage and Benefits Analysis Complete. Files generated.")


if __name__ == "__main__":
    main()
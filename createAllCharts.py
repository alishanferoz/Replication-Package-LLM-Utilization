import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
import re
from wordcloud import WordCloud
import io
import time

# Try to import google-generativeai, handle if missing
try:
    import google.genai as genai

    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False
    print(
        "Warning: google-generativeai library not found. Install via 'pip install google-generativeai' to use AI shortening.")

# ---------------------------------------------------------
# 1. CONFIGURATION & MAPPING
# ---------------------------------------------------------

# PUT YOUR GEMINI API KEY HERE
GEMINI_API_KEY = ""

# Frequency Scales mappings
FREQ_SCALE_USAGE = ["Never", "Rarely", "Sometimes", "Often", "Always"]
FREQ_SCALE_LIMITS = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
FREQ_SCALE_BENEFIT = ["No benefit", "Low benefit", "Moderate benefit", "High benefit", "Very high benefit"]

# # Colors (Formal palette)
# COLORS_LIKERT_5 = ['#BF7208', '#3246FC', '#6675FA', '#A1AAFF', '#C3C8FA']  # Red to Blue diverging
# COLORS_BENEFIT_5 = ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac']  # Brown to Teal
# COLORS_BINARY = '#4c72b0'  # Standard Seaborn Blue

# Colors (Formal palette)
COLORS_LIKERT_5 = ['#8c510a', '#d8b365', '#f6e8c3', '#c7eae5', '#5ab4ac'] # Brown to Teal
COLORS_BENEFIT_5 = ['#762A83', '#AF8DC3', '#F7F7F7', '#7FBF7B', '#1B7837']
COLORS_BINARY = '#4c72b0' # Standard Seaborn Blue


# Country to Continent Mapping
CONTINENT_MAP = {
    'Pakistan': 'Asia', 'India': 'Asia', 'Iran': 'Asia', 'South Korea': 'Asia', 'Israel': 'Asia',
    'Vietnam': 'Asia', 'Saudi Arabia': 'Asia', 'Singapore': 'Asia',
    'Egypt': 'Africa', 'South Africa': 'Africa', 'Tunisia': 'Africa', 'Kenya': 'Africa',
    'Canada': 'North America', 'United States of America': 'North America', 'Mexico': 'North America',
    'Brazil': 'South America', 'Argentina': 'South America',
    'Germany': 'Europe', 'Hungary': 'Europe', 'Poland': 'Europe', 'Spain': 'Europe', 'Slovakia': 'Europe',
    'Netherlands': 'Europe', 'Austria': 'Europe', 'Romania': 'Europe', 'Russia': 'Asia',
    'Czechia': 'Europe', 'Greece': 'Europe',
    'Ireland': 'Europe', 'Italy': 'Europe', 'Portugal': 'Europe', 'Montenegro': 'Europe', 'France': 'Europe',
    'United Kingdom': 'Europe',
    'Australia': 'Australia'
}
def get_text_color(hex_color):
    """Returns 'black' for light backgrounds and 'white' for dark backgrounds."""
    try:
        rgb = mcolors.hex2color(hex_color)
        # Luminance formula: 0.299*R + 0.587*G + 0.114*B
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return 'black' if luminance > 0.5 else 'white'
    except:
        return 'black'
# Cache for shortened labels to avoid repeated API calls
LABEL_CACHE = {}


# ---------------------------------------------------------
# 2. AI & TEXT UTILS
# ---------------------------------------------------------

def shorten_label_ai(text):
    """
    Uses Gemini API to shorten text if configured, otherwise falls back to regex/truncation.
    """
    if not text:
        return ""

    # Return cached if available
    if text in LABEL_CACHE:
        return LABEL_CACHE[text]

    # Clean initial text
    clean_text = text.replace("\n", " ").strip()

    if HAS_GENAI and GEMINI_API_KEY != "":
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel('gemini-2.0-flash')  # Using a fast model
            prompt = f"Shorten the following survey option label to a very concise 2-5 word phrase for a chart axis. Remove filler words. Keep the core meaning. Input: '{clean_text}'"
            response = model.generate_content(prompt)
            shortened = response.text.strip().replace('"', '').replace("'", "")
            print(f"AI Shortened: '{clean_text[:30]}...' -> '{shortened}'")
            LABEL_CACHE[text] = shortened
            time.sleep(0.5)  # Avoid hitting rate limits too hard
            return shortened
        except Exception as e:
            print(f"AI Error: {e}. Falling back to standard shortening.")

    # Fallback: Simple heuristic shortening
    # Remove things in brackets
    shortened = re.sub(r'\[.*?\]', '', clean_text)
    # Take first 6 words max
    words = shortened.split()
    if len(words) > 6:
        shortened = " ".join(words[:6]) + "..."
    LABEL_CACHE[text] = shortened
    return shortened


def clean_text_value(val):
    if pd.isna(val) or val == "":
        return np.nan
    val = str(val).strip()
    # Remove parenthetical explanations
    val = re.sub(r'\s*\(.*?\)', '', val)
    # Standardize Yes/No
    if val.lower() == 'yes': return 'Yes'
    if val.lower() == 'no': return 'No'
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


def normalize_domain(df):
    col_name = "What is your primary domain of software development?"
    if col_name not in df.columns:
        for c in df.columns:
            if "primary domain" in c:
                col_name = c
                break

    if col_name not in df.columns:
        return df, None

    def map_domain(val):
        if pd.isna(val): return "Non"
        v = str(val).lower().strip()
        if any(x in v for x in ['data science', 'data engineer', 'data analyst', 'ai', 'genai', 'machine learning',
                                'nlp', 'vision', 'intelligence', 'analytics', 'hpc', 'gpgpu', 'data engineering']):
            return "ai & data science"

        # 2. Database Development (New Category)
        if any(x in v for x in ['database', 'sql', 'storage', 'data management']):
            return "database development"

        # 3. Backend Development
        if any(x in v for x in ['backend', 'back end', 'mvc', 'api', 'serverless', 'java services', 'python', 'js']):
            # Distinguish general backend from full stack later
            if 'front' not in v:
                return "backend development"

        # 4. Web Development (General & Frontend)
        if any(x in v for x in ['web', 'frontend', 'front end', 'web app', 'web dev', 'web science', 'webhosting']):
            if 'mobile' in v: return "full stack"  # Hybrid web/mobile
            return "web development"

        # 5. Full Stack
        if any(x in v for x in ['full stack', 'fullstack', 'asp.net']):
            return "full stack"

        # 6. Mobile & Desktop Development
        if any(x in v for x in ['mobile', 'android', 'ios', 'flutter', 'desktop', 'windows software']):
            return "mobile & desktop development"

        # 7. DevOps & Infrastructure
        if any(x in v for x in ['devops', 'cloud', 'automation', 'ci/cd', 'platform engineering', 'network', 'sysadmin',
                                'system administration']):
            return "devops & automation"

        # 8. Embedded, Systems & Robotics
        if any(x in v for x in
               ['embedded', 'robotics', 'navigation', 'systems', 'iot', 'automotive', 'telecommunications']):
            return "embedded & robotics"

        # 9. Media & Design (Updated name)
        if any(x in v for x in ['logo & branding', 'marketing', 'designing', 'game development']):
            return "media designing"

        # 10. Industry Specific: Finance
        if any(x in v for x in ['finance', 'fintech', 'banking', 'payments', 'fiance', 'insurance']):
            return "finance & fintech"

        # 11. Industry Specific: Healthcare
        if any(x in v for x in ['healthcare', 'health care', 'medical']):
            return "healthcare"

        # 12. Industry Specific: Manufacturing & ERP
        if any(x in v for x in ['manufacturing', 'mes', 'industrial', 'agriculture', 'supply chain']):
            return "manufacturing & supply chain"

        if any(x in v for x in ['erp', 'sap', 'crm', 'payroll', 'hr', 'b2b']):
            return "ERP & CRM"

        # 13. Others
        if any(x in v for x in ['e-commerce', 'e commerce', 'retail']):
            return "e-commerce"
        if any(x in v for x in ['education']):
            return "education"
        if any(x in v for x in ['research', 'scientific', 'physics', 'chemistry', 'quantum']):
            return "scientific research & computing"
        if any(x in v for x in ['product development']):
            return "product development"
        if any(x in v for x in ['government', 'legal']):
            return "government & legal"

        # Catch-all for "Software Development", "IT", or "Architecting"
        if any(x in v for x in ['it', 'architecting']):
            return "IT"
        if any(x in v for x in ['software development', 'application development']):
            return "Software Development"

    df['Normalized_Domain'] = df[col_name].apply(map_domain)
    return df, col_name


# ---------------------------------------------------------
# 4. CHART GENERATION HELPERS
# ---------------------------------------------------------

def wrap_labels(labels, width=30):
    return ['\n'.join(re.findall(f'.{{1,{width}}}(?:\\s+|$)', l)) for l in labels]


def plot_wordcloud(text_data, title, filename):
    wc = WordCloud(width=800, height=400, background_color='white',
                   color_func=lambda *args, **kwargs: "black").generate_from_frequencies(text_data)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_diverging_likert_limitations(df_subset, title, filename, colors):
    """
    Dedicated function for Limitations to strictly use the 'Very Often' scale
    and ensure robust detection of that value.
    """
    scale = ["Never", "Rarely", "Sometimes", "Often", "Very Often"]
    data = []
    questions = []

    for col in df_subset.columns:
        match = re.search(r'\[(.*?)\]', col)
        raw_label = match.group(1).replace('[?', '') if match else col
        raw_label = raw_label.replace("Other", "").strip()
        if not raw_label: continue

        # Apply AI Shortening here
        q_label = shorten_label_ai(raw_label)

        counts = df_subset[col].value_counts()
        row = {k: counts.get(k, 0) for k in scale}

        # Robust fallback: check for case-insensitive matches if count is 0
        if row['Very Often'] == 0:
            for key in counts.index:
                if str(key).lower().strip() == 'very often':
                    row['Very Often'] += counts[key]

        total_val = sum(row.values())
        row['Question'] = q_label
        row['Total'] = total_val

        if row['Total'] > 0:
            data.append(row)
            questions.append(q_label)

    if not data:
        return

    plot_df = pd.DataFrame(data).set_index('Question')
    plot_df = plot_df[scale]  # Reorder

    # Always Counts for this one
    xlabel = 'Count'

    fig_height = len(questions) * 1.0 + 2
    fig, ax = plt.subplots(figsize=(12, fig_height))

    plot_df.plot(kind='barh', stacked=True, ax=ax, color=colors, edgecolor='none', width=0.8)

    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    for i, c in enumerate(ax.containers):
        # Determine text color based on bar color
        segment_color = colors[i % len(colors)]
        txt_color = get_text_color(segment_color)

        labels = []
        for v in c:
            w = v.get_width()
            if w > 0:
                labels.append(f"{int(w)}")
            else:
                labels.append("")
        ax.bar_label(c, labels=labels, label_type='center', fontsize=12, color=txt_color, fontweight='bold')
    # for c in ax.containers:
    #     labels = []
    #     for v in c:
    #         w = v.get_width()
    #         if w > 0:
    #             labels.append(f"{int(w)}")
    #         else:
    #             labels.append("")
    #     ax.bar_label(c, labels=labels, label_type='center', fontsize=12, color='white', fontweight='bold')

    ax.set_yticklabels(wrap_labels(plot_df.index, 40), fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_diverging_likert(df_subset, title, filename, scale, colors, show_percentage=False, bar_height_equal=True):
    data = []
    questions = []

    for col in df_subset.columns:
        match = re.search(r'\[(.*?)\]', col)
        raw_label = match.group(1).replace('[?', '') if match else col
        raw_label = raw_label.replace("Other", "").strip()
        if not raw_label: continue

        # Apply AI Shortening here
        q_label = shorten_label_ai(raw_label)

        counts = df_subset[col].value_counts()
        row = {k: counts.get(k, 0) for k in scale}
        total_val = sum(row.values())
        row['Question'] = q_label
        row['Total'] = total_val

        if row['Total'] > 0:
            data.append(row)
            questions.append(q_label)

    if not data:
        return

    plot_df = pd.DataFrame(data).set_index('Question')
    plot_df = plot_df[scale]  # Reorder

    if show_percentage:
        plot_df_pct = plot_df.div(plot_df.sum(axis=1), axis=0) * 100
        xlabel = 'Percentage'
    else:
        plot_df_pct = plot_df
        xlabel = 'Count'

    # Dynamic Height: Increase per bar to make it "bigger"
    fig_height = len(questions) * 1.0 + 2  # increased multiplier
    fig, ax = plt.subplots(figsize=(12, fig_height))

    plot_df_pct.plot(kind='barh', stacked=True, ax=ax, color=colors, edgecolor='none',
                     width=0.8 if bar_height_equal else 0.5)

    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')
    ax.set_xlabel(xlabel)
    ax.invert_yaxis()
    for i, c in enumerate(ax.containers):
        # Determine text color based on bar color
        segment_color = colors[i % len(colors)]
        txt_color = get_text_color(segment_color)

        labels = []
        for v in c:
            w = v.get_width()
            if w > 0:
                labels.append(f"{w:.0f}%" if show_percentage else f"{int(w)}")
            else:
                labels.append("")
        ax.bar_label(c, labels=labels, label_type='center', fontsize=12, color=txt_color, fontweight='bold')

    # for c in ax.containers:
    #     labels = []
    #     for v in c:
    #         w = v.get_width()
    #         if w > 0:
    #             labels.append(f"{w:.0f}%" if show_percentage else f"{int(w)}")
    #         else:
    #             labels.append("")
    #     ax.bar_label(c, labels=labels, label_type='center', fontsize=12, color='white', fontweight='bold')

    # Wrap labels even if shortened, just in case
    ax.set_yticklabels(wrap_labels(plot_df.index, 40), fontsize=11)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


def plot_simple_bar(data_series, title, filename, orientation='v', color=COLORS_BINARY, xlabel="", ylabel=""):
    # 1. First, apply AI shortening
    new_index = [shorten_label_ai(i) for i in data_series.index]

    # 2. IMPORTANT: Apply wrap_labels to the shortened text to handle long phrases
    # For horizontal charts (h), we want narrower wrapping (width=20)
    # For vertical charts (v), we can go a bit wider (width=25)
    wrap_width = 20 if orientation == 'h' else 25
    wrapped_index = wrap_labels(new_index, width=wrap_width)

    data_series.index = wrapped_index

    # Dynamic size calculation
    if orientation == 'v':
        fig_size = (12, 8)
    else:
        # Increase height per bar for horizontal charts to accommodate multiple lines
        fig_size = (14, len(data_series) * 1.2 + 2)

    plt.figure(figsize=fig_size)

    if orientation == 'v':
        ax = data_series.plot(kind='bar', color=color, width=0.6)
        ax.set_xticklabels(data_series.index, rotation=0, fontsize=16)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontweight='bold',
                        fontsize=20)
    else:
        data_series = data_series.sort_values(ascending=True)
        ax = data_series.plot(kind='barh', color=color, width=0.7)  # Slightly thicker bars
        ax.tick_params(axis='x', labelsize=18)

        # Set alignment to 'right' so wrapped lines stack cleanly against the axis
        ax.set_yticklabels(data_series.index, fontsize=16, ha='right', va='center')
        ax.set_xlabel(xlabel, fontsize=16)

        for p in ax.patches:
            ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', xytext=(10, 0), textcoords='offset points', fontweight='bold',
                        fontsize=20)

    # Use subplots_adjust to ensure the left margin is wide enough for the wrapped labels
    if orientation == 'h':
        plt.subplots_adjust(left=0.35)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_simple_bar_sdlc(data_series, title, filename, orientation='v', color=COLORS_BINARY, xlabel="", ylabel=""):
    # Shorten indices if they are text
    new_index = [shorten_label_ai(i) for i in data_series.index]
    data_series.index = new_index

    # Dynamic size
    if orientation == 'v':
        fig_size = (12, 8)
    else:
        fig_size = (12, len(data_series) * 0.8 + 2)

    plt.figure(figsize=fig_size)

    if orientation == 'v':
        ax = data_series.plot(kind='bar', color=color, width=0.6)
        ax.set_xticklabels(wrap_labels(data_series.index, 20), rotation=0, fontsize=16)
        ax.tick_params(axis='y', labelsize=18)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontweight='bold',fontsize=20)
    else:
        data_series = data_series.sort_values(ascending=True)
        ax = data_series.plot(kind='barh', color=color, width=0.6)
        ax.tick_params(axis='x', labelsize=18)
        ax.set_yticklabels(wrap_labels(data_series.index, 40), fontsize=16)
        ax.set_xlabel(xlabel, fontsize=16)
        for p in ax.patches:
            ax.annotate(f'{int(p.get_width())}', (p.get_width(), p.get_y() + p.get_height() / 2.),
                        ha='left', va='center', xytext=(5, 0), textcoords='offset points', fontweight='bold',fontsize=20)

    # plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# ---------------------------------------------------------
# 5. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    file_path = 'results-survey797953.xlsx'
    df = process_data(file_path)

    # --- A) Experience Chart (Updated Segregation) ---
    exp_col = "How many years of total experience do you have in Software Development?"
    if exp_col in df.columns:
        df['Experience'] = pd.to_numeric(df[exp_col], errors='coerce')

        # Define categories
        # < 2, 2 to 5, 6 to 10, Above 10
        def cat_experience(years):
            if pd.isna(years): return None
            if years < 2: return "< 2 years"
            if 2 <= years <= 5: return "2 - 5 years"
            if 6 <= years <= 10: return "6 - 10 years"
            return "> 10 years"

        df['Exp_Cat'] = df['Experience'].apply(cat_experience)

        # Explicit order
        order = ["< 2 years", "2 - 5 years", "6 - 10 years", "> 10 years"]
        exp_counts = df['Exp_Cat'].value_counts().reindex(order).fillna(0)

        plt.figure(figsize=(10, 6))
        ax = exp_counts.plot(kind='bar', color=COLORS_BINARY, edgecolor='black', width=0.6)

        # plt.title('Years of Experience in Software Development', fontsize=16, fontweight='bold')
        plt.xlabel('Experience Range', fontsize=16)
        plt.ylabel('Count', fontsize=16)
        plt.xticks(rotation=0, fontsize=16)

        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontweight='bold',
                            fontsize=12)

        plt.tight_layout()
        plt.savefig('chart_A_experience.png', bbox_inches='tight')
        plt.close()

    # --- B) Domain Analysis ---
    df, dom_col = normalize_domain(df)
    if dom_col:
        domain_counts = df['Normalized_Domain'].value_counts()
        domain_counts.to_csv("domain_frequencies.csv")
        plot_wordcloud(domain_counts.to_dict(),
                       "Primary Domain of Software Development",
                       "chart_B_domain_cloud.png")

    # --- C) Country Analysis ---
    country_col = "Please select your country of residence from the list below:"
    # if country_col in df.columns:
    #     df['Continent'] = df[country_col].map(CONTINENT_MAP).fillna('Other')
    #     cont_counts = df['Continent'].value_counts()
    #     plt.figure(figsize=(6, 8))
    #     hm_data = pd.DataFrame(cont_counts)
    #     sns.heatmap(hm_data, annot=True, fmt='d', cmap="YlGnBu", cbar=False, linewidths=1)
    #     plt.title("Respondents by Continent")
    #     plt.tight_layout()
    #     plt.savefig('chart_C_continent_heatmap.png')
    #     plt.close()
    if country_col in df.columns:
        df['Continent'] = df[country_col].map(CONTINENT_MAP).fillna('Other')
        # unmapped_list = df[df['Continent'] == 'Other'][country_col].unique()
        # print(unmapped_list)
        cont_counts = df['Continent'].value_counts()
        plt.figure(figsize=(6, 8))
        hm_data = pd.DataFrame(cont_counts)
        # Added annot_kws to increase font size
        ax = sns.heatmap(hm_data, annot=True, fmt='d', cmap="YlGnBu", cbar=False, linewidths=1, annot_kws={"size": 14,"weight": "bold"})
        # Increase Y-axis (Continent names) font size
        plt.yticks(fontsize=14, rotation=0)
        # plt.title("Respondents by Continent")
        plt.ylabel('Continent',fontsize=16)
        plt.tight_layout()
        plt.savefig('chart_C_continent_heatmap.png')
        plt.close()
    # --- D) SDLC Phases ---
    phase_cols = [c for c in df.columns if "In which phases of the software development lifecycle" in c]
    if phase_cols:
        phase_counts = {}
        for col in phase_cols:
            match = re.search(r'\[(.*?)\]', col)
            label = match.group(1) if match else "Unknown"
            label = label.replace(
                'Requirements gathering or analysis',
                'Requirements gathering and analysis'
            ).replace(
                'Software design or architecture',
                'Software architecture and design'
            )
            count = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0).sum()
            phase_counts[label] = count

        s_phase = pd.Series(phase_counts)
        order = ['Requirements gathering and analysis', 'Software architecture and design', 'Implementation (Coding)',
                 'Testing']
        s_phase = s_phase.reindex(order).dropna()
        plot_simple_bar_sdlc(s_phase,
                        "Usage of LLMs in SDLC Phases",
                        "chart_D_phases.png", orientation='v',
                        # xlabel="SDLC Phase", ylabel="Count"
                        xlabel="", ylabel=""
                        )

    # --- E) Requirement Gathering Tasks (Frequency) ---
    # req_cols = [c for c in df.columns if "In which of the following requirement-gathering tasks" in c]
    # if req_cols:
    #     plot_diverging_likert(df[req_cols], "Frequency of LLM Usage in Requirement Gathering", "chart_E_req_freq.png",
    #                           FREQ_SCALE_USAGE, COLORS_LIKERT_5)

    # # --- F) Other Frequency Charts ---
    # design_cols = [c for c in df.columns if "In which of the following software designing tasks" in c]
    # if design_cols:
    #     plot_diverging_likert(df[design_cols], "LLM Usage in Design Tasks", "chart_F_design.png", FREQ_SCALE_USAGE,
    #                           COLORS_LIKERT_5)

    # dev_cols = [c for c in df.columns if "In which of the following development tasks" in c]
    # if dev_cols:
    #     plot_diverging_likert(df[dev_cols], "LLM Usage in Development Tasks", "chart_F_dev.png", FREQ_SCALE_USAGE,
    #                           COLORS_LIKERT_5)

    # test_cols = [c for c in df.columns if "In which of the following software testing tasks" in c]
    # if test_cols:
    #     plot_diverging_likert(df[test_cols], "LLM Usage in Testing Tasks", "chart_F_test.png", FREQ_SCALE_USAGE,
    #                           COLORS_LIKERT_5)
    # limit_cols = [c for c in df.columns if "limitations" in c.lower() and "face" in c.lower()]
    # # limit_cols = [c for c in df.columns if "How often do you face these limitations" in c]
    # if limit_cols:
    #     plot_diverging_likert_limitations(df[limit_cols], "Limitations Encountered", "chart_F_limits.png", FREQ_SCALE_LIMITS,
    #                           COLORS_LIKERT_5)
    # limit_cols = [c for c in df.columns if "limitations" in c.lower() and "face" in c.lower()]
    # if limit_cols:
    #     plot_diverging_likert_limitations(df[limit_cols], "Limitations Encountered", "chart_F_limits.png",
    #                                       COLORS_LIKERT_5)

    # access_cols = [c for c in df.columns if "How often do you use the following methods" in c]
    # if access_cols:
    #     plot_diverging_likert(df[access_cols], "LLM Access Methods", "chart_F_access.png", FREQ_SCALE_USAGE,
    #                           COLORS_LIKERT_5)

    # --- G & H) Benefits (Percentage) ---
    # ben_task_cols = [c for c in df.columns if
    #                  "How much do you benefit from the outputs of LLMs for the following tasks" in c]
    # if ben_task_cols:
    #     plot_diverging_likert(df[ben_task_cols], "Benefit from LLMs (Tasks)", "chart_G_benefit_tasks.png",
    #                           FREQ_SCALE_BENEFIT, COLORS_BENEFIT_5, show_percentage=True)

    # ben_int_cols = [c for c in df.columns if "How much benefit do you get from the integration of LLM" in c]
    # if ben_int_cols:
    #     plot_diverging_likert(df[ben_int_cols], "Benefit from LLM Integration", "chart_H_benefit_int.png",
    #                           FREQ_SCALE_BENEFIT, COLORS_BENEFIT_5, show_percentage=True)

    # --- I & J) Yes/No Bars ---
    diagram_cols = [c for c in df.columns if "What design diagrams do you usually generate" in c]
    if diagram_cols:
        counts = {}
        for col in diagram_cols:
            match = re.search(r'\[(.*?)\]', col)
            label = match.group(1).replace('[?', '') if match else "Unknown"
            counts[label] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0).sum()
        plot_simple_bar(pd.Series(counts),
                        "Design Diagrams Generated",
                        "chart_I_diagrams.png", orientation='v')

    test_scen_cols = [c for c in df.columns if "What type of test scenarios" in c]
    if test_scen_cols:
        counts = {}
        for col in test_scen_cols:
            match = re.search(r'\[(.*?)\]', col)
            label = match.group(1).replace('[?', '') if match else "Unknown"
            counts[label] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0).sum()
        plot_simple_bar(pd.Series(counts),
                        "Test Scenarios Generated",
                        "chart_J_scenarios.png", orientation='h')

    code_cols = [c for c in df.columns if "Which code generation tasks have you used" in c]
    if code_cols:
        counts = {}
        for col in code_cols:
            match = re.search(r'\[(.*?)\]', col)
            label = match.group(1).replace('[?', '') if match else "Unknown"
            counts[label] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0).sum()
        plot_simple_bar(pd.Series(counts),
                        "Code Generation Tasks",
                        "chart_J_codegen.png", orientation='h')

    exp_ben_cols = [c for c in df.columns if "Which of the following benefits have you experienced" in c]
    if exp_ben_cols:
        counts = {}
        for col in exp_ben_cols:
            match = re.search(r'\[(.*?)\]', col)
            label = match.group(1).replace('[?', '') if match else "Unknown"
            counts[label] = df[col].apply(lambda x: 1 if str(x).lower() == 'yes' else 0).sum()
        plot_simple_bar(pd.Series(counts),
                        "Benefits Experienced",
                        "chart_J_benefits_exp.png", orientation='h')

    # --- K) Future Replacement ---
    future_col = "Do you think large language models (LLMs) could replace software developers in the future?"
    # if future_col in df.columns:
    #     counts = df[future_col].value_counts()
    #     plt.figure(figsize=(8, 8))
    #     plt.pie(counts, labels=counts.index, autopct='%5.1f%%', colors=sns.color_palette('pastel'))
    #     plt.title("Will LLMs Replace Developers?")
    #     plt.tight_layout()
    #     plt.savefig('chart_K_future.png')
    #     plt.close()
    if future_col in df.columns:
        counts = df[future_col].value_counts()
        plt.figure(figsize=(10, 10))  # Increased figure size slightly

        # Added textprops for bold and larger font
        plt.pie(counts,
                labels=counts.index,
                autopct='%1.1f%%',
                colors=sns.color_palette('pastel'),
                textprops={'fontsize': 20, 'weight': 'bold'})

        # plt.title("Will LLMs Replace Developers?", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('chart_K_future.png', dpi=300)
        plt.close()
    df.to_csv('cleaned_survey_data.csv', index=False)
    print("Analysis Complete. Files generated.")


if __name__ == "__main__":
    main()

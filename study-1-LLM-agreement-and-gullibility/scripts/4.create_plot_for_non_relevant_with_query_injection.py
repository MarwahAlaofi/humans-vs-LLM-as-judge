import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

folder_path = '../non-relevant_with_query_injection_files'
output_folder = '../tables_and_plots'
os.makedirs(output_folder, exist_ok=True)

field_name_mapping = {
    'passage_and_query': "NonRelP+Q",
    'passage_and_query_words': "NonRelP+QWs",
}

response_colors = {
    0: "#f5f3f6",
    1: "#F0E68C",
    2: "#D0F0C0",
    3: "#71BC78"
}

model_name_mapping = {
    'anthropic.claude-3-haiku-20240307-v1:0': 'Claude-3 Haiku',
    'anthropic.claude-3-opus-20240229-v1:0': 'Claude-3 Opus',
    'cohere.command-r-v1:0': 'Command-R',
    'cohere.command-r-plus-v1:0': 'Command-R+',
    'meta.llama3-8b-instruct-v1:0': 'LLaMA3 8B',
    'meta.llama3-70b-instruct-v1:0': 'LLaMA3 70B',
    'gpt-4o-2024-05-13': 'GPT-4o',
    'gpt-4-0613': 'GPT-4',
    "gpt-35-turbo-1106": 'GPT-3.5-turbo'
}

author_mapping = {
    'simple': 'Basic',
    'Upadhyay': 'Rationale',
    'thomas': 'Utility',
}

plt.rcParams.update({
    'text.usetex': True,
    'svg.fonttype': 'none',
    'text.latex.preamble': r'\usepackage{libertine}',
    'font.size': 14,
    'font.family': 'Linux Libertine',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'libertine',
    'mathtext.it': 'libertine:italic',
    'mathtext.bf': 'libertine:bold'
})

def extract_model_prompt(filename):
    parts = filename.replace('.csv', '').split('_')
    model = parts[0]
    prompt = parts[2]
    mapped_prompt = author_mapping.get(prompt, prompt)
    mapped_model = model_name_mapping.get(model, model)
    return mapped_model, mapped_prompt

def process_and_plot(model_data, model):
    sorted_prompts = sorted(model_data.keys(), key=lambda x: list(author_mapping.values()).index(author_mapping.get(x, x)))
    fig, axes = plt.subplots(1, len(sorted_prompts), figsize=(7, 4), sharey=True)
    if len(sorted_prompts) == 1:
        axes = [axes]

    for ax, prompt in zip(axes, sorted_prompts):
        df = model_data[prompt]
        df['field'] = df['field'].map(field_name_mapping)
        df = df[df['field'].isin(field_name_mapping.values())]
        if 'NonRelP' not in df['field'].values:
            nonrelp_row = pd.DataFrame({'field': ['NonRelP'], 'O_score': [0], 'id': [0]})
            df = pd.concat([df, nonrelp_row], ignore_index=True)

        pivot_df = df.pivot_table(index='field', columns='O_score', values='id', aggfunc='size', fill_value=0)
        pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)
        colors = [response_colors.get(x) for x in range(4)]
        bars = pivot_df.plot(kind='bar', stacked=True, width=0.7, ax=ax, color=colors)
        ax.set_title(prompt)
        ax.set_ylabel('Score Distribution' if ax is axes[0] else '')
        ax.set_xlabel('')
        ax.set_xticklabels(pivot_df.index, rotation=30, ha='center', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

        for rect in bars.patches:
            height = rect.get_height()
            if height > 0.05:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2, f"{height:.2f}", ha='center',
                        va='center', color='black', fontsize=8)

        ax.get_legend().remove()

    legend_elements = [Patch(facecolor=response_colors[i], label=f'{i}') for i in range(len(response_colors))]
    if len(axes) > 1:
        axes[0].legend(handles=legend_elements, title="Relevance Scores", loc='upper center',
                       bbox_to_anchor=(0.40, -0.35), ncol=2, fontsize='small', title_fontsize='small')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(bottom=0.37)
    plt.savefig(f'{output_folder}/{model}_NonRelP_plots.pdf')
    plt.close()

model_plots_data = {}
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        model, prompt = extract_model_prompt(file_name)
        file_path = os.path.join(folder_path, file_name)
        df = pd.read_csv(file_path)
        df = df.dropna(subset=['O_score'])
        df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
        df = df.dropna(subset=['O_score'])
        df['O_score'] = np.round(df['O_score']).astype(int)

        if model not in model_plots_data:
            model_plots_data[model] = {}

        if prompt in model_plots_data[model]:
            model_plots_data[model][prompt] = pd.concat([model_plots_data[model][prompt], df])
        else:
            model_plots_data[model][prompt] = df

for model, data in model_plots_data.items():
    process_and_plot(data, model)

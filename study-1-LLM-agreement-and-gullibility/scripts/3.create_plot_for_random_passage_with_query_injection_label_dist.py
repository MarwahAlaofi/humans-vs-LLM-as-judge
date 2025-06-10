import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Patch

folder_path = '../random_passage_with_query_injection_files'

response_colors = {
    0: "#f5f3f6",  # Dark Salmon
    1: "#F0E68C",  # Light Golden Yellow
    2: "#D0F0C0",  # Pastel Green
    3: "#71BC78"   # Dark Green
}

model_name_mapping = {
    'anthropic.claude-3-haiku-20240307-v1:0': 'Claude-3 Haiku',
    'anthropic.claude-3-opus-20240229-v1:0': 'Claude-3 Opus',
    'cohere.command-r-v1:0': 'Command-R',
    'cohere.command-r-plus-v1:0': 'Command-R+',
    'meta.llama3-8b-instruct-v1:0': 'LLaMA3 8B',
    'meta.llama3-70b-instruct-v1:0': 'LLaMA3 70B',
    'gpt-4o': 'GPT-4o',
    'gpt-4-0613': 'GPT-4',
    "gpt-35-turbo-1106": 'GPT-3.5-turbo'
}

# Mapping dictionary
author_mapping = {
    'simple': 'Basic',
    'Upadhyay': 'Rational',
    'thomas': 'Utility',
}

# Define field name mapping
field_name_mapping = {
    'brown_random_text_100': "RandP-100",
    'brown_random_text_with_query_100': "RandP-100+Q",
    'brown_random_text_with_query_words_100': "RandP-100+QWs",
    'brown_random_text_200': "RandP-200",
    'brown_random_text_with_query_200': "RandP-200+Q",
    'brown_random_text_with_query_words_200': "RandP-200+QWs",
    'brown_random_text_400': "RandP-400",
    'brown_random_text_with_query_400': "RandP-400+Q",
    'brown_random_text_with_query_words_400': "RandP-400+QWs",
}


def extract_info_from_filename(filename):
    parts = filename.split('_')
    model = model_name_mapping[parts[0]]
    prompt = author_mapping[parts[2]]
    return model, prompt


# Function to load data, process, and plot the relevance label distributions for each LLM-prompt
def process_and_plot(file_path):
    df = pd.read_csv(file_path)
    df['field'] = df['field'].map(field_name_mapping)
    df = df[df['field'].isin(field_name_mapping.values())]

    file = os.path.basename(file_path)
    model, prompt = extract_info_from_filename(file)

    df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
    df = df.dropna(subset=['O_score'])
    df['O_score'] = df['O_score'].astype(int)
    df['O_score'] = pd.Categorical(df['O_score'], categories=[0, 1, 2, 3], ordered=True)

    pivot_df = df.pivot_table(index='field', columns='O_score', values='id', aggfunc='size', fill_value=0)
    pivot_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [response_colors.get(x) for x in range(4)]
    pivot_df.plot(kind='bar', stacked=True, ax=ax, color=colors)


    ax.set_title(f"{model} - {prompt}")
    ax.set_xlabel("")
    ax.set_ylabel('Score Distribution')
    ax.set_xticklabels(pivot_df.index, rotation=45, ha='right')

    for container in ax.containers:
        for rect in container:
            height = rect.get_height()
            if height > 0.05:
                ax.text(rect.get_x() + rect.get_width() / 2, rect.get_y() + height / 2, f"{height:.2f}",
                        ha='center', va='center', color='black')

    legend_elements = [Patch(facecolor=response_colors[i], label=f'{i}') for i in response_colors]
    ax.legend(handles=legend_elements, title="Relevance Score", bbox_to_anchor=(0.5, -0.4), loc='upper center', ncol=4)

    plt.tight_layout()
    output_path = os.path.join("../tables_and_plots/", "random_passages_label_dist_" + model + "_" + prompt + '.pdf')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close(fig)


# Process each file in the directory
for file_name in os.listdir(folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(folder_path, file_name)
        process_and_plot(file_path)



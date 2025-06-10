import os
import pandas as pd
import krippendorff
from sklearn.metrics import cohen_kappa_score, mean_absolute_error, confusion_matrix
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Function to process model names
def process_model_name(model_name):
    if '.' in model_name and '-' in model_name:
        parts = model_name.split('.')
        main_part = '.'.join(parts[1:])
        model_name = main_part.split(':')[0]
    return model_name

# Mapping dictionary
author_mapping = {
    'simple': 'Basic',
    'Upadhyay': 'Rationale',
    'thomas': 'Utility',
}

# Ensure LaTeX rendering works properly
plt.rcParams.update({
    'text.usetex': True,
    'svg.fonttype': 'none',
    'text.latex.preamble': r'\usepackage{libertine}',
    'font.size': 18,
    'font.family': 'Linux Libertine',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'libertine',
    'mathtext.it': 'libertine:italic',
    'mathtext.bf': 'libertine:bold'
})

# Define custom color palette for models
model_palette = {
    'Claude-3 Haiku': 'skyblue',
    'Claude-3 Opus': 'royalblue',
    'Command-R': 'gold',
    'Command-R+': 'orange',
    'LLaMA3 8B': 'orchid',
    'LLaMA3 70B': 'blueviolet',
    'GPT-3.5-turbo': 'lightgreen',
    'GPT-4': 'y',
    'GPT-4o': 'darkgreen',
}

# Mapping dictionary
model_name_mapping = {
    'claude-3-haiku-20240307-v1': 'Claude-3 Haiku',
    'claude-3-opus-20240229-v1': 'Claude-3 Opus',
    'command-r-v1': 'Command-R',
    'command-r-plus-v1': 'Command-R+',
    'llama3-8b-instruct-v1': 'LLaMA3 8B',
    'llama3-70b-instruct-v1': 'LLaMA3 70B',
    'gpt-4o-2024-05-13': 'GPT-4o',
    'gpt-4': 'GPT-4',
    "gpt-35-turbo":'GPT-3.5-turbo',
}

# Define custom markers for prompts
prompt_markers = {
    'Basic': 's',
    'Rationale': 'd',
    'Utility': '*',
}

# Write results and Markdown sections to a Markdown file
results_directory = '../tables_and_plots'
os.makedirs(results_directory, exist_ok=True)

metrics = ['Krippendorff\'s Alpha (on Graded Labels)', "Cohen\'s Kappa (on Binary Labels)"]

directory = '../LLM_relevance_labelling/'

all_data = pd.DataFrame()
durations = {}
o_score_counts = {}

all_data = pd.DataFrame()
durations = {}
o_score_counts = {}

# The code for creating table 3
for file in tqdm(os.listdir(directory), desc="Processing files"):
    if file.endswith('.csv'):
        filepath = os.path.join(directory, file)
        prompt_name = file.split('_')[2]  # Extract prompt name from the file name

        try:
            df = pd.read_csv(filepath)
            df = df.dropna(subset=['O_score'])

            # convert into a number
            df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
            df = df.dropna(subset=['O_score'])

            # convert to a binary scale
            df['nist_judgment_binary'] = (df['nist_judgment'] >= 2).astype(int)
            df['O_score_binary'] = (df['O_score'] >= 2).astype(int)

            # Add model and prompt_name to the DataFrame
            df['model'] = df['model'].apply(process_model_name)
            model_name = df['model'].iloc[0]
            df['prompt'] = prompt_name
            df['prompt'] = df['prompt'].replace(author_mapping)
            prompt_name = df['prompt'].iloc[0]

            # add the model prompt data to the main data having all model data
            all_data = pd.concat([all_data, df], ignore_index=True)

        except Exception as e:
            print(f"Error processing file {file}: {e}")

all_data['model'] = all_data['model'].map(model_name_mapping)

model_prompt_pairs = all_data[['model', 'prompt']].drop_duplicates()

results = []
markdown_sections = []

for _, row in tqdm(model_prompt_pairs.iterrows(), desc="Calculating metrics", total=model_prompt_pairs.shape[0]):
    model = row['model']
    prompt_name = row['prompt']

    model_data = all_data[(all_data['model'] == model) & (all_data['prompt'] == prompt_name)].copy()

    row_count = len(model_data)


    kappa = cohen_kappa_score(model_data['nist_judgment_binary'], model_data['O_score_binary'])

    alpha_data = model_data[['nist_judgment', 'O_score']].to_numpy().T
    alpha = krippendorff.alpha(reliability_data=alpha_data, level_of_measurement='ordinal')

    mae_graded = mean_absolute_error(model_data['nist_judgment'], model_data['O_score'])
    mae_binary = mean_absolute_error(model_data['nist_judgment_binary'], model_data['O_score_binary'])

    average_cost = model_data['cost'].mean()
    cost_sum = model_data['cost'].sum()

    o_score_binary_1_ratio = (model_data['O_score_binary'] == 1).mean()

    accuracy_bin = (model_data['nist_judgment_binary'] == model_data['O_score_binary']).mean()
    accuracy = (model_data['nist_judgment_binary'] == model_data['O_score_binary']).mean()

    cm_bin = confusion_matrix(model_data['nist_judgment_binary'], model_data['O_score_binary'], labels=[0,1])
    precision_label_0 = cm_bin[0, 0] / (cm_bin[0, 0] + cm_bin[1, 0]) if (cm_bin[0, 0] + cm_bin[1, 0]) > 0 else 0
    precision_label_1 = cm_bin[1, 1] / (cm_bin[0, 1] + cm_bin[1, 1]) if (cm_bin[0, 1] + cm_bin[1, 1]) > 0 else 0
    total_labels = np.sum(cm_bin)
    graded_cm = confusion_matrix(model_data['nist_judgment'], model_data['O_score'])

    # Store results
    results.append(
        [model, prompt_name, round(kappa, 2), round(alpha, 2), round(mae_binary, 2), round(mae_graded, 2),
         round(average_cost, 8),round(average_cost*10000, 2), round(cost_sum, 2), row_count, round(accuracy, 2),
         round(o_score_binary_1_ratio, 2),
         round(precision_label_0, 2), round(precision_label_1, 2),
         total_labels,(1-(total_labels/4222))*100])

results_df = pd.DataFrame(results, columns=['Model', 'Prompt', 'Cohen\'s Kappa (on Binary Labels)',
                                            'Krippendorff\'s Alpha (on Graded Labels)', 'MAE (Binary)',
                                            'MAE (Graded)', 'Cost/Label', 'Cost/10000Labels', 'Total Cost',
                                            '#Parsable Scores', 'Accuracy', 'P(Label=1)',
                                            'Prec(Label=0)', 'Prec(Label=1)', "# Valid Scores", "missing %"])
model_order_1 = list(model_palette.keys())
author_order_1 = list(author_mapping.values())
tbl_results = results_df

tbl_results['Model'] = pd.Categorical(tbl_results['Model'], categories=model_order_1, ordered=True)
tbl_results['Prompt'] = pd.Categorical(tbl_results['Prompt'], categories=author_order_1, ordered=True)

columns_to_print = ['Model', 'Prompt', 'Cohen\'s Kappa (on Binary Labels)', 'Krippendorff\'s Alpha (on Graded Labels)', 'MAE (Binary)', 'MAE (Graded)', 'Accuracy',  'Prec(Label=0)',
                    'Prec(Label=1)', 'P(Label=1)']

# Now sort by these columns
results_df = results_df.sort_values(by=['Model', 'Prompt'])


with open(os.path.join(results_directory, f'table_3_agreement_metrics.md'), 'w') as f:
    f.write("\nMarkdown:\n")
    f.write(results_df[columns_to_print].to_markdown(index=False))
    f.write("\nLaTeX:\n")
    f.write(results_df[columns_to_print].to_latex(index=False))


# The code for plotting Figure 2
marker_size = 200

for metric in metrics:
    plt.figure(figsize=(15, 9))
    ax = plt.gca()

    # Create the scatter plot with custom colors and markers
    scatter = sns.scatterplot(data=results_df, x='Cost/10000Labels', y=metric, hue='Model', style='Prompt', s=marker_size,
                              palette=model_palette, markers=prompt_markers, ax=ax)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)

    # Increase the size of the utility prompt
    for _, row in results_df[results_df['Prompt'] == 'Utility'].iterrows():
        plt.scatter(row['Cost/10000Labels'], row[metric], color=model_palette[row['Model']], marker='*', s=250,
                    label='_nolegend_')

    max_cost_label = results_df['Cost/10000Labels'].max()

    # Increase the maximum limit by a small percentage for a buffer
    extended_max = max_cost_label * 1.1

    plt.xscale('log')
    plt.xlim(right=extended_max)
    def format_func(value, tick_number):
        # Use the exponential of the value because the axis is in log scale
        return f'{int(value)}'

    ax.xaxis.set_major_formatter(FuncFormatter(format_func))

    plt.xlabel('Cost in USD per 10K Labels (Logarithmic Scale)\nbetter ←        → worse', fontsize=20)
    plt.ylabel(metric, fontsize=20)
    ax.tick_params(axis='both', which='major', labelsize=15)

    if metric == "Cohen\'s Kappa (on Binary Labels)":
        plt.ylim(-0.1, 0.75)
        plt.fill_betweenx(y=[0.24, 0.58], x1=plt.xlim()[0], x2=plt.xlim()[1], color='lightgray', alpha=0.2)
        plt.text(2.8, 0.585, "Damessie et al.'s Bronze Judges Upper Bound - TREC 2004 Robust", color="black", fontsize=20)
        plt.text(2.8, 0.245, "Damessie et al.'s Bronze Judges Lower Bound - TREC 2004 Robust", color="black", fontsize=20)

        plt.axhline(y=0.52, color='darkgray', linestyle='--')
        plt.text(2.8, 0.525, "Cormack et al.'s Silver Judges - TREC-6 ad-hoc", color='black', fontsize=20)
        plt.axhline(y=0.41, color='darkgray', linestyle='--')
        plt.text(2.8, 0.415, "Hersh et al.'s Silver Judges - OHSUMED", color='black', fontsize=20)

    else:
        plt.ylim(-0.1, 0.75)
        plt.axhline(y=0.41, color='gray', linestyle=':')
        plt.axhline(y=0.69, color='gray', linestyle=':')
        plt.fill_betweenx(y=[0.41, 0.69], x1=plt.xlim()[0], x2=plt.xlim()[1], color='lightgray', alpha=0.2)
        plt.text(3, 0.695, "Damessie et al.'s Bronze Judges Upper Bound - TREC 2004 Robust", color="black", fontsize=17)
        plt.text(3, 0.415, "Damessie et al.'s Bronze Judges Lower Bound - TREC 2004 Robust", color="black", fontsize=17)

    handles, labels = scatter.get_legend_handles_labels()

    model_handles = [handle for handle, label in zip(handles, labels) if label in model_palette]
    model_labels = [label for label in labels if label in model_palette]

    filtered_labels = [label for label in labels if label in prompt_markers]

    label_to_handle = {label: handle for handle, label in zip(handles, labels) if label in prompt_markers}

    sorted_labels = sorted(filtered_labels, key=lambda x: list(prompt_markers.keys()).index(x))

    prompt_labels = sorted_labels
    prompt_handles = [label_to_handle[label] for label in sorted_labels]

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    sorted_model_handles_labels = sorted(zip(model_handles, model_labels),
                                         key=lambda hl: list(model_palette.keys()).index(hl[1]) if hl[1] in model_palette else float('inf'))

    sorted_model_handles, sorted_model_labels = zip(*sorted_model_handles_labels)
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # model_legend = ax.legend(sorted_model_handles, sorted_model_labels, title='Model', loc='upper left',
    #                          bbox_to_anchor=(1.01, 1), borderaxespad=0., ncol=2, handletextpad=0.1,
    #                      framealpha=0.5)

    # ax.add_artist(model_legend)
    #
    # ax.legend(prompt_handles, prompt_labels, title='Prompt', loc='upper left', handletextpad=0.1, bbox_to_anchor=(1.01, 0.7),
    #           borderaxespad=0., ncol=1, framealpha=0.5)
    # plt.tight_layout(rect=[0, 0, 20, 1])

    plot_directory = '../tables_and_plots'
    plt.tight_layout()
    os.makedirs(plot_directory, exist_ok=True)
    plt.savefig(os.path.join(plot_directory, f'fig_2_agreement_vs_cost_{metric}_plot.pdf'))
    plt.close()


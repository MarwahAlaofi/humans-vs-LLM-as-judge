from sklearn.metrics import cohen_kappa_score, mean_absolute_error, confusion_matrix
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import kendalltau, pearsonr
from tqdm import tqdm

def mean_absolute_error_with_zero_true(y_pred):
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_pred))

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
    'UpadhyayWithNoNumberExample': 'Rationale',
    'thomas': 'Utility',
}

# Mapping dictionary
model_name_mapping_2 = {
    'claude-3-haiku-20240307-v1': 'Claude-3 Haiku',
    'claude-3-opus-20240229-v1': 'Claude-3 Opus',
    'command-r-v1': 'Command-R',
    'command-r-plus-v1': 'Command-R+',
    'llama3-8b-instruct-v1': 'LLaMA3 8B',
    'llama3-70b-instruct-v1': 'LLaMA3 70B',
    'gpt-35-turbo-1106': 'GPT-3.5-turbo',
    'gpt-4-0613': 'GPT-4',
    'gpt-4o': 'GPT-4o',
}

# Ensure LaTeX rendering works properly
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
    'GPT-4o-mini': 'aquamarine',
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
    'gpt-4o-mini': 'GPT-4o-mini',
}
# Define custom markers for prompts
prompt_markers = {
    'Basic': 's',
    'Rationale': 'd',
    'Utility': '*',
}


all_df = pd.DataFrame(columns=['model', 'prompt', 'field', 'O_score'])

# Directory setup
random_dir = '../random_passage_with_query_injection_files'
nonrelevant_dir = '../non-relevant_with_query_injection_files'
random_score_dir = '../random_passage_with_score_injection_files'
nonrelevant_score_dir = '../non-relevant_with_score_description_injection_files'

gullibility_results = {}

# get MAE data
for file_name in tqdm(os.listdir(random_dir), desc="Processing random files"):
    if file_name.endswith('.csv'):
        file_path = os.path.join(random_dir, file_name)
        model_name = file_name.split("_")[0]
        model_name = process_model_name(model_name)
        prompt_name = file_name.split("_")[2]

        df = pd.read_csv(file_path)
        df = df.dropna(subset=['O_score'])
        df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
        df = df.dropna(subset=['O_score'])

        for field_suffix in ['query_100', 'query_words_100']:
            field_name = f'brown_random_text_with_{field_suffix}'
            filtered_data = df[df['field'] == field_name].copy()

            filtered_data["model"] = model_name
            filtered_data["prompt"] = prompt_name
            dataset = "dl21" if "dl21" in file_name else "dl22"

            field_name = "RandP+QWs" if "words" in field_name else "RandP+Q"
            filtered_data["field"] = field_name
            all_df = pd.concat([all_df, filtered_data[['model', 'prompt', 'field', 'O_score']]], ignore_index=True)

            errors = filtered_data['O_score']
            # if len(errors) < 53:
            #     print(f"{model_name}-{prompt_name}-{field_name}:{len(errors)}")
            mae = mean_absolute_error_with_zero_true(errors)
            gullibility_results[(model_name, prompt_name, field_name, dataset)] = (mae,errors)

for file_name in tqdm(os.listdir(nonrelevant_dir), desc="Processing non-relevant files"):
    if file_name.endswith('.csv'):
        file_path = os.path.join(nonrelevant_dir, file_name)
        model_name = file_name.split("_")[0]
        model_name = process_model_name(model_name)

        prompt_name = file_name.split("_")[2]

        df = pd.read_csv(file_path)
        df = df.dropna(subset=['O_score'])
        df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
        df = df.dropna(subset=['O_score'])

        for field in ['passage_and_query', 'passage_and_query_words']:
            filtered_data = df[df['field'] == field].copy()

            filtered_data["model"] = model_name
            filtered_data["prompt"] = prompt_name
            dataset = "dl21" if "dl21" in file_name else "dl22"

            field_name = "NonRelP+QWs" if "words" in field else "NonRelP+Q"
            filtered_data["field"] = field_name
            all_df = pd.concat([all_df, filtered_data[['model', 'prompt', 'field', 'O_score']]], ignore_index=True)

            errors = filtered_data['O_score']
            # if len(errors) < 25:
            #     print(f"{model_name}-{prompt_name}-{field_name}:{len(errors)}")
            mae = mean_absolute_error_with_zero_true(errors)
            gullibility_results[(model_name, prompt_name, field_name, dataset)] = (mae,errors)

for file_name in tqdm(os.listdir(random_score_dir), desc="Processing random score files"):
    if file_name.endswith('.csv'):
        file_path = os.path.join(random_score_dir, file_name)
        model_name = file_name.split("_")[0]
        model_name = process_model_name(model_name)

        prompt_name = file_name.split("_")[2]
        dataset = "dl21" if "dl21" in file_name else "dl22"

        df = pd.read_csv(file_path)
        df = df.dropna(subset=['O_score'])
        df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
        df = df.dropna(subset=['O_score'])

        # filtered_data = df[df['field'] == "brown_random_text_100"]

        df["model"] = model_name
        df["prompt"] = prompt_name
        field_name = "RandP+Inst"
        df["field"] = field_name
        all_df = pd.concat([all_df, df[['model', 'prompt', 'field', 'O_score']]], ignore_index=True)

        errors = df['O_score']
        # if len(errors) < 53:
        #     print(f"{model_name}-{prompt_name}-{field_name}:{len(errors)}")
        mae = mean_absolute_error_with_zero_true(errors)
        gullibility_results[(model_name, prompt_name, field_name, dataset)] = (mae,errors)

for file_name in tqdm(os.listdir(nonrelevant_score_dir), desc="Processing non-relevant score files"):
    if file_name.endswith('.csv'):
        file_path = os.path.join(nonrelevant_score_dir, file_name)
        model_name = file_name.split("_")[0]
        model_name = process_model_name(model_name)

        prompt_name = file_name.split("_")[2]
        dataset = "dl21" if "dl21" in file_name else "dl22"

        df = pd.read_csv(file_path)
        df = df.dropna(subset=['O_score'])
        df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
        df = df.dropna(subset=['O_score'])

        # filtered_data = df[df['field'] == "passage"]

        df["model"] = model_name
        df["prompt"] = prompt_name
        field_name = "NonRelP+Inst"
        df["field"] = field_name
        all_df = pd.concat([all_df, df[['model', 'prompt', 'field', 'O_score']]], ignore_index=True)

        errors = df['O_score']
        # if len(errors) < 25:
        #     print(f"{model_name}-{prompt_name}-{field_name}:{len(errors)}")
        mae = mean_absolute_error_with_zero_true(errors)
        gullibility_results[(model_name, prompt_name, field_name, dataset)] = (mae,errors)

# Convert the results to a DataFrame
df_results = pd.DataFrame.from_dict(gullibility_results, orient='index', columns=['MAE', 'SCORES'])
df_results.index = pd.MultiIndex.from_tuples(df_results.index, names=['Model', 'Prompt', 'Test', "Dataset"])
df_results = df_results.reset_index()

df_results['Prompt'] = df_results['Prompt'].map(author_mapping)
test_order = ["RandP+Q", "RandP+QWs", "NonRelP+Q", "NonRelP+QWs", "RandP+Inst", "NonRelP+Inst"]
df_results['Model'] = df_results['Model'].map(model_name_mapping_2)

model_test_avg = df_results.groupby(['Model', 'Test'])['MAE'].mean().reset_index()
model_avg = df_results.groupby(['Model'])['MAE'].mean().reset_index()

keywordstuffing_df_results = df_results[df_results['Test'].isin(["RandP+QWs","RandP+Q","NonRelP+Q","NonRelP+QWs"])]
inststuffing_df_results = df_results[df_results['Test'].isin(["RandP+Inst","NonRelP+Inst"])]

model_prompt_keyword_stuffing_avg = keywordstuffing_df_results.groupby(['Model', 'Prompt'])['MAE'].mean().reset_index()
model_prompt_instruction_stuffing_avg = inststuffing_df_results.groupby(['Model', 'Prompt'])['MAE'].mean().reset_index()

model_keyword_stuffing_avg = keywordstuffing_df_results.groupby(['Model'])['MAE'].mean().reset_index()
model_instruction_stuffing_avg = inststuffing_df_results.groupby(['Model'])['MAE'].mean().reset_index()

# Path to the directory with CSV and TXT files
directory = '../LLM_relevance_labelling'
all_data = pd.DataFrame()
durations = {}
o_score_counts = {}

# get kappa
for file in tqdm(os.listdir(directory), desc="Processing files"):
    if file.endswith('.csv'):
        filepath = os.path.join(directory, file)
        prompt_name = file.split('_')[2]  # Extract prompt name from the file name

        try:
            df = pd.read_csv(filepath)

            df = df.dropna(subset=['O_score'])

            df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
            df = df.dropna(subset=['O_score'])

            df['nist_judgment'] = np.round(df['nist_judgment']).astype(int)
            df['O_score'] = np.round(df['O_score']).astype(int)

            # convert to a binary scale
            df['nist_judgment_binary'] = (df['nist_judgment'] > 1).astype(int)
            df['O_score_binary'] = (df['O_score'] > 1).astype(int)

            df['model'] = df['model'].apply(process_model_name)
            model_name = df['model'].iloc[0]
            df['prompt'] = prompt_name
            df['prompt'] = df['prompt'].replace(author_mapping)
            prompt_name = df['prompt'].iloc[0]
            all_data = pd.concat([all_data, df], ignore_index=True)

        except Exception as e:
            print(f"Error processing file {file}: {e}")

all_data['model'] = all_data['model'].map(model_name_mapping)
model_prompt_pairs = all_data[['model', 'prompt']].drop_duplicates()
results = []

for _, row in tqdm(model_prompt_pairs.iterrows(), desc="Calculating metrics", total=model_prompt_pairs.shape[0]):
    model = row['model']
    prompt_name = row['prompt']
    model_data = all_data[(all_data['model'] == model) & (all_data['prompt'] == prompt_name)].copy()

    row_count = len(model_data)
    kappa = cohen_kappa_score(model_data['nist_judgment_binary'], model_data['O_score_binary'])

    results.append(
        [model, prompt_name, round(kappa, 2)])

results_df = pd.DataFrame(results, columns=['Model', 'Prompt', 'Cohen\'s Kappa (on Binary Labels)'])
results_df = results_df.sort_values(by='Model')

results_df_with_mae_instr = pd.merge(results_df, model_prompt_instruction_stuffing_avg, on=['Model', 'Prompt'], how='left')

marker_size = 200

plt.figure(figsize=(12, 8))
ax = plt.gca()  # Get the current Axes instance

scatter = sns.scatterplot(data=results_df_with_mae_instr, y='Cohen\'s Kappa (on Binary Labels)', x="MAE", hue='Model', style='Prompt', s=marker_size,
                          palette=model_palette, markers=prompt_markers, ax=ax)
plot_data = results_df_with_mae_instr[['Cohen\'s Kappa (on Binary Labels)', 'MAE', 'Model', 'Prompt']]
print(plot_data)
mae = results_df_with_mae_instr['MAE']
cohen_kappa = results_df_with_mae_instr['Cohen\'s Kappa (on Binary Labels)']
pearson_corr, _ = pearsonr(mae, cohen_kappa)

# Print the Pearson correlation
print(f"Instruction stuffing: Pearson correlation between MAE and Cohen's Kappa: {pearson_corr}")

for _, row in results_df_with_mae_instr[results_df_with_mae_instr['Prompt'] == 'Utility'].iterrows():
    plt.scatter(row["MAE"],  row["Cohen\'s Kappa (on Binary Labels)"], color=model_palette[row['Model']], marker='*', s=250,
                label='_nolegend_')

max_mae = results_df_with_mae_instr['MAE'].max()

extended_max = max_mae * 1.1

plt.ylim(0,0.6)
plt.xlabel('MAE (Instruction Injection Gullibility)\nbetter ←        → worse', fontsize=20)
plt.ylabel("Cohen\'s Kappa (on Binary Labels)",fontsize=20)
ax.grid(True, which='major', axis='y', linestyle='solid', linewidth=0.4,
        color='darkgray')
ax.grid(True, which='major', axis='x', linestyle=':', linewidth=0.3,
        color='black')

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
                                     key=lambda hl: list(model_palette.keys()).index(hl[1]) if hl[
                                                                                                   1] in model_palette else float(
                                         'inf'))

sorted_model_handles, sorted_model_labels = zip(*sorted_model_handles_labels)

model_legend = ax.legend(sorted_model_handles, sorted_model_labels, title='Model', loc='upper right',
                         bbox_to_anchor=(1, 0.99), borderaxespad=0., ncol=1)
ax.add_artist(model_legend)
ax.legend(prompt_handles, prompt_labels, title='Prompt', loc='upper right', bbox_to_anchor=(1, 0.55),
          borderaxespad=0., ncol=1)

plot_directory = '../tables_and_plots'
plt.tight_layout()
os.makedirs(plot_directory, exist_ok=True)
plt.savefig(os.path.join(plot_directory, f'k_against_mae_of_inst_plot.pdf'))
plt.close()

results_df_with_mae_keyword_Stuffing = pd.merge(results_df, model_prompt_keyword_stuffing_avg, on=['Model', 'Prompt'], how='left')
plt.figure(figsize=(12, 8))
ax = plt.gca()
scatter = sns.scatterplot(data=results_df_with_mae_keyword_Stuffing, y='Cohen\'s Kappa (on Binary Labels)', x="MAE", hue='Model', style='Prompt', s=marker_size,
                          palette=model_palette, markers=prompt_markers, ax=ax)
plot_data = results_df_with_mae_keyword_Stuffing[['Cohen\'s Kappa (on Binary Labels)', 'MAE', 'Model', 'Prompt']]
print(plot_data)
mae = results_df_with_mae_keyword_Stuffing['MAE']
cohen_kappa = results_df_with_mae_keyword_Stuffing['Cohen\'s Kappa (on Binary Labels)']

corr, _ = pearsonr(mae, cohen_kappa)

print(f"Keyword stuffing: Pearson correlation between MAE and Cohen's Kappa: {corr}")

for _, row in results_df_with_mae_keyword_Stuffing[results_df_with_mae_keyword_Stuffing['Prompt'] == 'Utility'].iterrows():
    plt.scatter(row["MAE"],  row["Cohen\'s Kappa (on Binary Labels)"], color=model_palette[row['Model']], marker='*', s=250,
                label='_nolegend_')
def format_func(value, tick_number):
    return f'{int(value)}'

plt.ylim(0,0.6)
plt.xlabel('MAE (Keyword Stuffing Gullibility)\nbetter ←        → worse', fontsize=20)
plt.ylabel("Cohen\'s Kappa (on Binary Labels)",fontsize=20)
ax.grid(True, which='major', axis='y', linestyle='solid', linewidth=0.4,
        color='darkgray')
ax.grid(True, which='major', axis='x', linestyle=':', linewidth=0.3,
        color='black')

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
                                     key=lambda hl: list(model_palette.keys()).index(hl[1]) if hl[
                                                                                                   1] in model_palette else float(
                                         'inf'))

sorted_model_handles, sorted_model_labels = zip(*sorted_model_handles_labels)

model_legend = ax.legend(sorted_model_handles, sorted_model_labels, title='Model', loc='upper right',
                         bbox_to_anchor=(1, 0.99), borderaxespad=0., ncol=1, fontsize=16,title_fontsize=17)
ax.add_artist(model_legend)
ax.legend(prompt_handles, prompt_labels, title='Prompt', loc='upper right', bbox_to_anchor=(1, 0.50),
          borderaxespad=0., ncol=1,fontsize=16,title_fontsize=17)

plot_directory = '../tables_and_plots'
plt.tight_layout()
os.makedirs(plot_directory, exist_ok=True)
plt.savefig(os.path.join(plot_directory, f'k_against_mae_of_keyword_stuffing_plot.pdf'))
plt.close()


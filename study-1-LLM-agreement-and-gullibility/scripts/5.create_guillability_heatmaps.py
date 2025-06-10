import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

def mean_absolute_error_with_zero_true(y_pred):
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_pred))

plt.rcParams.update({
    'text.usetex': True,
    'svg.fonttype': 'none',
    'text.latex.preamble': r'\usepackage{libertine}',
    'font.size': 12,
    'font.family': 'Linux Libertine',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'libertine',
    'mathtext.it': 'libertine:italic',
    'mathtext.bf': 'libertine:bold'
})


# Mapping dictionary
model_name_mapping = {
    'anthropic.claude-3-haiku-20240307-v1:0': 'Claude-3 Haiku',
    'anthropic.claude-3-opus-20240229-v1:0': 'Claude-3 Opus',
    'cohere.command-r-v1:0': 'Command-R',
    'cohere.command-r-plus-v1:0': 'Command-R+',
    'meta.llama3-8b-instruct-v1:0': 'LLaMA3 8B',
    'meta.llama3-70b-instruct-v1:0': 'LLaMA3 70B',
    'gpt-35-turbo-1106': 'GPT-3.5-turbo',
    'gpt-4-0613': 'GPT-4',
    'gpt-4o': 'GPT-4o',
}

all_df = pd.DataFrame(columns=['model', 'prompt', 'field', 'O_score'])

# Directory setup
random_dir = '../random_passage_with_query_injection_files'
nonrelevant_dir = '../non-relevant_with_query_injection_files'
random_score_dir = '../random_passage_with_score_injection_files'
nonrelevant_score_dir = '../non-relevant_with_score_description_injection_files'

# processing query injection tests for random passages for each LLM
gullibility_results = {}
for file_name in tqdm(os.listdir(random_dir), desc="Processing random files"):
    if file_name.endswith('.csv'):
        file_path = os.path.join(random_dir, file_name)
        model_name = file_name.split("_")[0]
        prompt_name = file_name.split("_")[2]
        dataset = "dl21" if "dl21" in file_name else "dl22"
        if prompt_name == "UpadhyayWithNoNumberExample" or prompt_name == "Upadhyay":
            prompt_name = "Upadhyay"

        df = pd.read_csv(file_path)
        df = df.dropna(subset=['O_score'])
        df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
        df = df.dropna(subset=['O_score'])

        for field_suffix in ['query_100', 'query_words_100']:
            field_name = f'brown_random_text_with_{field_suffix}'
            filtered_data = df[df['field'] == field_name].copy()

            filtered_data["model"] = model_name
            filtered_data["prompt"] = prompt_name
            field_name = "RandP+QWs" if "words" in field_name else "RandP+Q"
            filtered_data["field"] = field_name
            all_df = pd.concat([all_df, filtered_data[['model', 'prompt', 'field', 'O_score']]], ignore_index=True)

            scores = filtered_data['O_score']
            # if len(scores) < 53:
            #     print(f"{model_name}-{prompt_name}-{field_name}:{len(scores)}")
            mae = mean_absolute_error_with_zero_true(scores)
            gullibility_results[(model_name, prompt_name, field_name, dataset)] = mae

# processing query injection tests for non-relevant passages for each LLM
for file_name in tqdm(os.listdir(nonrelevant_dir), desc="Processing non-relevant files"):
    if file_name.endswith('.csv'):
        file_path = os.path.join(nonrelevant_dir, file_name)
        model_name = file_name.split("_")[0]
        prompt_name = file_name.split("_")[2]

        dataset = "dl21" if "dl21" in file_name else "dl22"
        prompt_name = file_name.split("_")[2]

        df = pd.read_csv(file_path)
        df = df.dropna(subset=['O_score'])
        df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
        df = df.dropna(subset=['O_score'])

        for field in ['passage_and_query', 'passage_and_query_words']:
            filtered_data = df[df['field'] == field].copy()

            filtered_data["model"] = model_name
            filtered_data["prompt"] = prompt_name
            if prompt_name == "UpadhyayWithNoNumberExample" or prompt_name == "Upadhyay":
                prompt_name = "Upadhyay"

            field_name = "NonRelP+QWs" if "words" in field else "NonRelP+Q"
            filtered_data["field"] = field_name
            all_df = pd.concat([all_df, filtered_data[['model', 'prompt', 'field', 'O_score']]], ignore_index=True)
            scores = filtered_data['O_score']

            # if len(scores) < 25:
            #     print(f"{model_name}-{prompt_name}-{field_name}:{len(scores)}")
            mae = mean_absolute_error_with_zero_true(scores)
            gullibility_results[(model_name, prompt_name, field_name, dataset)] = mae

# processing instruction injection tests for random passages for each LLM
for file_name in tqdm(os.listdir(random_score_dir), desc="Processing random passages instruction injection files"):
    if file_name.endswith('.csv'):

        file_path = os.path.join(random_score_dir, file_name)
        model_name = file_name.split("_")[0]

        dataset = "dl21" if "dl21" in file_name else "dl22"

        prompt_name = file_name.split("_")[2]
        if prompt_name == "UpadhyayWithNoNumberExample" or prompt_name == "Upadhyay":
            prompt_name = "Upadhyay"

        df = pd.read_csv(file_path)
        df = df.dropna(subset=['O_score'])
        df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
        df = df.dropna(subset=['O_score'])

        df["model"] = model_name
        df["prompt"] = prompt_name
        field_name = "RandP+Inst"
        df["field"] = field_name
        all_df = pd.concat([all_df, df[['model', 'prompt', 'field', 'O_score']]], ignore_index=True)

        scores = df['O_score']
        # if len(scores) < 53:
        #     print(f"{model_name}-{prompt_name}-{field_name}:{len(scores)}")
        mae = mean_absolute_error_with_zero_true(scores)
        gullibility_results[(model_name, prompt_name, field_name, dataset)] = mae

for file_name in tqdm(os.listdir(nonrelevant_score_dir), desc="Processing non-relevant passages instruction injection files"):
    if file_name.endswith('.csv'):

        file_path = os.path.join(nonrelevant_score_dir, file_name)
        model_name = file_name.split("_")[0]

        prompt_name = file_name.split("_")[2]
        dataset = "dl21" if "dl21" in file_name else "dl22"

        if prompt_name == "UpadhyayWithNoNumberExample" or prompt_name == "Upadhyay":
            prompt_name = "Upadhyay"

        df = pd.read_csv(file_path)
        df = df.dropna(subset=['O_score'])
        df['O_score'] = pd.to_numeric(df['O_score'], errors='coerce')
        df = df.dropna(subset=['O_score'])

        df["model"] = model_name
        df["prompt"] = prompt_name
        field_name = "NonRelP+Inst"
        df["field"] = field_name
        all_df = pd.concat([all_df, df[['model', 'prompt', 'field', 'O_score']]], ignore_index=True)

        scores = df['O_score']
        # if len(scores) < 25:
        #     print(f"{model_name}-{prompt_name}-{field_name}:{len(scores)}")
        mae = mean_absolute_error_with_zero_true(scores)
        gullibility_results[(model_name, prompt_name, field_name, dataset)] = mae

# Convert the results to a DataFrame
df_results = pd.DataFrame.from_dict(gullibility_results, orient='index', columns=['MAE'])
df_results.index = pd.MultiIndex.from_tuples(df_results.index, names=['Model', 'Prompt', 'Test', "Dataset"])
df_results = df_results.reset_index()
pd.set_option('display.max_rows', None)

print(df_results[df_results["Prompt"] == "Upadhyay"].sort_values(by=["Model","Prompt","Test"]))
test_order = ["RandP+Q", "RandP+QWs", "NonRelP+Q", "NonRelP+QWs", "RandP+Inst", "NonRelP+Inst"]

keyword_stuffing_test_order = ["RandP+Q", "RandP+QWs", "NonRelP+Q", "NonRelP+QWs"]
instruction_injection_test_order = ["RandP+Inst", "NonRelP+Inst"]

df_results['Model'] = df_results['Model'].map(model_name_mapping)

keywordstuffing_df_results = df_results[df_results['Test'].isin(keyword_stuffing_test_order)]
inststuffing_df_results = df_results[df_results['Test'].isin(instruction_injection_test_order)]

model_prompt_keyword_stuffing_avg = keywordstuffing_df_results.groupby(['Model', 'Prompt'])['MAE'].mean().reset_index()
model_prompt_instruction_stuffing_avg = inststuffing_df_results.groupby(['Model', 'Prompt'])['MAE'].mean().reset_index()
model_prompt_test_keyword_stuffing_avg = keywordstuffing_df_results.groupby(['Model', 'Prompt','Test'])['MAE'].mean().reset_index()

model_test_keyword_stuffing_avg = keywordstuffing_df_results.groupby(['Model', 'Test'])['MAE'].mean().reset_index()
model_test_instruction_stuffing_avg = inststuffing_df_results.groupby(['Model', 'Test'])['MAE'].mean().reset_index()

print("model_prompt_keyword_stuffing_avg")
print(model_prompt_keyword_stuffing_avg)
print("model_prompt_test_keyword_stuffing_avg")
print(model_prompt_test_keyword_stuffing_avg)

print("model_prompt_instruction_stuffing_avg")
print(model_prompt_instruction_stuffing_avg)
print("############################")
print("model_test_keyword_stuffing_avg")
print(model_test_keyword_stuffing_avg)
print("model_test_instruction_stuffing_avg")
print(model_test_instruction_stuffing_avg)

model_order = [model_name_mapping[key] for key in model_name_mapping]

model_test_keyword_stuffing_avg['Test'] = pd.Categorical(model_test_keyword_stuffing_avg['Test'], categories=keyword_stuffing_test_order, ordered=True)

model_test_keyword_stuffing_avg['Model'] = pd.Categorical(model_test_keyword_stuffing_avg['Model'], categories=model_order, ordered=True)

df_sorted = model_test_keyword_stuffing_avg.sort_values(['Model', 'Test'])

pivot_table = df_sorted.pivot(index="Model", columns="Test", values="MAE")

pivot_table = pivot_table[keyword_stuffing_test_order]

plt.figure(figsize=(7, 4))
ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", linewidths=0.5, cmap="OrRd", linecolor='white', annot_kws={"size": 9}, cbar_kws={'label': 'MAE', 'shrink': 0.5}, vmin=0, vmax=1.6)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=13.5)
plt.subplots_adjust(top=0.97, bottom=0.1, left=0.21, right=0.99)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=13, rotation=0)

plt.ylabel('')
plt.xlabel('')


plt.savefig('../tables_and_plots/mae_for_models_test_keyword_heatmap.pdf')


model_test_instruction_stuffing_avg['Test'] = pd.Categorical(model_test_instruction_stuffing_avg['Test'], categories=instruction_injection_test_order, ordered=True)

model_test_instruction_stuffing_avg['Model'] = pd.Categorical(model_test_instruction_stuffing_avg['Model'], categories=model_order, ordered=True)

df_sorted = model_test_instruction_stuffing_avg.sort_values(['Model', 'Test'])

pivot_table = df_sorted.pivot(index="Model", columns="Test", values="MAE")

pivot_table = pivot_table[instruction_injection_test_order]

plt.figure(figsize=(5.5, 4))
ax = sns.heatmap(pivot_table, annot=True, fmt=".2f", linewidths=0.5, cmap="OrRd", linecolor='white', annot_kws={"size": 9}, cbar_kws={'label': 'MAE', 'shrink': 0.5}, vmin=0, vmax=1.6)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
plt.subplots_adjust(top=0.97, bottom=0.1, left=0.22, right=0.99)
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=8)

ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=0)

plt.ylabel('')
plt.xlabel('')

plt.savefig('../tables_and_plots/mae_for_models_test_inst_heatmap.pdf')

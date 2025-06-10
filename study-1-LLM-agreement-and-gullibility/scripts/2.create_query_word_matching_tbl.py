import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import os
from tqdm import tqdm


# Function to process model names
def process_model_name(model_name):
    if '.' in model_name and '-' in model_name:
        parts = model_name.split('.')
        main_part = '.'.join(parts[1:])
        model_name = main_part.split(':')[0]
    return model_name


model_name_mapping = {
    'claude-3-haiku-20240307-v1': 'Claude-3 Haiku',
    'claude-3-opus-20240229-v1': 'Claude-3 Opus',
    'command-r-v1': 'Command-R',
    'command-r-plus-v1': 'Command-R+',
    'llama3-8b-instruct-v1': 'LLaMA3 8B',
    'llama3-70b-instruct-v1': 'LLaMA3 70B',
    'gpt-4o': 'GPT-4o',
    'gpt-4-0613': 'GPT-4',
    "gpt-35-turbo-1106": 'GPT-3.5-turbo'
}

# Mapping dictionary
author_mapping = {
    'simple': 'Basic',
    'Upadhyay': 'Upadhyay et al.',
    'thomas': 'Thomas et al.',
}


def read_and_process_file(filename, col1='nist_judgment', col2='O_score'):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filename)

    # Extract model and prompt from the filename
    filename_parts = os.path.basename(filename).split('_')
    df['model'] = filename_parts[0]
    df['prompt'] = filename_parts[2]

    # Ensure 'O_score' column is numeric and drop rows with NaNs in 'O_score'
    df[col2] = pd.to_numeric(df[col2], errors='coerce')
    df = df.dropna(subset=[col2])

    # Ensure 'col1' is numeric and drop rows with NaNs in 'col1'
    df[col1] = pd.to_numeric(df[col1], errors='coerce')
    df = df.dropna(subset=[col1])

    # Convert both 'col1' and 'col2' to integers
    df[col1] = df[col1].astype(int)
    df[col2] = df[col2].astype(int)

    return df


def compare_columns_and_analyze_text(df, text_col='passage', query_col='query'):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    punctuation_table = str.maketrans('', '', string.punctuation)

    def clean_and_tokenize(text):
        if pd.isna(text):
            return []
        text = text.translate(punctuation_table)
        tokens = word_tokenize(text.lower())
        return [stemmer.stem(word) for word in tokens if word not in stop_words]

    df['query_proportion'] = df.apply(
        lambda row: calculate_query_proportion(row, query_col, text_col, clean_and_tokenize), axis=1)
    df['TP'] = (df['nist_judgment'] >= 2) & (df['O_score'] >= 2)
    df['TN'] = (df['nist_judgment'] <= 1) & (df['O_score'] <= 1)
    df['FP'] = (df['nist_judgment'] <= 1) & (df['O_score'] >= 2)
    df['FN'] = (df['nist_judgment'] >= 2) & (df['O_score'] <= 1)
    return df


def calculate_query_proportion(row, query_col, text_col, tokenizer):
    query_tokens = tokenizer(row[query_col])
    text_tokens = tokenizer(row[text_col])
    common_tokens = set(query_tokens) & set(text_tokens)
    return len(common_tokens) / len(query_tokens) if query_tokens else 0


def aggregate_metrics(data):
    return {
        'TP_mean': round(data[data['TP']]['query_proportion'].mean(), 2),
        'TN_mean': round(data[data['TN']]['query_proportion'].mean(), 2),
        'FP_mean': round(data[data['FP']]['query_proportion'].mean(), 2),
        'FN_mean': round(data[data['FN']]['query_proportion'].mean(), 2),
    }


def write_results_to_md(df, filename):
    with open(filename, 'w') as file:
        file.write("\nMarkdown\n")
        file.write(df.to_markdown(index=False))
        file.write("\nLaTeX (basic prompt only)\n")
        # write only basic prompts results
        file.write(
            df[df["prompt"] == "Basic"][['model', "TP_mean", "TN_mean", "FP_mean", "FN_mean"]]
            .style.hide(axis='index')
            .format(precision=2)
            .to_latex(column_format='lrrrr', position='center'))


dir_path = "../LLM_relevance_labelling"
out_path = "../tables_and_plots"

all_data = []

file_list = [f for f in os.listdir(dir_path) if f.endswith(".csv")]
for file_name in tqdm(file_list, desc="Processing files"):
    file_path = os.path.join(dir_path, file_name)
    df = read_and_process_file(file_path)
    processed_df = compare_columns_and_analyze_text(df)
    all_data.append(processed_df)

grouped_data = pd.concat(all_data).groupby(['model', 'prompt'])

metrics_list = []
for (model, prompt), group in grouped_data:
    metrics = aggregate_metrics(group)
    metrics.update({'model': model, 'prompt': prompt})
    metrics_list.append(metrics)

metrics_df = pd.DataFrame(metrics_list)
metrics_df = metrics_df[['model', 'prompt', 'TP_mean', 'TN_mean', 'FP_mean', 'FN_mean']]
metrics_df["model"] = metrics_df["model"].apply(process_model_name)

metrics_df["model"] = metrics_df["model"].map(model_name_mapping)
metrics_df["prompt"] = metrics_df["prompt"].map(author_mapping)
metrics_df.sort_values(by=["model", "prompt"], inplace=True)

md_file_path = os.path.join(out_path, "table_4_query_word_matching_tbl_summary.md")
write_results_to_md(metrics_df, md_file_path)

# Combine all data for detailed CSV
combined_data_df = pd.concat(all_data)
column_order = ['model', 'prompt', 'query', 'passage', 'nist_judgment', 'O_score', 'query_proportion']
combined_data_df = combined_data_df[column_order]
detailed_csv_path = os.path.join(out_path, "query_word_matching_tbl_detailed.csv")
combined_data_df.to_csv(detailed_csv_path, index=False)

print(f"Summary of results has been written to {md_file_path}")
print(f"Combined detailed data have been written to {detailed_csv_path}")

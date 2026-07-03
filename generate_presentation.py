import nbformat as nbf
import os
import argparse

def format_path(path):
    return path.replace('\\', '/')

def generate_presentation(results_dir):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    }
    
    nb.cells.append(nbf.v4.new_markdown_cell("# Thesis Results Presentation\n\nThis notebook contains the dynamically generated results, training logs, and visualizations for the Malaria and Tuberculosis datasets."))
    
    nb.cells.append(nbf.v4.new_code_cell("""import sys\nimport os\nimport pandas as pd\nfrom IPython.display import display, Image\n\n# Ensure src is in the path to load utils\nsys.path.append(os.path.abspath('src'))\nfrom utils import plot_training_history_from_csv"""))
    
    datasets = ["malaria", "tb"]
    models = ["MobileNetV2", "Custom CNN", "VGG16", "ResNet50", "DenseNet121"]
    
    for ds in datasets:
        nb.cells.append(nbf.v4.new_markdown_cell(f"## {ds.upper()} Pipeline Results"))
        
        # EDA
        eda_dist = format_path(os.path.join(results_dir, f"eda_distribution_{ds}.png"))
        eda_samp = format_path(os.path.join(results_dir, f"eda_samples_{ds}.png"))
        
        nb.cells.append(nbf.v4.new_markdown_cell(f"### Exploratory Data Analysis ({ds.upper()})"))
        
        eda_code = []
        if os.path.exists(eda_dist): eda_code.append(f"display(Image(filename='{eda_dist}'))")
        if os.path.exists(eda_samp): eda_code.append(f"display(Image(filename='{eda_samp}'))")
        
        if eda_code:
            nb.cells.append(nbf.v4.new_code_cell("\\n".join(eda_code)))
        else:
            nb.cells.append(nbf.v4.new_markdown_cell(f"_EDA images not found in {results_dir}._"))
            
        # Models
        for model in models:
            nb.cells.append(nbf.v4.new_markdown_cell(f"### {model} ({ds.upper()})"))
            log_path = format_path(os.path.join(results_dir, f"training_log_{ds}_{model}.csv"))
            if os.path.exists(log_path):
                nb.cells.append(nbf.v4.new_code_cell(f"plot_training_history_from_csv('{log_path}', title='{model} on {ds.upper()}')"))
            else:
                nb.cells.append(nbf.v4.new_markdown_cell(f"_Training log for {model} not found in {results_dir}._"))
                
        # Benchmarks
        nb.cells.append(nbf.v4.new_markdown_cell(f"### Benchmark Results ({ds.upper()})"))
        
        csv_path = format_path(os.path.join(results_dir, f"comparative_results_{ds}.csv"))
        roc_path = format_path(os.path.join(results_dir, f"comparative_roc_{ds}.png"))
        f1_path = format_path(os.path.join(results_dir, f"comparative_f1_{ds}.png"))
        
        bench_code = []
        if os.path.exists(csv_path):
            bench_code.append(f"df = pd.read_csv('{csv_path}')\\ndisplay(df)")
        if os.path.exists(roc_path): bench_code.append(f"display(Image(filename='{roc_path}'))")
        if os.path.exists(f1_path): bench_code.append(f"display(Image(filename='{f1_path}'))")
        
        if bench_code:
            nb.cells.append(nbf.v4.new_code_cell("\\n".join(bench_code)))
        else:
            nb.cells.append(nbf.v4.new_markdown_cell(f"_Benchmark results not found in {results_dir}._"))
            
    with open('presentation.ipynb', 'w') as f:
        nbf.write(nb, f)
    print(f"Presentation notebook generated successfully as 'presentation.ipynb'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate presentation notebook from results directory")
    parser.add_argument('results_dir', type=str, nargs='?', default='.', help="Directory containing the results (CSVs, PNGs)")
    args = parser.parse_args()
    generate_presentation(args.results_dir)

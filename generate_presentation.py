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
    
    datasets = {"malaria": "Malaria", "tb": "Tuberculosis"}
    splits = ["70_20_10", "75_15_10", "90_5_5"]  # Train_Test_Val
    models = {"MobileNetV2": "MobileNetV2_Attention", "Custom CNN": "Custom_CNN_Attention", "VGG16": "VGG16_Attention",
              "ResNet50": "ResNet50_Attention", "DenseNet121": "DenseNet121_Attention"}
    generated_dir = os.path.join(results_dir, "generated_results")
    
    for ds, display_name in datasets.items():
        nb.cells.append(nbf.v4.new_markdown_cell(f"## {display_name} Pipeline Results"))
        
        # EDA and Grad-CAM attention map
        visuals_dir = os.path.join(generated_dir, display_name, "Visuals_and_EDA")
        nb.cells.append(nbf.v4.new_markdown_cell(f"### Exploratory Data Analysis and Attention Map ({display_name})"))
        visuals = sorted(os.listdir(visuals_dir)) if os.path.isdir(visuals_dir) else []
        eda_code = [f"display(Image(filename='{format_path(os.path.join(visuals_dir, v))}'))" for v in visuals if v.endswith(".png")]
        if eda_code:
            nb.cells.append(nbf.v4.new_code_cell("\n".join(eda_code)))
        else:
            nb.cells.append(nbf.v4.new_markdown_cell(f"_EDA images not found in {visuals_dir}._"))
        
        for split in splits:
            split_label = split.replace("_", ":")
            nb.cells.append(nbf.v4.new_markdown_cell(f"### Split {split_label} (Train:Test:Val)"))
            
            # Training histories
            for model, layer_name in models.items():
                log_path = format_path(os.path.join(results_dir, f"training_log_{ds}_{split}_{layer_name}.csv"))
                if os.path.exists(log_path):
                    nb.cells.append(nbf.v4.new_code_cell(f"plot_training_history_from_csv('{log_path}', title='{model} on {display_name} ({split_label})')"))
                else:
                    nb.cells.append(nbf.v4.new_markdown_cell(f"_Training log for {model} ({split_label}) not found in {results_dir}._"))
            
            # Benchmarks
            csv_path = format_path(os.path.join(results_dir, f"comparative_results_{ds}_{split}.csv"))
            general_dir = os.path.join(generated_dir, display_name, "Evaluation_Charts", f"Split_{split}", "General_Performance")
            charts = [
                format_path(os.path.join(general_dir, f"confusion_matrices_{display_name}_{split}.png")),
                format_path(os.path.join(generated_dir, display_name, "Evaluation_Charts", f"Split_{split}", "AUC-ROC", f"roc_curves_{display_name}_{split}.png")),
                format_path(os.path.join(generated_dir, display_name, "Evaluation_Charts", f"Split_{split}", "F1-Score", f"bar_chart_F1-Score_{display_name}_{split}.png")),
            ]
            bench_code = []
            if os.path.exists(csv_path):
                bench_code.append(f"df = pd.read_csv('{csv_path}')\ndisplay(df)")
            bench_code += [f"display(Image(filename='{c}'))" for c in charts if os.path.exists(c)]
            
            if bench_code:
                nb.cells.append(nbf.v4.new_code_cell("\n".join(bench_code)))
            else:
                nb.cells.append(nbf.v4.new_markdown_cell(f"_Benchmark results for split {split_label} not found in {results_dir}._"))
    
    report_path = format_path(os.path.join(generated_dir, "Documentation_and_Reports", "report_tables.md"))
    if os.path.exists(report_path):
        nb.cells.append(nbf.v4.new_markdown_cell("## Report Tables"))
        nb.cells.append(nbf.v4.new_code_cell(f"from IPython.display import Markdown\ndisplay(Markdown(open('{report_path}').read()))"))
            
    with open('presentation.ipynb', 'w') as f:
        nbf.write(nb, f)
    print(f"Presentation notebook generated successfully as 'presentation.ipynb'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate presentation notebook from results directory")
    parser.add_argument('results_dir', type=str, nargs='?', default='.', help="Directory containing the results (CSVs, PNGs)")
    args = parser.parse_args()
    generate_presentation(args.results_dir)

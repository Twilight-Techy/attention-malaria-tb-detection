# Attention-Improved Malaria and Tuberculosis Detection

A deep learning framework for automated screening of **malaria** in blood smears and
**tuberculosis** in chest X-rays, built around **CBAM** (Convolutional Block Attention Modules)
and evaluated across five architectures, with the comparison run properly: matched training
conditions, statistical significance testing, and latency and size measurements for deployment
on constrained hardware.

> **On authorship.** The research design, hypothesis, and study framing belong to a separate
> research project and are not mine. This repository is the implementation I built to support
> it: the attention modules, preprocessing pipeline, training harness, and benchmark suite.

---

## What the framework does

Attention modules are inserted into a convolutional backbone so the network learns *where* and
*what* to weight before classifying. `cbam_block` applies channel attention followed by spatial
attention:

- **Channel attention**: average- and max-pooled descriptors pass through a shared bottleneck
  MLP (reduction ratio 8), are summed, and gate the channels through a sigmoid.
- **Spatial attention**: the channel-pooled mean and max are concatenated and convolved with a
  7×7 kernel to produce a per-pixel gate.

The same block drops into every architecture, which is what makes the comparison fair.

## Architectures compared

| Model | Backbone | Where CBAM goes |
|---|---|---|
| Custom CNN | 3 conv blocks, trained from scratch | after the 2nd and 3rd blocks |
| ResNet50 | ImageNet, frozen then fine-tuned | after the convolutional base |
| VGG16 | ImageNet, frozen then fine-tuned | after the convolutional base |
| MobileNetV2 | ImageNet, frozen then fine-tuned | after the convolutional base |
| DenseNet121 | ImageNet, frozen then fine-tuned | after the convolutional base |

Transfer models share one builder (`build_transfer_model`), so the only variable between runs
is the backbone.

## Preprocessing

Medical images are not natural images, so the pipeline does more than resize and normalize.
Each image is converted to **LAB** colour space, **CLAHE** is applied to the L channel only
(which lifts local contrast in stained smears and X-rays without distorting colour), then a 3×3
Gaussian blur suppresses acquisition noise before augmentation.

## Training

Two phases, because fine-tuning a frozen ImageNet backbone from the start washes out the
pretrained features:

1. Train with the backbone frozen (`compile_model`, lr `1e-4`).
2. `unfreeze_and_finetune` releases the top N layers and continues at lr `1e-5`.

Early stopping, checkpointing, LR scheduling, and CSV logging throughout; the logs let a run
resume and are what the history plots are drawn from.

## Evaluation

This is where most comparative studies stop early. Here each model reports:

- **Accuracy, precision, recall, F1, AUC** on a held-out test set
- **Comparative ROC curves** and per-metric bar charts across all five models
- **McNemar's test** between model pairs, so a claim that one architecture beats another is
  backed by a significance test rather than a decimal place
- **Grad-CAM heatmaps**: for a clinical tool, seeing *where* the model looked matters as much
  as whether it was right
- **Inference latency (ms/image) and on-disk model size**, since the target is deployment in
  resource-constrained settings where MobileNetV2 winning on cost may matter more than
  DenseNet121 winning on F1

### Splits and the `generated_results/` folder

Every architecture is trained and tested on three Train:Test:Val splits of each dataset:
**70:20:10**, **75:15:10** and **90:5:5** (5 models × 3 splits × 2 datasets = 30 runs). After each
split is benchmarked, its test-set predictions are saved (`evaluation_<dataset>_<split>.npz`), so the
run can resume across sessions. At the end, `src/results.py` builds the full deliverable:

```
generated_results/
  Documentation_and_Reports/   all_metrics_results.csv, report_tables.md, detailed_classification_reports.txt,
                               dataset_metadata.txt, summarized_report.md, student_defense_notes.md
  Malaria/ and Tuberculosis/
    Evaluation_Charts/Split_<split>/
      Accuracy, Precision, Recall, F1-Score, AUC-ROC, AUC-PR/   bar chart, histogram, pie chart (+ ROC / PR curves)
      Combined_Metrics/                                        box plot of all classification metrics
      General_Performance/                                     confusion matrices, training history
    Visuals_and_EDA/          class distribution, sample images, Grad-CAM attention map
  training_logs/              per-epoch log for every model, dataset and split
```

The disease class (Parasitized / Tuberculosis) is the positive class throughout. Regression metrics
are computed on the predicted disease probability. The datasets have no pixel masks, so the
"segmentation" metrics (IoU, Dice, mAP, pixel accuracy) are computed per class at image level and
averaged over the two classes.

---

## Layout

```
src/
  attention.py      CBAM: channel and spatial attention
  models.py         the five architectures, one shared transfer builder
  data_loader.py    CLAHE/LAB preprocessing, denoising, augmentation, splits
  train.py          two-phase training, callbacks, CSV logging
  benchmark.py      comparative evaluation, latency, model size
  utils.py          metrics, ROC and bar charts, McNemar, Grad-CAM
  results.py        builds generated_results/: charts, EDA, attention maps, logs, reports
  download_data.py  pulls the NIH malaria and TB datasets via the Kaggle API
main.ipynb          orchestrates the pipeline
kaggle_main.ipynb   Kaggle/Colab GPU variant
```

## Running it

```bash
pip install -r requirements.txt

# Kaggle API token at ~/.kaggle/kaggle.json
python src/download_data.py
```

Then open `main.ipynb` (or `kaggle_main.ipynb` on a GPU runtime), select an architecture, and
run through training and evaluation. See `NOTEBOOK_GUIDE.md` for the walkthrough.

**Stack:** TensorFlow, Keras, OpenCV, scikit-learn, statsmodels, pandas, matplotlib, seaborn.

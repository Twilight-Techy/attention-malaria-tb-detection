# Jupyter Notebook Execution Guide

The `main.ipynb` notebook is designed to be highly modular and memory-efficient for cloud environments (like Google Colab or Azure ML). To achieve this, several lines of Python code have been intentionally commented out.

This guide explains how to use the commented code blocks effectively.

---

### 1. Swapping Model Architectures
By default, the notebook only instantiates the **MobileNetV2** architecture:
```python
# model = build_custom_cnn_attention()
# model = build_resnet50_attention()
# model = build_vgg16_attention()
model = build_mobilenetv2_attention()
# model = build_densenet121_attention()
```
**Why?** Training all 5 heavy models in a single execution would cause the cloud GPU to run out of memory (OOM). 
**How to use:** To train a different model, simply comment out `MobileNetV2` and uncomment the model you wish to train (e.g., `build_resnet50_attention()`).

### 2. Switching Datasets (Malaria vs Tuberculosis)
By default, the notebook loads the **Malaria** dataset:
```python
malaria_train, malaria_val = load_malaria_data(base_dir, batch_size=32)

# tb_train, tb_val = load_tb_data(base_dir, batch_size=32)
```
**Why?** To allow you to test the pipeline end-to-end on a single disease without loading massive amounts of unused image data into RAM.
**How to use:** When you are ready to train your Tuberculosis models, comment out the `malaria_train` line, uncomment the `tb_train` line, and ensure you pass `tb_train` into the `train_model()` function in the training cell.

### 3. Running McNemar's Statistical Test
By default, the McNemar test execution is commented out:
```python
# y_pred_probs_m2 = model_vgg.predict(malaria_val)
# perform_mcnemar_test(y_true, y_pred_probs_m1, y_pred_probs_m2)
```
**Why?** McNemar's test requires comparing the predictions of **two different trained models** simultaneously. Since the notebook only trains one model per run (to save memory), this code would throw a `NameError` if left uncommented.
**How to use:** 
1. Train your first model and save its predictions (`y_pred_probs_m1`).
2. Train your second model (e.g., VGG16) and save its predictions (`y_pred_probs_m2`).
3. Uncomment the `perform_mcnemar_test` line to calculate the statistical significance (p-value) between the two models.

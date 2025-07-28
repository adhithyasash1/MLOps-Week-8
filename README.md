# 🧪 analyzing model robustness under data poisoning

[![build](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/adhithyasash1/adhithyasash1-mlops-week-8/actions)
[![pipeline](https://img.shields.io/badge/dvc-cml--pipeline-blue)](https://dvc.org/doc/cml)
[![storage](https://img.shields.io/badge/data--versioning-enabled-yellow)](https://dvc.org)
[![accuracy](https://img.shields.io/badge/baseline%20accuracy-93.3%25-blue)](baseline_acc.txt)

this project explores how injecting label noise (i.e. "data poisoning") impacts the performance of a machine learning classifier. the pipeline is fully reproducible using `dvc`, tracked using `dvclive`, and runs automatically via `github actions` + `cml`.

---

## 🧠 objective

evaluate how the validation accuracy, loss, and f1-score degrade as the **poison rate** (i.e., the fraction of corrupted training labels) increases from 0% to 50%.

---

## 📂 directory layout

```
adhithyasash1-mlops-week-8/
├── data/                  → dvc-tracked iris dataset
├── models/                → trained models (output)
├── plots/                 → confusion matrices per run
├── dvclive/               → metrics + plots tracked via dvclive
├── train.py               → full ml pipeline script
├── plot_summary.py        → final summary plot across runs
├── params.yaml            → configurable poison rate
├── dvc.yaml               → pipeline definition
├── .github/workflows/     → github actions + cml pipeline
├── results.csv            → exported run-wise metrics
├── summary_plot.png       → accuracy/loss vs poison rate
```

---

## ⚙️ pipeline overview

* 💾 **dvc** is used to define stages, manage outputs, and handle experiments
* 📈 **dvclive** tracks key metrics during each run
* 🧪 **github actions** orchestrates multiple experiments via looping `dvc exp run`
* 🧮 models trained: logistic regression, random forest, and svm
* 🖼️ confusion matrices generated for each model
* 📊 a final summary plot shows trends in accuracy/loss vs. poison rate

---

## 🔁 experiment automation

the `run.yml` ci workflow:

* pulls dataset via dvc
* runs experiments for poison rates: 0.0, 0.05, 0.10, 0.25, 0.50
* collects metrics via `dvc exp show --csv`
* plots accuracy & loss trends
* creates a markdown report as a cml comment on the pr

---

## 📉 key output

### 🔬 sample summary plot

![summary plot](summary_plot.png)

### 📊 baseline accuracy

extracted and stored in `baseline_acc.txt`: **93.3%**

---

## 📥 run it locally

```bash
# install dependencies
pip install -r requirements.txt

# run a single experiment
dvc exp run --set-param train.poison_rate=0.25

# export metrics
dvc exp show --csv > results.csv

# generate plot
python plot_summary.py
```

---

## 🙋‍♂️ author

R Sashi Adhithya (21F3000611)
github: [@adhithyasash1](https://github.com/adhithyasash1)

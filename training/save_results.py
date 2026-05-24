import json
import joblib
import numpy as np
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_model(model, path):
    ensure_dir(os.path.dirname(path))
    joblib.dump(model, path)


def save_report(report, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(report, f, indent=4)


def save_confusion_matrix(cm, path):
    ensure_dir(os.path.dirname(path))
    np.save(path, cm)


def save_metrics(metrics, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)
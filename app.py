# app.py
import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

st.set_page_config(page_title="MMFL Healthcare Demo", layout="wide")

st.title("🧠 Multimodal Federated Learning for Predictive Healthcare Analytics")
st.caption("Lightweight Federated Learning Simulation (Streamlit-safe)")

# =========================
# Utilities
# =========================
def to_tensor(X, y):
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long),
    )

def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

def plot_roc(y_true, y_score, title):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend()
    st.pyplot(fig)
    return auc

# =========================
# Models
# =========================
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )

    def forward(self, x):
        return self.net(x)

class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 32)

    def forward(self, x):
        return self.fc(x)

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        return self.fc(x)

# =========================
# Load data (TABULAR ONLY)
# =========================
@st.cache_data
def load_data():
    diabetes = pd.read_csv("diabetes.csv")
    heart = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    return diabetes, heart

diabetes, heart = load_data()

@st.cache_data
def preprocess():
    # Hospital A – Diabetes
    XA = diabetes.drop(columns=["Outcome"])
    yA = diabetes["Outcome"].values
    XA = StandardScaler().fit_transform(XA)

    # Hospital C – Heart Failure
    XC = heart.drop(columns=["DEATH_EVENT"])
    yC = heart["DEATH_EVENT"].values
    XC = StandardScaler().fit_transform(XC)

    return XA, yA, XC, yC

XA, yA, XC, yC = preprocess()

XA_tr, XA_te, yA_tr, yA_te = train_test_split(
    XA, yA, test_size=0.25, stratify=yA, random_state=42
)
XC_tr, XC_te, yC_tr, yC_te = train_test_split(
    XC, yC, test_size=0.25, stratify=yC, random_state=42
)

# =========================
# Baseline (Before FL)
# =========================
def train_eval_baseline(Xtr, ytr, Xte, yte, input_dim):
    model = nn.Sequential(Encoder(input_dim), Classifier())
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    Xtr_t, ytr_t = to_tensor(Xtr, ytr)
    Xte_t, yte_t = to_tensor(Xte, yte)

    for _ in range(5):
        opt.zero_grad()
        loss_fn(model(Xtr_t), ytr_t).backward()
        opt.step()

    logits = model(Xte_t)
    probs = torch.softmax(logits, dim=1)[:, 1].detach().numpy()
    preds = np.argmax(logits.detach().numpy(), axis=1)

    return {
        "acc": accuracy_score(yte, preds),
        "f1": f1_score(yte, preds),
        "cm": confusion_matrix(yte, preds),
        "auc": roc_auc_score(yte, probs),
        "probs": probs,
    }

# =========================
# Federated Learning
# =========================
def fedavg(models, weights):
    avg = {}
    for k in models[0].state_dict():
        avg[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
    return avg

def federated_training(rounds=5):
    proj_global = ProjectionHead()
    history = []

    clients = [
        (XA_tr, yA_tr, XA.shape[1]),
        (XC_tr, yC_tr, XC.shape[1]),
    ]

    for r in range(rounds):
        updates, sizes = [], []
        for Xtr, ytr, dim in clients:
            enc = Encoder(dim)
            proj = ProjectionHead()
            proj.load_state_dict(proj_global.state_dict())
            clf = Classifier()

            model = nn.Sequential(enc, proj, clf)
            opt = optim.Adam(model.parameters(), lr=0.001)
            loss_fn = nn.CrossEntropyLoss()

            Xtr_t, ytr_t = to_tensor(Xtr, ytr)
            for _ in range(2):
                opt.zero_grad()
                loss_fn(model(Xtr_t), ytr_t).backward()
                opt.step()

            updates.append(proj)
            sizes.append(len(Xtr))

        proj_global.load_state_dict(
            fedavg(updates, [s / sum(sizes) for s in sizes])
        )
        history.append(r + 1)

    return proj_global, history

def eval_after_fl(Xte, yte, input_dim, proj_global):
    enc = Encoder(input_dim)
    clf = Classifier()
    model = nn.Sequential(enc, proj_global, clf)

    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    Xte_t, yte_t = to_tensor(Xte, yte)
    for _ in range(3):
        opt.zero_grad()
        loss_fn(model(Xte_t), yte_t).backward()
        opt.step()

    logits = model(Xte_t)
    probs = torch.softmax(logits, dim=1)[:, 1].detach().numpy()
    preds = np.argmax(logits.detach().numpy(), axis=1)

    return {
        "acc": accuracy_score(yte, preds),
        "f1": f1_score(yte, preds),
        "cm": confusion_matrix(yte, preds),
        "auc": roc_auc_score(yte, probs),
        "probs": probs,
    }

# =========================
# UI
# =========================
if st.button("🚀 Run Federated Learning"):

    st.subheader("📉 Before Federated Learning")
    resA_before = train_eval_baseline(XA_tr, yA_tr, XA_te, yA_te, XA.shape[1])
    resC_before = train_eval_baseline(XC_tr, yC_tr, XC_te, yC_te, XC.shape[1])

    col1, col2 = st.columns(2)
    with col1:
        st.write("Hospital A – Diabetes")
        st.write(resA_before)
        plot_confusion_matrix(resA_before["cm"], "Confusion Matrix (Before)")
        plot_roc(yA_te, resA_before["probs"], "ROC Curve (Before)")

    with col2:
        st.write("Hospital C – Heart Failure")
        st.write(resC_before)
        plot_confusion_matrix(resC_before["cm"], "Confusion Matrix (Before)")
        plot_roc(yC_te, resC_before["probs"], "ROC Curve (Before)")

    st.subheader("🌍 Federated Learning Process")
    proj_global, rounds = federated_training()

    fig, ax = plt.subplots()
    ax.plot(rounds, rounds, marker="o")
    ax.set_xlabel("Federated Rounds")
    ax.set_ylabel("Global Update Step")
    ax.set_title("Federated Learning Progress")
    st.pyplot(fig)

    st.subheader("📈 After Federated Learning")
    resA_after = eval_after_fl(XA_te, yA_te, XA.shape[1], proj_global)
    resC_after = eval_after_fl(XC_te, yC_te, XC.shape[1], proj_global)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Hospital A – Diabetes")
        st.write(resA_after)
        plot_confusion_matrix(resA_after["cm"], "Confusion Matrix (After)")
        plot_roc(yA_te, resA_after["probs"], "ROC Curve (After)")

    with col2:
        st.write("Hospital C – Heart Failure")
        st.write(resC_after)
        plot_confusion_matrix(resC_after["cm"], "Confusion Matrix (After)")
        plot_roc(yC_te, resC_after["probs"], "ROC Curve (After)")

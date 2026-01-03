import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

st.set_page_config(page_title="MMFL Healthcare Demo", layout="wide")

st.title("🧠 Multimodal Federated Learning for Predictive Healthcare Analytics")
st.write("Lightweight Federated Learning Simulation (Streamlit-safe)")

# ---------------- Utility ----------------
def to_tensor(X, y):
    return (
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )

# ---------------- Models ----------------
class Encoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
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
    def __init__(self, num_classes):
        super().__init__()
        self.fc = nn.Linear(32, num_classes)
    def forward(self, x):
        return self.fc(x)

# ---------------- Load data ----------------
@st.cache_data
def load_data():
    diabetes = pd.read_csv("diabetes.csv")
    notes = pd.read_csv("mtsamples.csv")
    heart = pd.read_csv("heart_failure_clinical_records_dataset.csv")
    return diabetes, notes, heart

diabetes, notes, heart = load_data()

# ---------------- Preprocessing ----------------
@st.cache_data
def preprocess():
    # Hospital A
    X_A = diabetes.drop(columns=["Outcome"])
    y_A = diabetes["Outcome"].values
    X_A = StandardScaler().fit_transform(X_A)

    # Hospital B
    notes_clean = notes.dropna(subset=["transcription", "medical_specialty"]).sample(2000, random_state=42)
    X_B = TfidfVectorizer(max_features=1000, stop_words="english").fit_transform(
        notes_clean["transcription"]
    ).toarray()
    y_B = LabelEncoder().fit_transform(notes_clean["medical_specialty"])

    # Hospital C
    X_C = heart.drop(columns=["DEATH_EVENT"])
    y_C = heart["DEATH_EVENT"].values
    X_C = StandardScaler().fit_transform(X_C)

    return X_A, y_A, X_B, y_B, X_C, y_C

X_A, y_A, X_B, y_B, X_C, y_C = preprocess()

# ---------------- Train/Test split ----------------
XA_tr, XA_te, yA_tr, yA_te = train_test_split(X_A, y_A, test_size=0.25, stratify=y_A, random_state=42)
XB_tr, XB_te, yB_tr, yB_te = train_test_split(X_B, y_B, test_size=0.25, random_state=42)
XC_tr, XC_te, yC_tr, yC_te = train_test_split(X_C, y_C, test_size=0.25, stratify=y_C, random_state=42)

# ---------------- Training functions ----------------
def train_eval_baseline(Xtr, ytr, Xte, yte, dim, classes):
    model = nn.Sequential(Encoder(dim), Classifier(classes))
    opt = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    Xtr_t, ytr_t = to_tensor(Xtr, ytr)
    Xte_t, yte_t = to_tensor(Xte, yte)

    for _ in range(5):
        opt.zero_grad()
        loss_fn(model(Xtr_t), ytr_t).backward()
        opt.step()

    preds = model(Xte_t).argmax(dim=1).numpy()
    return accuracy_score(yte, preds), f1_score(yte, preds, average="macro"), confusion_matrix(yte, preds)

def fedavg(models, weights):
    avg = {}
    for k in models[0].state_dict():
        avg[k] = sum(weights[i] * models[i].state_dict()[k] for i in range(len(models)))
    return avg

# ---------------- Run ----------------
if st.button("🚀 Run Federated Learning"):

    st.subheader("📉 Before Federated Learning")
    cols = st.columns(3)

    accA, f1A, cmA = train_eval_baseline(XA_tr, yA_tr, XA_te, yA_te, X_A.shape[1], 2)
    accB, f1B, cmB = train_eval_baseline(XB_tr, yB_tr, XB_te, yB_te, X_B.shape[1], len(np.unique(y_B)))
    accC, f1C, cmC = train_eval_baseline(XC_tr, yC_tr, XC_te, yC_te, X_C.shape[1], 2)

    for col, name, acc, f1, cm in zip(
        cols,
        ["Hospital A", "Hospital B", "Hospital C"],
        [accA, accB, accC],
        [f1A, f1B, f1C],
        [cmA, cmB, cmC],
    ):
        col.metric(name, f"Acc: {acc:.3f}", f"F1: {f1:.3f}")
        col.write(cm)

    st.subheader("🌍 Federated Training")

    proj_global = ProjectionHead()
    clients = [
        (XA_tr, yA_tr, X_A.shape[1], 2),
        (XB_tr, yB_tr, X_B.shape[1], len(np.unique(y_B))),
        (XC_tr, yC_tr, X_C.shape[1], 2),
    ]

    for _ in range(5):
        updates, sizes = [], []
        for Xtr, ytr, dim, cls in clients:
            model = nn.Sequential(Encoder(dim), ProjectionHead(), Classifier(cls))
            model[1].load_state_dict(proj_global.state_dict())

            opt = optim.Adam(model.parameters(), lr=0.001)
            Xtr_t, ytr_t = to_tensor(Xtr, ytr)

            for _ in range(2):
                opt.zero_grad()
                nn.CrossEntropyLoss()(model(Xtr_t), ytr_t).backward()
                opt.step()

            updates.append(model[1])
            sizes.append(len(Xtr))

        proj_global.load_state_dict(fedavg(updates, [s / sum(sizes) for s in sizes]))

    st.subheader("📈 After Federated Learning")
    cols = st.columns(3)

    for col, (Xte, yte, dim, cls), name in zip(
        cols,
        [
            (XA_te, yA_te, X_A.shape[1], 2),
            (XB_te, yB_te, X_B.shape[1], len(np.unique(y_B))),
            (XC_te, yC_te, X_C.shape[1], 2),
        ],
        ["Hospital A", "Hospital B", "Hospital C"],
    ):
        model = nn.Sequential(Encoder(dim), proj_global, Classifier(cls))
        opt = optim.Adam(model.parameters(), lr=0.001)
        Xte_t, yte_t = to_tensor(Xte, yte)

        for _ in range(3):
            opt.zero_grad()
            nn.CrossEntropyLoss()(model(Xte_t), yte_t).backward()
            opt.step()

        preds = model(Xte_t).argmax(dim=1).numpy()
        col.metric(name, f"Acc: {accuracy_score(yte, preds):.3f}",
                   f"F1: {f1_score(yte, preds, average='macro'):.3f}")
        col.write(confusion_matrix(yte, preds))

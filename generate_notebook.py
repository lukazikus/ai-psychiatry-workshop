#!/usr/bin/env python3
"""
Generate the E-DAIC Depression Characterization Colab Notebook.

Run: python generate_notebook.py
Output: Depression_Characterization_EDAIC.ipynb
"""

import json
import os


def _source(text):
    """Convert a multi-line string to .ipynb source array format."""
    text = text.strip("\n")
    lines = text.split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text):
    """Create a markdown cell."""
    return {"cell_type": "markdown", "metadata": {}, "source": _source(text)}


def code(text):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "metadata": {},
        "source": _source(text),
        "execution_count": None,
        "outputs": [],
    }


cells = []

# =================================================================
# SECTION 1 — Welcome, Setup & Environment
# =================================================================

cells.append(md(
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 40px 30px; border-radius: 16px; color: white; margin-bottom: 20px;">\n'
'  <h1 style="margin:0; font-size:2.2em; font-weight:700;">\U0001f9e0 Machine Learning for Depression Characterization</h1>\n'
'  <h3 style="margin:8px 0 0 0; font-weight:400; opacity:0.9;">Using the E-DAIC Dataset \u2014 A Multimodal Approach</h3>\n'
'  <hr style="border: 1px solid rgba(255,255,255,0.3); margin: 16px 0;">\n'
'  <p style="margin:0; font-size:0.95em; opacity:0.85;">\n'
'    EMBS Workshop &nbsp;|&nbsp; Interactive Notebook &nbsp;|&nbsp; ~45 minutes\n'
'  </p>\n'
'</div>'
))

cells.append(md(
'## \U0001f4cb Learning Objectives\n'
'\n'
'By the end of this workshop, you will be able to:\n'
'\n'
'1. **Understand** the E-DAIC dataset and PHQ-8 depression scoring system\n'
'2. **Extract & visualize** multimodal features (text, audio, visual) relevant to depression\n'
'3. **Train & evaluate** classical ML and deep learning models for both regression (PHQ-8 score) and classification (depressed vs. not)\n'
'4. **Critically assess** methodological pitfalls (subject leakage, interviewer bias) and ethical considerations\n'
'\n'
'---\n'
'\n'
'### \U0001f5fa\ufe0f Notebook Roadmap\n'
'\n'
'| # | Section | Time |\n'
'|---|---------|------|\n'
'| 1 | Setup & Environment | 3 min |\n'
'| 2 | Background: Depression & E-DAIC | 4 min |\n'
'| 3 | Data Loading & Exploration | 5 min |\n'
'| 4 | Multimodal Feature Engineering | 8 min |\n'
'| 5 | Feature Fusion & Preparation | 3 min |\n'
'| 6 | Classical ML \u2014 Regression | 6 min |\n'
'| 7 | Classical ML \u2014 Classification | 5 min |\n'
'| 8 | Deep Learning \u2014 MLP | 5 min |\n'
'| 9 | Results & Discussion | 4 min |\n'
'| 10 | Ethics & Future Directions | 2 min |\n'
'\n'
'<div style="background: #fff3e0; padding: 12px 16px; border-left: 5px solid #ff9800; border-radius: 4px; margin-top:12px;">\n'
'\u26a0\ufe0f <b>Disclaimer:</b> This notebook is for <b>educational and research purposes only</b>. The models presented here are <b>not</b> clinical diagnostic tools and should never be used for actual clinical decision-making.\n'
'</div>'
))

# --- Install dependencies ---
cells.append(md('### \u2699\ufe0f 1.1 \u2014 Install Dependencies'))

cells.append(code(
'%%capture\n'
'# Install required packages (pinned versions for reproducibility)\n'
'!pip install -q \\\n'
'    scikit-learn==1.4.2 \\\n'
'    xgboost==2.0.3 \\\n'
'    sentence-transformers==2.7.0 \\\n'
'    librosa==0.10.2 \\\n'
'    seaborn==0.13.2 \\\n'
'    tqdm \\\n'
'    umap-learn\n'
'\n'
'print("\u2705 All dependencies installed successfully!")'
))

# --- Imports ---
cells.append(md('### \U0001f4e6 1.2 \u2014 Import Libraries'))

cells.append(code(
'# ---- Standard Library ----\n'
'import os\n'
'import glob\n'
'import warnings\n'
'import re\n'
'from collections import Counter\n'
'\n'
'# ---- Data & Numerical ----\n'
'import numpy as np\n'
'import pandas as pd\n'
'from scipy import stats\n'
'\n'
'# ---- Visualization ----\n'
'import matplotlib.pyplot as plt\n'
'import matplotlib.patches as mpatches\n'
'import seaborn as sns\n'
'from IPython.display import display, HTML, Audio\n'
'\n'
'# ---- Machine Learning ----\n'
'from sklearn.preprocessing import StandardScaler\n'
'from sklearn.linear_model import Ridge, LogisticRegression\n'
'from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n'
'from sklearn.model_selection import GridSearchCV\n'
'from sklearn.base import clone\n'
'from sklearn.metrics import (\n'
'    mean_squared_error, mean_absolute_error, r2_score,\n'
'    accuracy_score, f1_score, precision_score, recall_score,\n'
'    roc_auc_score, roc_curve, precision_recall_curve,\n'
'    confusion_matrix, classification_report\n'
')\n'
'import xgboost as xgb\n'
'\n'
'# ---- Deep Learning ----\n'
'import torch\n'
'import torch.nn as nn\n'
'import torch.optim as optim\n'
'from torch.utils.data import DataLoader, TensorDataset\n'
'\n'
'# ---- NLP ----\n'
'from sentence_transformers import SentenceTransformer\n'
'\n'
'# ---- Dimensionality Reduction ----\n'
'from umap import UMAP\n'
'\n'
'# ---- Utilities ----\n'
'from tqdm.auto import tqdm\n'
'\n'
'# ---- Suppress warnings for cleaner output ----\n'
'warnings.filterwarnings("ignore")\n'
'\n'
'# ---- Plotting theme ----\n'
"plt.rcParams.update({\n"
"    'figure.figsize': (10, 5),\n"
"    'figure.dpi': 100,\n"
"    'axes.spines.top': False,\n"
"    'axes.spines.right': False,\n"
"    'font.size': 11,\n"
"    'axes.titlesize': 14,\n"
"    'axes.titleweight': 'bold',\n"
"    'axes.labelsize': 12,\n"
"})\n"
"sns.set_palette([\n"
"    '#2196F3', '#FF5722', '#4CAF50', '#FFC107',\n"
"    '#9C27B0', '#00BCD4', '#E91E63', '#3F51B5'\n"
"])\n"
'\n'
'# Color constants\n'
"C = {\n"
"    'blue': '#2196F3', 'orange': '#FF5722', 'green': '#4CAF50',\n"
"    'amber': '#FFC107', 'purple': '#9C27B0', 'teal': '#00BCD4',\n"
"    'red': '#F44336', 'indigo': '#3F51B5',\n"
"    'not_dep': '#26A69A', 'dep': '#EF5350',\n"
"    'train': '#42A5F5', 'dev': '#FFA726', 'test': '#AB47BC',\n"
"}\n"
'\n'
'print("\u2705 All libraries imported successfully!")\n'
'print(f"   PyTorch: {torch.__version__}")\n'
"print(f\"   scikit-learn: {__import__('sklearn').__version__}\")\n"
"print(f\"   Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}\")"
))

# --- Drive mount ---
cells.append(md(
'### \U0001f4c2 1.3 \u2014 Mount Google Drive & Configure Paths\n'
'\n'
'<div style="background: #e3f2fd; padding: 12px 16px; border-left: 5px solid #2196F3; border-radius: 4px;">\n'
'\U0001f4dd <b>Edit <code>DATA_ROOT</code></b> below to point to the Google Drive folder containing your '
'participant directories (e.g., <code>300_P</code>, <code>301_P</code>) and label CSV files.\n'
'</div>'
))

cells.append(code(
'# ---- Mount Google Drive ----\n'
'from google.colab import drive\n'
"drive.mount('/content/drive')\n"
'\n'
'# ================================================================\n'
'# CONFIGURATION -- EDIT THIS SECTION TO MATCH YOUR SETUP\n'
'# ================================================================\n'
'\n'
'# Path to Google Drive folder containing participant folders and label CSVs\n'
"DATA_ROOT = '/content/drive/MyDrive/E-DAIC/'  # <-- Change this path\n"
'\n'
'# Label file names (train / dev / test) -- loose CSVs on Drive\n'
"TRAIN_LABELS_FILE = 'train_split_Depression_AVEC2019.csv'\n"
"DEV_LABELS_FILE   = 'dev_split_Depression_AVEC2019.csv'\n"
"TEST_LABELS_FILE  = 'test_split_Depression_AVEC2019.csv'\n"
'\n'
'# Participant folder pattern: {pid}_P\n'
"PARTICIPANT_FOLDER = '{pid}_P'\n"
'\n'
'# Feature file patterns inside each participant folder\n'
"TRANSCRIPT_FILE      = '{pid}_Transcript.csv'\n"
"AUDIO_FEATURE_FILE   = '{pid}_OpenSMILE2.3.0_egemaps.csv'\n"
"VISUAL_FEATURE_FILE  = '{pid}_OpenFace2.1.0_Pose_gaze_AUs.csv'\n"
'\n'
'# ================================================================\n'
'\n'
'# ---- Verify DATA_ROOT exists ----\n'
'if not os.path.exists(DATA_ROOT):\n'
'    print(f"\u274c DATA_ROOT not found: {DATA_ROOT}")\n'
'    print("   Please update DATA_ROOT in the configuration above to point to your dataset folder.")\n'
'else:\n'
'    print(f"\u2705 DATA_ROOT exists: {DATA_ROOT}")\n'
'\n'
'# ---- Load labels and filter to available participant folders ----\n'
'df_train = pd.read_csv(os.path.join(DATA_ROOT, TRAIN_LABELS_FILE))\n'
'df_dev   = pd.read_csv(os.path.join(DATA_ROOT, DEV_LABELS_FILE))\n'
'df_test  = pd.read_csv(os.path.join(DATA_ROOT, TEST_LABELS_FILE))\n'
'\n'
'# Find which participants have folders on Drive\n'
'available_pids = set()\n'
'for d in os.listdir(DATA_ROOT):\n'
'    full = os.path.join(DATA_ROOT, d)\n'
'    if os.path.isdir(full):\n'
'        clean_d = d.replace("_P", "").replace("_p", "").split(".")[0].strip()\n'
'        try:\n'
'            pid = int(clean_d)\n'
'            available_pids.add(pid)\n'
'            available_pids.add(f"{pid}_P")\n'
'        except ValueError:\n'
'            available_pids.add(d)\n'
'\n'
'# Filter labels to only available participants\n'
'n_before = len(df_train) + len(df_dev) + len(df_test)\n'
'df_train = df_train[df_train["Participant_ID"].isin(available_pids) | df_train["Participant_ID"].astype(str).isin(available_pids)].reset_index(drop=True)\n'
'df_dev   = df_dev[df_dev["Participant_ID"].isin(available_pids) | df_dev["Participant_ID"].astype(str).isin(available_pids)].reset_index(drop=True)\n'
'df_test  = df_test[df_test["Participant_ID"].isin(available_pids) | df_test["Participant_ID"].astype(str).isin(available_pids)].reset_index(drop=True)\n'
'\n'
"df_train['Split'] = 'train'\n"
"df_dev['Split']   = 'dev'\n"
"df_test['Split']  = 'test'\n"
'df_labels = pd.concat([df_train, df_dev, df_test], ignore_index=True)\n'
"df_labels['Gender'] = df_labels['Gender'].str.strip()\n"
'\n'
'print(f"\U0001f4c1 Found participant folders on Drive")\n'
'print(f"\U0001f465 Using {len(df_labels)}/{n_before} participants: "\n'
'      f"Train={len(df_train)}, Dev={len(df_dev)}, Test={len(df_test)}")'
))

# --- Seeds + GPU ---
cells.append(md('### \U0001f3b2 1.4 \u2014 Reproducibility & Device Setup'))

cells.append(code(
'# ---- Set random seeds for reproducibility ----\n'
'SEED = 42\n'
'np.random.seed(SEED)\n'
'torch.manual_seed(SEED)\n'
'if torch.cuda.is_available():\n'
'    torch.cuda.manual_seed_all(SEED)\n'
'\n'
'# ---- Device configuration ----\n'
"device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
'print(f"\U0001f5a5\ufe0f  Using device: {device}")\n'
"if device.type == 'cuda':\n"
'    print(f"   GPU: {torch.cuda.get_device_name(0)}")\n'
'    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")\n'
'else:\n'
'    print("   \u26a0\ufe0f No GPU detected. Deep learning cells will be slower.")\n'
'    print("   Tip: Runtime -> Change runtime type -> T4 GPU")'
))

# =================================================================
# SECTION 2 — Background
# =================================================================

cells.append(md(
'---\n'
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 20px 24px; border-radius: 12px; color: white;">\n'
'<h2 style="margin:0;">\U0001f4da Section 2 \u2014 Background: Depression & the E-DAIC Dataset</h2>\n'
'<p style="margin:4px 0 0 0; opacity:0.8;">\u23f1\ufe0f ~4 minutes</p>\n'
'</div>'
))

cells.append(md(
'### What is Depression?\n'
'\n'
'**Major Depressive Disorder (MDD)** is one of the most prevalent mental health conditions worldwide, affecting over **280 million people** globally (WHO, 2023). Key characteristics include:\n'
'\n'
'- Persistent feelings of sadness, hopelessness, or emptiness\n'
'- Loss of interest in activities (anhedonia)\n'
'- Changes in sleep, appetite, and energy levels\n'
'- Difficulty concentrating; psychomotor changes\n'
'- In severe cases, suicidal ideation\n'
'\n'
'Depression manifests in **behavioral signals** that can be observed across modalities:\n'
'\n'
'| Modality | Observable Signals |\n'
'|---|---|\n'
'| **Speech/Text** | Reduced vocabulary, more first-person pronouns, negative sentiment, slower speech rate |\n'
'| **Audio/Voice** | Flatter prosody (reduced pitch variation), lower energy, longer pauses |\n'
'| **Visual/Facial** | Reduced facial expressivity, less eye contact, fewer smiles, more downward gaze |'
))

cells.append(md(
'### The PHQ-8 Questionnaire\n'
'\n'
'The **Patient Health Questionnaire-8 (PHQ-8)** is a widely used self-report instrument for measuring depression severity. It consists of 8 items (the PHQ-9 minus the suicidal ideation question), each scored 0\u20133:\n'
'\n'
'| Score Range | Severity Level | Clinical Interpretation |\n'
'|:-----------:|:--------------:|:----------------------:|\n'
'| **0 \u2013 4**  | None / Minimal | No clinical significance |\n'
'| **5 \u2013 9**  | Mild           | Watchful waiting; may not require treatment |\n'
'| **10 \u2013 14** | Moderate       | Treatment plan recommended |\n'
'| **15 \u2013 19** | Moderately Severe | Active treatment (therapy + medication) |\n'
'| **20 \u2013 24** | Severe         | Immediate intervention needed |\n'
'\n'
'> \U0001f3af **Binary threshold:** PHQ-8 \u2265 10 is commonly used as the cutoff for **clinically significant depression** (this is the `PHQ_Binary` label in our dataset).\n'
'\n'
'In this notebook, we tackle **two tasks**:\n'
'1. **Regression:** Predict the exact PHQ-8 score (0\u201324)\n'
'2. **Classification:** Predict depressed (\u226510) vs. not depressed (<10)'
))

cells.append(md(
'### The E-DAIC Dataset\n'
'\n'
'The **Extended Distress Analysis Interview Corpus (E-DAIC)** is an extension of the DAIC-WOZ dataset, created at USC\'s Institute for Creative Technologies and used in the **AVEC 2019** challenge.\n'
'\n'
'**Key facts:**\n'
'- **275 participants** undergo semi-structured clinical interviews\n'
'- Conducted by a virtual agent named **"Ellie"** (Wizard-of-Oz and AI-controlled sessions)\n'
'- Multimodal data: **audio recordings**, **video**, and **text transcripts**\n'
'- Ground truth: **PHQ-8 scores** + **PCL-C (PTSD)** scores\n'
'- Pre-extracted features: **OpenSMILE** (audio) and **OpenFace** (visual)\n'
'\n'
'**Official data splits:**\n'
'\n'
'| Split | Participants | Purpose |\n'
'|:-----:|:----------:|:-------:|\n'
'| **Train** | 163 | Model training |\n'
'| **Dev**   | 56  | Hyperparameter tuning & validation |\n'
'| **Test**  | 56  | Final evaluation |\n'
'\n'
'<div style="background: #fce4ec; padding: 12px 16px; border-left: 5px solid #e91e63; border-radius: 4px; margin-top: 12px;">\n'
'\U0001f512 <b>Data access:</b> The E-DAIC dataset requires a signed data-use agreement from <a href="https://dcapswoz.ict.usc.edu/" style="color: #e91e63;">USC ICT</a>. If you are a workshop participant, the data has been shared with you via Google Drive.\n'
'</div>'
))

# =================================================================
# SECTION 3 — Data Loading & EDA
# =================================================================

cells.append(md(
'---\n'
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 20px 24px; border-radius: 12px; color: white;">\n'
'<h2 style="margin:0;">\U0001f50d Section 3 \u2014 Data Loading & Exploratory Analysis</h2>\n'
'<p style="margin:4px 0 0 0; opacity:0.8;">\u23f1\ufe0f ~5 minutes</p>\n'
'</div>'
))

cells.append(md('### 3.1 \u2014 Load Label Files'))

cells.append(code(
'# ---- Labels were loaded and subsampled in Cell 1.3 ----\n'
'print(f"\U0001f4ca Total participants: {len(df_labels)}")\n'
'print(f"   Train: {len(df_train)} | Dev: {len(df_dev)} | Test: {len(df_test)}")\n'
'print(f"\\n\U0001f4cb Columns: {list(df_labels.columns)}")\n'
'print(f"\\n\U0001f4c8 PHQ-8 Score statistics:")\n'
"display(df_labels.groupby('Split')['PHQ_Score'].describe().round(2))\n"
'display(df_labels.head(10))'
))

cells.append(md('### 3.2 \u2014 PHQ-8 Score Distribution'))

cells.append(code(
'fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n'
'\n'
'# (a) Overall distribution with severity bands\n'
'ax = axes[0]\n'
'severity_bands = [\n'
"    (0, 4, 'None/Minimal', '#C8E6C9'),\n"
"    (5, 9, 'Mild', '#FFF9C4'),\n"
"    (10, 14, 'Moderate', '#FFE0B2'),\n"
"    (15, 19, 'Mod. Severe', '#FFCCBC'),\n"
"    (20, 24, 'Severe', '#FFCDD2'),\n"
']\n'
'for lo, hi, label, color in severity_bands:\n'
'    ax.axvspan(lo - 0.5, hi + 0.5, alpha=0.3, color=color, label=label)\n'
'\n'
"ax.hist(df_labels['PHQ_Score'], bins=range(0, 26), color=C['indigo'],\n"
"        edgecolor='white', alpha=0.85, zorder=3)\n"
"ax.set_xlabel('PHQ-8 Score')\n"
"ax.set_ylabel('Number of Participants')\n"
"ax.set_title('(a) PHQ-8 Distribution with Severity Bands')\n"
'ax.legend(fontsize=8, loc="upper right")\n'
'ax.set_xlim(-0.5, 24.5)\n'
'\n'
'# (b) Distribution by split\n'
'ax = axes[1]\n'
"split_colors = {'train': C['train'], 'dev': C['dev'], 'test': C['test']}\n"
"for split in ['train', 'dev', 'test']:\n"
"    subset = df_labels[df_labels['Split'] == split]['PHQ_Score']\n"
'    ax.hist(subset, bins=range(0, 26), alpha=0.55, label=f\'{split} (n={len(subset)})\',\n'
"            color=split_colors[split], edgecolor='white')\n"
"ax.set_xlabel('PHQ-8 Score')\n"
"ax.set_ylabel('Count')\n"
"ax.set_title('(b) Distribution by Split')\n"
'ax.legend()\n'
'\n'
'# (c) Binary class balance\n'
'ax = axes[2]\n'
"class_counts = df_labels.groupby(['Split', 'PHQ_Binary']).size().unstack(fill_value=0)\n"
"class_counts.columns = ['Not Depressed', 'Depressed']\n"
"class_counts.plot(kind='bar', ax=ax, color=[C['not_dep'], C['dep']],\n"
"                  edgecolor='white', width=0.7)\n"
"ax.set_title('(c) Class Balance by Split')\n"
"ax.set_ylabel('Count')\n"
"ax.set_xlabel('')\n"
"ax.tick_params(axis='x', labelrotation=0)\n"
'for container in ax.containers:\n'
'    ax.bar_label(container, fontsize=9)\n'
'ax.legend()\n'
'\n'
'plt.tight_layout()\n'
'plt.show()\n'
'\n'
'# Print class ratios\n'
"total_dep = df_labels['PHQ_Binary'].sum()\n"
'total_not = len(df_labels) - total_dep\n'
'print(f"\\n\U0001f4ca Overall class balance:")\n'
'print(f"   Not Depressed (PHQ<10): {total_not} ({100*total_not/len(df_labels):.1f}%)")\n'
'print(f"   Depressed (PHQ>=10):    {total_dep} ({100*total_dep/len(df_labels):.1f}%)")'
))

cells.append(md('### 3.3 \u2014 Demographics & Comorbidity'))

cells.append(code(
'fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n'
'\n'
'# (a) Gender distribution\n'
'ax = axes[0]\n'
"gender_dep = df_labels.groupby(['Gender', 'PHQ_Binary']).size().unstack(fill_value=0)\n"
"gender_dep.columns = ['Not Depressed', 'Depressed']\n"
"gender_dep.plot(kind='bar', ax=ax, color=[C['not_dep'], C['dep']],\n"
"                edgecolor='white', width=0.6)\n"
"ax.set_title('(a) Depression by Gender')\n"
"ax.set_ylabel('Count')\n"
"ax.set_xlabel('Gender')\n"
"ax.tick_params(axis='x', labelrotation=0)\n"
'for container in ax.containers:\n'
'    ax.bar_label(container, fontsize=9)\n'
'\n'
'# (b) PHQ-8 vs PCL-C scatter\n'
'ax = axes[1]\n'
"colors = df_labels['PHQ_Binary'].map({0: C['not_dep'], 1: C['dep']})\n"
"ax.scatter(df_labels['PHQ_Score'], df_labels['PCL-C (PTSD)'],\n"
"           c=colors, alpha=0.6, edgecolor='white', s=50, zorder=3)\n"
"r, p = stats.pearsonr(df_labels['PHQ_Score'].dropna(),\n"
"                       df_labels['PCL-C (PTSD)'].dropna())\n"
"ax.set_xlabel('PHQ-8 Score (Depression)')\n"
"ax.set_ylabel('PCL-C Score (PTSD)')\n"
"ax.set_title(f'(b) Depression vs PTSD  (r={r:.2f}, p={p:.1e})')\n"
'legend_elements = [\n'
"    mpatches.Patch(facecolor=C['not_dep'], label='Not Depressed'),\n"
"    mpatches.Patch(facecolor=C['dep'], label='Depressed'),\n"
']\n'
'ax.legend(handles=legend_elements, fontsize=9)\n'
'\n'
'# (c) PHQ-8 by gender (violin)\n'
'ax = axes[2]\n'
"gender_groups = df_labels['Gender'].unique()\n"
'parts = ax.violinplot(\n'
"    [df_labels[df_labels['Gender'] == g]['PHQ_Score'].dropna().values\n"
'     for g in gender_groups],\n'
'    showmeans=True, showmedians=True\n'
')\n'
'ax.set_xticks(range(1, len(gender_groups) + 1))\n'
'ax.set_xticklabels(gender_groups)\n'
"ax.set_ylabel('PHQ-8 Score')\n"
"ax.set_title('(c) PHQ-8 Distribution by Gender')\n"
"ax.axhline(y=10, color=C['red'], linestyle='--', alpha=0.5, label='Depression threshold (10)')\n"
'ax.legend(fontsize=9)\n'
'\n'
'plt.tight_layout()\n'
'plt.show()'
))

cells.append(md(
'### 3.4 \u2014 Explore a Sample Participant\n'
'\n'
"Let's peek inside a single participant's data folder to understand the file structure."
))

cells.append(code(
'# Pick a sample participant from the training set\n'
"sample_pid = df_train['Participant_ID'].iloc[0]\n"
"clean_sample_pid = str(sample_pid).replace('_P', '').replace('_p', '').split('.')[0].strip()\n"
'sample_folder = os.path.join(DATA_ROOT, PARTICIPANT_FOLDER.format(pid=clean_sample_pid))\n'
'if not os.path.exists(sample_folder):\n'
'    sample_folder = os.path.join(DATA_ROOT, str(sample_pid))\n'
'\n'
'print(f"\U0001f464 Sample participant: {sample_pid}")\n'
"print(f\"   PHQ-8 Score: {df_train.iloc[0]['PHQ_Score']}\")\n"
"print(f\"   PHQ Binary:  {df_train.iloc[0]['PHQ_Binary']}\")\n"
'print(f"   Folder: {sample_folder}\\n")\n'
'if os.path.exists(sample_folder):\n'
'    files = []\n'
'    for root, dirs, fnames in os.walk(sample_folder):\n'
'        for f in fnames:\n'
'            rel = os.path.relpath(os.path.join(root, f), sample_folder)\n'
'            files.append(rel)\n'
'    files = sorted(files)\n'
'    print(f"\U0001f4c1 Files in folder ({len(files)} total):")\n'
'    for f in files:\n'
'        size = os.path.getsize(os.path.join(sample_folder, f))\n'
"        icon = '\U0001f4c4' if f.endswith('.csv') else '\U0001f3b5' if f.endswith('.wav') else '\U0001f4e6'\n"
'        print(f"   {icon} {f}  ({size/1024:.1f} KB)")\n'
'else:\n'
'    print(f"\u274c Folder not found: {sample_folder}")\n'
'    print("   Please check PARTICIPANT_FOLDER pattern in the configuration cell.")'
))

cells.append(code(
'# ---- Display a sample transcript ----\n'
'tfile = TRANSCRIPT_FILE.format(pid=clean_sample_pid)\n'
'transcript_path = os.path.join(sample_folder, tfile)\n'
'if not os.path.exists(transcript_path):\n'
'    transcript_path = os.path.join(sample_folder, "features", tfile)\n'
'if not os.path.exists(transcript_path):\n'
'    possible = glob.glob(os.path.join(sample_folder, "**", "*[Tt]ranscript*"), recursive=True)\n'
'    if possible:\n'
'        transcript_path = possible[0]\n'
'\n'
'if os.path.exists(transcript_path):\n'
"    df_transcript = pd.read_csv(transcript_path, sep='\\t')\n"
'    print(f"\U0001f4dd Transcript columns: {list(df_transcript.columns)}")\n'
'    print(f"   Total turns: {len(df_transcript)}\\n")\n'
'    print("--- First 10 turns ---")\n'
'    display(df_transcript.head(10))\n'
'else:\n'
'    print(f"\u26a0\ufe0f Transcript not found at: {transcript_path}")\n'
'    print("   Trying to auto-detect transcript file...")\n'
"    possible = glob.glob(os.path.join(sample_folder, '*[Tt]ranscript*'))\n"
'    if possible:\n'
'        transcript_path = possible[0]\n'
'        print(f"   Found: {transcript_path}")\n'
"        df_transcript = pd.read_csv(transcript_path, sep=None, engine='python')\n"
'        display(df_transcript.head(10))\n'
'    else:\n'
'        print("   No transcript file found. Check the TRANSCRIPT_FILE pattern.")'
))

# =================================================================
# SECTION 4 — Feature Engineering
# =================================================================

cells.append(md(
'---\n'
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 20px 24px; border-radius: 12px; color: white;">\n'
'<h2 style="margin:0;">\U0001f52c Section 4 \u2014 Multimodal Feature Engineering</h2>\n'
'<p style="margin:4px 0 0 0; opacity:0.8;">\u23f1\ufe0f ~8 minutes \u2014 Text, Audio, Visual</p>\n'
'</div>\n'
'\n'
'We extract features from three modalities:\n'
'1. **Text** \u2014 Linguistic patterns from interview transcripts\n'
'2. **Audio** \u2014 Acoustic properties from pre-extracted OpenSMILE features\n'
'3. **Visual** \u2014 Facial behavior from pre-extracted OpenFace features'
))

# --- 4a: TEXT ---
cells.append(md(
'### \U0001f5e3\ufe0f 4a \u2014 Text / Linguistic Features\n'
'\n'
'Depression is associated with distinct language patterns:\n'
'- **Increased first-person singular pronouns** ("I", "me", "my") \u2014 self-focused attention\n'
'- **More negative emotion words** \u2014 hopelessness, sadness, worthlessness\n'
'- **Reduced vocabulary diversity** \u2014 limited cognitive flexibility\n'
'- **Shorter utterances** \u2014 psychomotor retardation'
))

cells.append(code(
'%%time\n'
'# ---- Load all transcripts and extract participant text ----\n'
'\n'
'def load_participant_text(pid, data_root):\n'
'    # Load transcript and return concatenated participant text.\n'
"    clean_pid = str(pid).replace('_P', '').replace('_p', '').split('.')[0].strip()\n"
'    folder = os.path.join(data_root, PARTICIPANT_FOLDER.format(pid=clean_pid))\n'
'    if not os.path.exists(folder):\n'
'        folder = os.path.join(data_root, str(pid))\n'
'\n'
'    tfile = TRANSCRIPT_FILE.format(pid=clean_pid)\n'
'    tpath = os.path.join(folder, tfile)\n'
'    if not os.path.exists(tpath):\n'
"        tpath = os.path.join(folder, 'features', tfile)\n"
'    if not os.path.exists(tpath):\n'
"        tpath = os.path.join(folder, 'Features', tfile)\n"
'\n'
'    if not os.path.exists(tpath):\n'
'        # Try recursive auto-detect\n'
"        possible = glob.glob(os.path.join(folder, '**', '*[Tt]ranscript*'), recursive=True)\n"
'        if possible:\n'
'            tpath = possible[0]\n'
'        else:\n'
'            return None\n'
'\n'
'    try:\n'
"        df = pd.read_csv(tpath, sep='\\t')\n"
'        text_col = None\n'
'        speaker_col = None\n'
'        for col in df.columns:\n'
"            if col.lower() in ['value', 'text', 'utterance', 'sentence', 'content']:\n"
'                text_col = col\n'
"            if col.lower() in ['speaker', 'role', 'participant_role']:\n"
'                speaker_col = col\n'
'\n'
'        if text_col is None:\n'
'            text_col = df.columns[-1]\n'
'\n'
'        if speaker_col:\n'
"            participant_mask = ~df[speaker_col].str.lower().str.contains('ellie|interviewer', na=False)\n"
"            participant_text = ' '.join(df.loc[participant_mask, text_col].dropna().astype(str))\n"
'        else:\n'
"            participant_text = ' '.join(df[text_col].dropna().astype(str))\n"
'\n'
'        return participant_text\n'
'    except Exception as e:\n'
'        return None\n'
'\n'
'\n'
'# Load text for all participants\n'
'text_data = {}\n'
'missing_text = []\n'
"for pid in tqdm(df_labels['Participant_ID'], desc='Loading transcripts'):\n"
'    text = load_participant_text(pid, DATA_ROOT)\n'
'    if text and len(text.strip()) > 0:\n'
'        text_data[pid] = text\n'
'    else:\n'
'        missing_text.append(pid)\n'
'\n'
'print(f"\\n\u2705 Loaded transcripts for {len(text_data)} / {len(df_labels)} participants")\n'
'if missing_text:\n'
'    print(f"\u26a0\ufe0f Missing transcripts for {len(missing_text)} participants: {missing_text[:10]}...")'
))

cells.append(md('#### Handcrafted Linguistic Features'))

cells.append(code(
'# ---- Negative-emotion word list (simplified LIWC-like) ----\n'
'NEGATIVE_WORDS = set([\n'
"    'sad', 'sadness', 'unhappy', 'depressed', 'depression', 'hopeless', 'helpless',\n"
"    'worthless', 'empty', 'lonely', 'anxious', 'anxiety', 'worried', 'afraid', 'fear',\n"
"    'angry', 'anger', 'frustrated', 'irritable', 'terrible', 'horrible', 'awful',\n"
"    'miserable', 'guilty', 'shame', 'regret', 'pain', 'hurt', 'crying', 'cry',\n"
"    'suffer', 'suffering', 'tired', 'exhausted', 'overwhelmed', 'stressed',\n"
"    'hate', 'never', 'nothing', 'nobody', 'alone', 'lost', 'broken', 'fail',\n"
"    'failure', 'useless', 'pointless', 'death', 'die', 'dead',\n"
'])\n'
'\n'
"FIRST_PERSON = set(['i', 'me', 'my', 'mine', 'myself'])\n"
'\n'
'\n'
'def extract_linguistic_features(text):\n'
'    # Extract handcrafted linguistic features from text.\n'
'    if not text or len(text.strip()) == 0:\n'
'        return None\n'
'\n'
"    words = re.findall(r'\\b\\w+\\b', text.lower())\n"
"    sentences = re.split(r'[.!?]+', text)\n"
'    sentences = [s.strip() for s in sentences if s.strip()]\n'
'\n'
'    n_words = len(words)\n'
'    if n_words == 0:\n'
'        return None\n'
'\n'
'    unique_words = set(words)\n'
'    n_unique = len(unique_words)\n'
'\n'
'    features = {\n'
"        'word_count': n_words,\n"
"        'unique_word_count': n_unique,\n"
"        'type_token_ratio': n_unique / n_words,\n"
"        'sentence_count': len(sentences),\n"
"        'avg_sentence_length': n_words / max(len(sentences), 1),\n"
"        'avg_word_length': np.mean([len(w) for w in words]),\n"
"        'question_ratio': sum(1 for s in text.split('\\n') if '?' in s) / max(len(sentences), 1),\n"
"        'first_person_ratio': sum(1 for w in words if w in FIRST_PERSON) / n_words,\n"
"        'negative_word_ratio': sum(1 for w in words if w in NEGATIVE_WORDS) / n_words,\n"
"        'filler_ratio': sum(1 for w in words if w in {'um', 'uh', 'like', 'yeah'}) / n_words,\n"
'    }\n'
'    return features\n'
'\n'
'\n'
'# Extract features for all participants\n'
'ling_features = {}\n'
"for pid, text in tqdm(text_data.items(), desc='Extracting linguistic features'):\n"
'    feats = extract_linguistic_features(text)\n'
'    if feats:\n'
'        ling_features[pid] = feats\n'
'\n'
"df_ling = pd.DataFrame.from_dict(ling_features, orient='index')\n"
"df_ling.index.name = 'Participant_ID'\n"
'df_ling = df_ling.reset_index()\n'
'\n'
'print(f"\u2705 Extracted {len(df_ling.columns)-1} linguistic features for {len(df_ling)} participants")\n'
'display(df_ling.describe().round(3))'
))

cells.append(md('#### Sentence-BERT Embeddings'))

cells.append(code(
'%%time\n'
'# ---- Compute SBERT embeddings ----\n'
'# Using all-MiniLM-L6-v2: fast, 384-dim, great for Colab free tier\n'
"sbert_model = SentenceTransformer('all-MiniLM-L6-v2', device=str(device))\n"
'\n'
'pids_with_text = sorted(text_data.keys())\n'
'texts_list = [text_data[pid] for pid in pids_with_text]\n'
'\n'
'# Encode (mean pooling is built into the model)\n'
'embeddings = sbert_model.encode(\n'
'    texts_list,\n'
'    show_progress_bar=True,\n'
'    batch_size=16,\n'
'    convert_to_numpy=True\n'
')\n'
'\n'
'# Create DataFrame with embedding columns\n'
"embed_cols = [f'sbert_{i}' for i in range(embeddings.shape[1])]\n"
'df_sbert = pd.DataFrame(embeddings, columns=embed_cols)\n'
"df_sbert['Participant_ID'] = pids_with_text\n"
'\n'
'print(f"\u2705 SBERT embeddings: {embeddings.shape[0]} participants x {embeddings.shape[1]} dimensions")'
))

cells.append(md('#### Text Feature Visualization'))

cells.append(code(
'# Merge linguistic features with labels for visualization\n'
"df_ling_viz = df_ling.merge(df_labels[['Participant_ID', 'PHQ_Score', 'PHQ_Binary']], on='Participant_ID')\n"
'\n'
'fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n'
'\n'
'# (a) First-person pronoun ratio by depression status\n'
'ax = axes[0]\n'
"for label, color, name in [(0, C['not_dep'], 'Not Dep'), (1, C['dep'], 'Depressed')]:\n"
"    subset = df_ling_viz[df_ling_viz['PHQ_Binary'] == label]['first_person_ratio']\n"
"    ax.hist(subset, bins=20, alpha=0.6, color=color, label=name, edgecolor='white')\n"
"ax.set_xlabel('First-Person Pronoun Ratio')\n"
"ax.set_ylabel('Count')\n"
"ax.set_title('(a) Self-Referential Language')\n"
'ax.legend()\n'
'\n'
'# (b) Feature correlations with PHQ-8\n'
'ax = axes[1]\n'
"feat_cols = [c for c in df_ling.columns if c != 'Participant_ID']\n"
"correlations = df_ling_viz[feat_cols + ['PHQ_Score']].corr()['PHQ_Score'].drop('PHQ_Score').sort_values()\n"
"correlations.plot(kind='barh', ax=ax, color=[C['dep'] if v > 0 else C['not_dep'] for v in correlations])\n"
"ax.set_xlabel('Correlation with PHQ-8')\n"
"ax.set_title('(b) Feature Correlation with Depression')\n"
"ax.axvline(x=0, color='black', linewidth=0.8)\n"
'\n'
'# (c) UMAP of SBERT embeddings\n'
'ax = axes[2]\n'
"sbert_labels = df_sbert.merge(df_labels[['Participant_ID', 'PHQ_Score']], on='Participant_ID')\n"
'X_umap_input = sbert_labels[embed_cols].values\n'
'umap_reducer = UMAP(n_components=2, random_state=SEED, n_neighbors=15)\n'
'X_umap = umap_reducer.fit_transform(X_umap_input)\n'
'\n'
"scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1],\n"
"                     c=sbert_labels['PHQ_Score'], cmap='RdYlGn_r',\n"
"                     alpha=0.7, s=40, edgecolor='white', linewidth=0.5)\n"
"plt.colorbar(scatter, ax=ax, label='PHQ-8 Score')\n"
"ax.set_xlabel('UMAP-1')\n"
"ax.set_ylabel('UMAP-2')\n"
"ax.set_title('(c) SBERT Embedding Space')\n"
'\n'
'plt.tight_layout()\n'
'plt.show()'
))

# --- 4b: AUDIO ---
cells.append(md(
'### \U0001f3b5 4b \u2014 Audio / Acoustic Features\n'
'\n'
'We use **pre-extracted OpenSMILE eGeMAPS** features, which capture:\n'
'- **Frequency (F0):** Fundamental frequency / pitch \u2014 depressed individuals often show flatter pitch contours\n'
'- **Energy / Loudness:** Overall vocal energy \u2014 often reduced in depression\n'
'- **Spectral features:** Spectral flux, slope, MFCCs \u2014 voice quality measures\n'
'- **Voice quality:** Jitter (pitch perturbation), shimmer (amplitude perturbation) \u2014 vocal tremor indicators'
))

cells.append(code(
'%%time\n'
'# ---- Load pre-extracted audio features ----\n'
'\n'
'def load_audio_features(pid, data_root):\n'
'    # Load OpenSMILE features and compute session-level statistics.\n'
"    clean_pid = str(pid).replace('_P', '').replace('_p', '').split('.')[0].strip()\n"
'    folder = os.path.join(data_root, PARTICIPANT_FOLDER.format(pid=clean_pid))\n'
'    if not os.path.exists(folder):\n'
'        folder = os.path.join(data_root, str(pid))\n'
'\n'
'    afile = AUDIO_FEATURE_FILE.format(pid=clean_pid)\n'
'    apath = os.path.join(folder, afile)\n'
'    if not os.path.exists(apath):\n'
"        apath = os.path.join(folder, 'features', afile)\n"
'    if not os.path.exists(apath):\n'
"        apath = os.path.join(folder, 'Features', afile)\n"
'\n'
'    if not os.path.exists(apath):\n'
"        possible = glob.glob(os.path.join(folder, '**', '*openSMILE*'), recursive=True) + \\\n"
"                   glob.glob(os.path.join(folder, '**', '*OpenSMILE*'), recursive=True) + \\\n"
"                   glob.glob(os.path.join(folder, '**', '*egemaps*'), recursive=True) + \\\n"
"                   glob.glob(os.path.join(folder, '**', '*eGeMAPS*'), recursive=True) + \\\n"
"                   glob.glob(os.path.join(folder, '**', '*covar*'), recursive=True)\n"
'        if possible:\n'
'            apath = possible[0]\n'
'        else:\n'
'            return None\n'
'\n'
'    try:\n'
'        try:\n'
"            df = pd.read_csv(apath, sep=';')\n"
'            if df.shape[1] <= 1:\n'
"                df = pd.read_csv(apath, sep=',')\n"
'        except Exception:\n'
"            df = pd.read_csv(apath, sep=None, engine='python')\n"
'\n'
'        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n'
'        df_num = df[numeric_cols]\n'
'\n'
'        stats_dict = {}\n'
'        for col in df_num.columns:\n'
'            vals = df_num[col].dropna()\n'
'            if len(vals) > 0:\n'
"                stats_dict[f'audio_{col}_mean'] = vals.mean()\n"
"                stats_dict[f'audio_{col}_std'] = vals.std()\n"
"                stats_dict[f'audio_{col}_min'] = vals.min()\n"
"                stats_dict[f'audio_{col}_max'] = vals.max()\n"
'\n'
'        return stats_dict\n'
'    except Exception:\n'
'        return None\n'
'\n'
'audio_features = {}\n'
'missing_audio = []\n'
"for pid in tqdm(df_labels['Participant_ID'], desc='Loading audio features'):\n"
'    feats = load_audio_features(pid, DATA_ROOT)\n'
'    if feats:\n'
'        audio_features[pid] = feats\n'
'    else:\n'
'        missing_audio.append(pid)\n'
'\n'
"df_audio = pd.DataFrame.from_dict(audio_features, orient='index')\n"
"df_audio.index.name = 'Participant_ID'\n"
'df_audio = df_audio.reset_index()\n'
'\n'
'print(f"\\n\u2705 Loaded audio features for {len(df_audio)} / {len(df_labels)} participants")\n'
'print(f"   Feature dimensions: {len(df_audio.columns)-1}")\n'
'if missing_audio:\n'
'    print(f"\u26a0\ufe0f Missing audio for {len(missing_audio)} participants")'
))

cells.append(md('#### Audio Feature Visualization'))

cells.append(code(
"df_audio_viz = df_audio.merge(df_labels[['Participant_ID', 'PHQ_Score', 'PHQ_Binary']], on='Participant_ID')\n"
"audio_feat_cols = [c for c in df_audio.columns if c.startswith('audio_') and c.endswith('_mean')]\n"
'\n'
'if len(audio_feat_cols) > 0:\n'
"    audio_corrs = df_audio_viz[audio_feat_cols + ['PHQ_Score']].corr()['PHQ_Score'].drop('PHQ_Score')\n"
'    top_audio = audio_corrs.abs().nlargest(6).index.tolist()\n'
'\n'
'    fig, axes = plt.subplots(2, 3, figsize=(16, 9))\n'
'    for idx, feat in enumerate(top_audio):\n'
'        ax = axes[idx // 3][idx % 3]\n'
"        for label, color, name in [(0, C['not_dep'], 'Not Dep'), (1, C['dep'], 'Depressed')]:\n"
"            subset = df_audio_viz[df_audio_viz['PHQ_Binary'] == label][feat].dropna()\n"
'            ax.violinplot([subset.values], positions=[label], showmeans=True)\n'
'        ax.set_xticks([0, 1])\n'
"        ax.set_xticklabels(['Not Dep', 'Depressed'])\n"
"        short_name = feat.replace('audio_', '').replace('_mean', '')\n"
"        ax.set_title(f'{short_name}\\n(r={audio_corrs[feat]:.3f})', fontsize=10)\n"
'\n'
"    fig.suptitle('Top 6 Audio Features Correlated with Depression', fontsize=14, fontweight='bold')\n"
'    plt.tight_layout()\n'
'    plt.show()\n'
'else:\n'
'    print("\u26a0\ufe0f No audio features found. Check the audio feature loading cell.")'
))

# --- 4c: VISUAL ---
cells.append(md(
'### \U0001f441\ufe0f 4c \u2014 Visual / Facial Features\n'
'\n'
'We use **pre-extracted OpenFace** features, which capture:\n'
'\n'
'| Feature Group | Description | Depression Relevance |\n'
'|---|---|---|\n'
'| **Action Units (AUs)** | Facial muscle activations (FACS) | Reduced AU6 (cheek raise), AU12 (smile); increased AU4 (brow lower) |\n'
'| **Head Pose** | Roll, pitch, yaw angles | Less head movement, more downward gaze |\n'
'| **Eye Gaze** | Gaze direction vectors | Increased gaze aversion |'
))

cells.append(code(
'%%time\n'
'def load_visual_features(pid, data_root):\n'
'    # Load OpenFace features and compute session-level statistics.\n'
"    clean_pid = str(pid).replace('_P', '').replace('_p', '').split('.')[0].strip()\n"
'    folder = os.path.join(data_root, PARTICIPANT_FOLDER.format(pid=clean_pid))\n'
'    if not os.path.exists(folder):\n'
'        folder = os.path.join(data_root, str(pid))\n'
'\n'
'    vfile = VISUAL_FEATURE_FILE.format(pid=clean_pid)\n'
'    vpath = os.path.join(folder, vfile)\n'
'    if not os.path.exists(vpath):\n'
"        vpath = os.path.join(folder, 'features', vfile)\n"
'    if not os.path.exists(vpath):\n'
"        vpath = os.path.join(folder, 'Features', vfile)\n"
'\n'
'    if not os.path.exists(vpath):\n'
"        possible = glob.glob(os.path.join(folder, '**', '*openFace*'), recursive=True) + \\\n"
"                   glob.glob(os.path.join(folder, '**', '*OpenFace*'), recursive=True) + \\\n"
"                   glob.glob(os.path.join(folder, '**', '*Pose*gaze*AU*'), recursive=True) + \\\n"
"                   glob.glob(os.path.join(folder, '**', '*AU*'), recursive=True)\n"
'        if possible:\n'
'            vpath = possible[0]\n'
'        else:\n'
'            return None\n'
'\n'
'    try:\n'
'        try:\n'
"            df = pd.read_csv(vpath, sep=';')\n"
'            if df.shape[1] <= 1:\n'
"                df = pd.read_csv(vpath, sep=',')\n"
'        except Exception:\n'
"            df = pd.read_csv(vpath, sep=None, engine='python')\n"
'\n'
'        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()\n'
'        df_num = df[numeric_cols]\n'
'\n'
'        stats_dict = {}\n'
'        for col in df_num.columns:\n'
'            vals = df_num[col].dropna()\n'
'            if len(vals) > 0:\n'
"                stats_dict[f'vis_{col}_mean'] = vals.mean()\n"
"                stats_dict[f'vis_{col}_std'] = vals.std()\n"
'\n'
'        return stats_dict\n'
'    except Exception:\n'
'        return None\n'
'\n'
'visual_features = {}\n'
'missing_visual = []\n'
"for pid in tqdm(df_labels['Participant_ID'], desc='Loading visual features'):\n"
'    feats = load_visual_features(pid, DATA_ROOT)\n'
'    if feats:\n'
'        visual_features[pid] = feats\n'
'    else:\n'
'        missing_visual.append(pid)\n'
'\n'
"df_visual = pd.DataFrame.from_dict(visual_features, orient='index')\n"
"df_visual.index.name = 'Participant_ID'\n"
'df_visual = df_visual.reset_index()\n'
'\n'
'print(f"\\n\u2705 Loaded visual features for {len(df_visual)} / {len(df_labels)} participants")\n'
'print(f"   Feature dimensions: {len(df_visual.columns)-1}")\n'
'if missing_visual:\n'
'    print(f"\u26a0\ufe0f Missing visual for {len(missing_visual)} participants")'
))

cells.append(md('#### Visual Feature Visualization'))

cells.append(code(
"df_vis_viz = df_visual.merge(df_labels[['Participant_ID', 'PHQ_Score', 'PHQ_Binary']], on='Participant_ID')\n"
"au_feats = [c for c in df_visual.columns if 'AU' in c and c.endswith('_mean')]\n"
'\n'
'if len(au_feats) > 0:\n'
"    dep_means = df_vis_viz[df_vis_viz['PHQ_Binary'] == 1][au_feats].mean()\n"
"    not_dep_means = df_vis_viz[df_vis_viz['PHQ_Binary'] == 0][au_feats].mean()\n"
"    clean_names = [c.replace('vis_', '').replace('_mean', '') for c in au_feats[:12]]\n"
'\n'
'    fig, ax = plt.subplots(figsize=(14, 5))\n'
'    x = np.arange(len(clean_names[:12]))\n'
'    width = 0.35\n'
"    ax.bar(x - width/2, not_dep_means.values[:12], width, label='Not Depressed',\n"
"           color=C['not_dep'], edgecolor='white')\n"
"    ax.bar(x + width/2, dep_means.values[:12], width, label='Depressed',\n"
"           color=C['dep'], edgecolor='white')\n"
'    ax.set_xticks(x)\n'
"    ax.set_xticklabels(clean_names[:12], rotation=45, ha='right', fontsize=9)\n"
"    ax.set_ylabel('Mean Activation')\n"
"    ax.set_title('Action Unit Activation: Depressed vs. Not Depressed')\n"
'    ax.legend()\n'
'    plt.tight_layout()\n'
'    plt.show()\n'
'else:\n'
'    print("\u26a0\ufe0f No AU features found. Check the visual feature loading cell.")'
))

# =================================================================
# SECTION 5 — Feature Fusion & Data Preparation
# =================================================================

cells.append(md(
'---\n'
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 20px 24px; border-radius: 12px; color: white;">\n'
'<h2 style="margin:0;">\U0001f517 Section 5 \u2014 Feature Fusion & Data Preparation</h2>\n'
'<p style="margin:4px 0 0 0; opacity:0.8;">\u23f1\ufe0f ~3 minutes</p>\n'
'</div>'
))

cells.append(md(
'### \u26a0\ufe0f Critical: Avoiding Subject Leakage\n'
'\n'
'<div style="background: #fce4ec; padding: 16px 20px; border-left: 5px solid #e91e63; border-radius: 4px;">\n'
'<b>Subject leakage</b> is the #1 methodological pitfall in depression detection research.\n'
'\n'
'It occurs when data from the <b>same participant</b> appears in both training and test sets. The model memorizes individual speaking styles instead of learning generalizable depression markers.\n'
'\n'
'<br><br>\n'
'<b>\u2705 Correct:</b> Split at the <b>participant level</b> (which the E-DAIC train/dev/test splits already do)<br>\n'
'<b>\u274c Wrong:</b> Random splitting of utterances or time segments across participants\n'
'</div>'
))

cells.append(code(
'# ---- Combine all feature modalities ----\n'
"df_text = df_ling.merge(df_sbert, on='Participant_ID', how='outer')\n"
'print(f"Text features: {len(df_text)} participants, {len(df_text.columns)-1} features")\n'
'\n'
"df_all = df_labels[['Participant_ID', 'PHQ_Score', 'PHQ_Binary', 'Split', 'Gender']].copy()\n"
"df_all = df_all.merge(df_text, on='Participant_ID', how='left')\n"
"df_all = df_all.merge(df_audio, on='Participant_ID', how='left')\n"
"df_all = df_all.merge(df_visual, on='Participant_ID', how='left')\n"
'\n'
"meta_cols = ['Participant_ID', 'PHQ_Score', 'PHQ_Binary', 'Split', 'Gender']\n"
"text_feat_cols = [c for c in df_text.columns if c != 'Participant_ID']\n"
"audio_feat_cols = [c for c in df_audio.columns if c != 'Participant_ID']\n"
"visual_feat_cols = [c for c in df_visual.columns if c != 'Participant_ID']\n"
'all_feat_cols = text_feat_cols + audio_feat_cols + visual_feat_cols\n'
'\n'
'print(f"\\n\U0001f4ca Feature Summary:")\n'
'print(f"   Text features:   {len(text_feat_cols)}")\n'
'print(f"   Audio features:  {len(audio_feat_cols)}")\n'
'print(f"   Visual features: {len(visual_feat_cols)}")\n'
'print(f"   Total (fused):   {len(all_feat_cols)}")\n'
'print(f"   Participants:    {len(df_all)}")'
))

cells.append(code(
'# ---- Handle missing values ----\n'
"train_mask = df_all['Split'] == 'train'\n"
'for col in all_feat_cols:\n'
'    if df_all[col].isna().any():\n'
'        train_mean = df_all.loc[train_mask, col].mean()\n'
'        df_all[col] = df_all[col].fillna(train_mean if not np.isnan(train_mean) else 0)\n'
'\n'
'missing_pct = df_all[all_feat_cols].isna().mean().mean() * 100\n'
'print(f"\u2705 Missing values after imputation: {missing_pct:.2f}%")\n'
'\n'
'# ---- Split into train / dev / test ----\n'
"df_train_split = df_all[df_all['Split'] == 'train']\n"
"df_dev_split   = df_all[df_all['Split'] == 'dev']\n"
"df_test_split  = df_all[df_all['Split'] == 'test']\n"
'\n'
'# ---- Feature scaling (fit on train only!) ----\n'
'feature_sets = {}\n'
"for name, cols in [('text', text_feat_cols), ('audio', audio_feat_cols),\n"
"                   ('visual', visual_feat_cols), ('fused', all_feat_cols)]:\n"
'    if len(cols) > 0:\n'
'        sc = StandardScaler()\n'
'        X_tr = sc.fit_transform(df_train_split[cols].values)\n'
'        X_dv = sc.transform(df_dev_split[cols].values)\n'
'        X_te = sc.transform(df_test_split[cols].values)\n'
'        feature_sets[name] = (X_tr, X_dv, X_te)\n'
'        print(f"   {name:8s} -> train: {X_tr.shape}, dev: {X_dv.shape}, test: {X_te.shape}")\n'
'\n'
'# Targets\n'
"y_train_reg = df_train_split['PHQ_Score'].values.astype(float)\n"
"y_dev_reg   = df_dev_split['PHQ_Score'].values.astype(float)\n"
"y_test_reg  = df_test_split['PHQ_Score'].values.astype(float)\n"
'\n'
"y_train_cls = df_train_split['PHQ_Binary'].values.astype(int)\n"
"y_dev_cls   = df_dev_split['PHQ_Binary'].values.astype(int)\n"
"y_test_cls  = df_test_split['PHQ_Binary'].values.astype(int)\n"
'\n'
'print(f"\\n\u2705 Data prepared for modeling!")\n'
'print(f"   Regression target range: [{y_train_reg.min():.0f}, {y_train_reg.max():.0f}]")\n'
'print(f"   Classification balance (train): {y_train_cls.sum()}/{len(y_train_cls)} depressed")'
))

# =================================================================
# SECTION 6 — Classical ML: Regression
# =================================================================

cells.append(md(
'---\n'
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 20px 24px; border-radius: 12px; color: white;">\n'
'<h2 style="margin:0;">\U0001f4c8 Section 6 \u2014 Classical ML: PHQ-8 Regression</h2>\n'
'<p style="margin:4px 0 0 0; opacity:0.8;">\u23f1\ufe0f ~6 minutes \u2014 Predict depression severity (PHQ-8 score, 0\u201324)</p>\n'
'</div>'
))

cells.append(md(
'### Evaluation Metrics for Regression\n'
'\n'
'| Metric | Formula | Interpretation |\n'
'|---|---|---|\n'
'| **RMSE** | $\\sqrt{\\frac{1}{n}\\sum(y_i - \\hat{y}_i)^2}$ | Lower is better; in PHQ-8 units |\n'
'| **MAE** | $\\frac{1}{n}\\sum|y_i - \\hat{y}_i|$ | Average absolute error in PHQ-8 units |\n'
'| **CCC** | Concordance Correlation Coefficient | Agreement between predicted and actual (range: -1 to 1) |\n'
'\n'
'> \U0001f3af **CCC** is the standard metric in the AVEC challenges because it captures both correlation and calibration.'
))

cells.append(code(
'# ---- Define Concordance Correlation Coefficient (CCC) ----\n'
'\n'
'def concordance_correlation_coefficient(y_true, y_pred):\n'
'    # Compute Lin\'s Concordance Correlation Coefficient.\n'
'    y_true = np.asarray(y_true, dtype=float)\n'
'    y_pred = np.asarray(y_pred, dtype=float)\n'
'    mean_true = np.mean(y_true)\n'
'    mean_pred = np.mean(y_pred)\n'
'    var_true = np.var(y_true)\n'
'    var_pred = np.var(y_pred)\n'
'    covar = np.mean((y_true - mean_true) * (y_pred - mean_pred))\n'
'    ccc = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-8)\n'
'    return ccc\n'
'\n'
'def evaluate_regression(y_true, y_pred):\n'
'    # Compute all regression metrics.\n'
'    return {\n'
"        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),\n"
"        'MAE': mean_absolute_error(y_true, y_pred),\n"
"        'CCC': concordance_correlation_coefficient(y_true, y_pred),\n"
'    }\n'
'\n'
'print("CCC test (perfect prediction):", concordance_correlation_coefficient([1,2,3], [1,2,3]))\n'
'print("CCC test (constant prediction):", concordance_correlation_coefficient([1,2,3], [2,2,2]))'
))

cells.append(md('### 6.1 \u2014 Train Regression Models'))

cells.append(code(
'%%time\n'
'reg_models = {\n'
'    \'Ridge\': Ridge(alpha=1.0),\n'
'    \'Random Forest\': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),\n'
'    \'XGBoost\': xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1,\n'
'                                 random_state=SEED, verbosity=0),\n'
'}\n'
'\n'
'reg_results = []\n'
'best_reg_pred_dev = None\n'
'best_reg_model = None\n'
'\n'
'for modality, (X_tr, X_dv, X_te) in feature_sets.items():\n'
'    for model_name, model in reg_models.items():\n'
'        m = clone(model)\n'
'        m.fit(X_tr, y_train_reg)\n'
'        y_pred_dev = np.clip(m.predict(X_dv), 0, 24)\n'
'        metrics = evaluate_regression(y_dev_reg, y_pred_dev)\n'
"        metrics['Model'] = model_name\n"
"        metrics['Modality'] = modality\n"
'        reg_results.append(metrics)\n'
'\n'
"        if modality == 'fused' and model_name == 'XGBoost':\n"
'            best_reg_pred_dev = y_pred_dev\n'
'            best_reg_pred_test = np.clip(m.predict(X_te), 0, 24)\n'
'            best_reg_model = m\n'
'\n'
'df_reg_results = pd.DataFrame(reg_results)\n'
'print("\u2705 All regression models trained!")'
))

cells.append(md('### 6.2 \u2014 Regression Results'))

cells.append(code(
"pivot_rmse = df_reg_results.pivot(index='Model', columns='Modality', values='RMSE').round(3)\n"
"pivot_ccc  = df_reg_results.pivot(index='Model', columns='Modality', values='CCC').round(3)\n"
"col_order = [c for c in ['text', 'audio', 'visual', 'fused'] if c in pivot_rmse.columns]\n"
'pivot_rmse = pivot_rmse[col_order]\n'
'pivot_ccc  = pivot_ccc[col_order]\n'
'\n'
'print("\U0001f4ca RMSE (lower is better) -- Dev Set")\n'
'print("=" * 60)\n'
"display(pivot_rmse.style.highlight_min(axis=None, color='#C8E6C9')\n"
'        .format("{:.3f}").set_caption("RMSE by Model x Modality"))\n'
'\n'
'print("\\n\U0001f4ca CCC (higher is better) -- Dev Set")\n'
'print("=" * 60)\n'
"display(pivot_ccc.style.highlight_max(axis=None, color='#C8E6C9')\n"
'        .format("{:.3f}").set_caption("CCC by Model x Modality"))'
))

cells.append(md('### 6.3 \u2014 Regression Visualizations'))

cells.append(code(
'fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n'
'\n'
'# (a) Predicted vs Actual\n'
'ax = axes[0]\n'
'ax.scatter(y_dev_reg, best_reg_pred_dev, alpha=0.6, s=50,\n'
"           c=C['blue'], edgecolor='white', label='Dev set')\n"
"ax.plot([0, 24], [0, 24], 'k--', alpha=0.5, label='Perfect prediction')\n"
"ax.set_xlabel('Actual PHQ-8 Score')\n"
"ax.set_ylabel('Predicted PHQ-8 Score')\n"
'rmse_val = np.sqrt(mean_squared_error(y_dev_reg, best_reg_pred_dev))\n'
'ccc_val = concordance_correlation_coefficient(y_dev_reg, best_reg_pred_dev)\n'
"ax.set_title(f'(a) XGBoost Fused -- Dev\\nRMSE={rmse_val:.2f}, CCC={ccc_val:.3f}')\n"
'ax.legend()\n'
'ax.set_xlim(-1, 25)\n'
'ax.set_ylim(-1, 25)\n'
'\n'
'# (b) Residual distribution\n'
'ax = axes[1]\n'
'residuals = y_dev_reg - best_reg_pred_dev\n'
"ax.hist(residuals, bins=20, color=C['purple'], edgecolor='white', alpha=0.8)\n"
"ax.axvline(x=0, color='black', linestyle='--', alpha=0.5)\n"
"ax.set_xlabel('Residual (Actual - Predicted)')\n"
"ax.set_ylabel('Count')\n"
"ax.set_title(f'(b) Residual Distribution\\nMean={residuals.mean():.2f}, Std={residuals.std():.2f}')\n"
'\n'
'# (c) Feature importance\n'
'ax = axes[2]\n'
"if hasattr(best_reg_model, 'feature_importances_'):\n"
'    importances = best_reg_model.feature_importances_\n'
'    feat_names = all_feat_cols\n'
'    top_idx = np.argsort(importances)[-15:]\n'
"    top_names = [feat_names[i].replace('audio_', 'A:').replace('vis_', 'V:')\n"
"                 .replace('sbert_', 'T:emb').replace('_mean', '')[:25]\n"
'                 for i in top_idx]\n'
"    ax.barh(range(len(top_idx)), importances[top_idx], color=C['teal'], edgecolor='white')\n"
'    ax.set_yticks(range(len(top_idx)))\n'
'    ax.set_yticklabels(top_names, fontsize=8)\n'
"    ax.set_xlabel('Feature Importance')\n"
"    ax.set_title('(c) Top 15 Features (XGBoost)')\n"
'\n'
'plt.tight_layout()\n'
'plt.show()'
))

# =================================================================
# SECTION 7 — Classification
# =================================================================

cells.append(md(
'---\n'
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 20px 24px; border-radius: 12px; color: white;">\n'
'<h2 style="margin:0;">\U0001f3f7\ufe0f Section 7 \u2014 Classical ML: Binary Classification</h2>\n'
'<p style="margin:4px 0 0 0; opacity:0.8;">\u23f1\ufe0f ~5 minutes \u2014 Depressed (PHQ-8 \u2265 10) vs. Not Depressed</p>\n'
'</div>'
))

cells.append(code(
'%%time\n'
'cls_models = {\n'
"    'Logistic Reg.': LogisticRegression(\n"
"        class_weight='balanced', max_iter=1000, random_state=SEED, C=1.0),\n"
"    'Random Forest': RandomForestClassifier(\n"
"        n_estimators=100, max_depth=10, class_weight='balanced',\n"
'        random_state=SEED, n_jobs=-1),\n'
"    'XGBoost': xgb.XGBClassifier(\n"
'        n_estimators=100, max_depth=5, learning_rate=0.1,\n'
'        scale_pos_weight=(y_train_cls == 0).sum() / max((y_train_cls == 1).sum(), 1),\n'
"        random_state=SEED, verbosity=0, eval_metric='logloss'),\n"
'}\n'
'\n'
'cls_results = []\n'
'cls_models_fitted = {}\n'
'\n'
'for modality, (X_tr, X_dv, X_te) in feature_sets.items():\n'
'    for model_name, model in cls_models.items():\n'
'        m = clone(model)\n'
'        m.fit(X_tr, y_train_cls)\n'
'        y_pred = m.predict(X_dv)\n'
"        y_prob = m.predict_proba(X_dv)[:, 1] if hasattr(m, 'predict_proba') else y_pred.astype(float)\n"
'\n'
'        metrics = {\n'
"            'Model': model_name,\n"
"            'Modality': modality,\n"
"            'Accuracy': accuracy_score(y_dev_cls, y_pred),\n"
"            'F1': f1_score(y_dev_cls, y_pred, zero_division=0),\n"
"            'Precision': precision_score(y_dev_cls, y_pred, zero_division=0),\n"
"            'Recall': recall_score(y_dev_cls, y_pred, zero_division=0),\n"
"            'AUC-ROC': roc_auc_score(y_dev_cls, y_prob) if len(np.unique(y_dev_cls)) > 1 else 0,\n"
'        }\n'
'        cls_results.append(metrics)\n'
'\n'
"        if modality == 'fused':\n"
'            cls_models_fitted[model_name] = (m, y_prob)\n'
'\n'
'df_cls_results = pd.DataFrame(cls_results)\n'
'print("\u2705 All classification models trained!")'
))

cells.append(md('### 7.1 \u2014 Classification Results'))

cells.append(code(
"fused_results = df_cls_results[df_cls_results['Modality'] == 'fused'].set_index('Model')\n"
"fused_results = fused_results.drop(columns='Modality')\n"
'\n'
'print("\U0001f4ca Classification Results -- Fused Features, Dev Set")\n'
'print("=" * 70)\n'
'display(fused_results.style\n'
"        .highlight_max(axis=0, color='#C8E6C9')\n"
'        .format("{:.3f}")\n'
'        .set_caption("Classification Metrics (Fused Features)"))\n'
'\n'
'print("\\n\U0001f4ca F1 Score by Modality")\n'
"pivot_f1 = df_cls_results.pivot(index='Model', columns='Modality', values='F1').round(3)\n"
"col_order = [c for c in ['text', 'audio', 'visual', 'fused'] if c in pivot_f1.columns]\n"
'display(pivot_f1[col_order].style\n'
"        .highlight_max(axis=None, color='#C8E6C9')\n"
'        .format("{:.3f}"))'
))

cells.append(md('### 7.2 \u2014 Confusion Matrices & ROC Curves'))

cells.append(code(
'fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n'
'\n'
"X_dv_fused = feature_sets['fused'][1]\n"
'for idx, (model_name, (model, _)) in enumerate(cls_models_fitted.items()):\n'
'    ax = axes[idx]\n'
'    y_pred = model.predict(X_dv_fused)\n'
'    cm = confusion_matrix(y_dev_cls, y_pred)\n'
"    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,\n"
"                xticklabels=['Not Dep', 'Depressed'],\n"
"                yticklabels=['Not Dep', 'Depressed'],\n"
"                cbar=False, annot_kws={'size': 14})\n"
"    ax.set_ylabel('Actual')\n"
"    ax.set_xlabel('Predicted')\n"
'    acc = accuracy_score(y_dev_cls, y_pred)\n'
'    f1 = f1_score(y_dev_cls, y_pred, zero_division=0)\n'
"    ax.set_title(f'{model_name}\\nAcc={acc:.2f}, F1={f1:.2f}')\n"
'\n'
"plt.suptitle('Confusion Matrices -- Fused Features, Dev Set', fontsize=14, fontweight='bold')\n"
'plt.tight_layout()\n'
'plt.show()'
))

cells.append(code(
'fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n'
'\n'
'# (a) ROC Curves\n'
'ax = axes[0]\n'
"colors_iter = [C['blue'], C['orange'], C['green']]\n"
'for idx, (model_name, (model, y_prob)) in enumerate(cls_models_fitted.items()):\n'
'    fpr, tpr, _ = roc_curve(y_dev_cls, y_prob)\n'
'    auc = roc_auc_score(y_dev_cls, y_prob)\n'
"    ax.plot(fpr, tpr, color=colors_iter[idx], lw=2,\n"
"            label=f'{model_name} (AUC={auc:.3f})')\n"
'\n'
"ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='Random (AUC=0.5)')\n"
"ax.set_xlabel('False Positive Rate')\n"
"ax.set_ylabel('True Positive Rate')\n"
"ax.set_title('(a) ROC Curves -- Fused Features')\n"
'ax.legend(fontsize=9)\n'
'ax.set_xlim(-0.02, 1.02)\n'
'ax.set_ylim(-0.02, 1.02)\n'
'\n'
'# (b) Precision-Recall Curves\n'
'ax = axes[1]\n'
'for idx, (model_name, (model, y_prob)) in enumerate(cls_models_fitted.items()):\n'
'    precision, recall, _ = precision_recall_curve(y_dev_cls, y_prob)\n'
'    ax.plot(recall, precision, color=colors_iter[idx], lw=2, label=model_name)\n'
'\n'
'baseline = y_dev_cls.sum() / len(y_dev_cls)\n'
"ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.4, label=f'Baseline ({baseline:.2f})')\n"
"ax.set_xlabel('Recall')\n"
"ax.set_ylabel('Precision')\n"
"ax.set_title('(b) Precision-Recall Curves')\n"
'ax.legend(fontsize=9)\n'
'\n'
'plt.tight_layout()\n'
'plt.show()'
))

# =================================================================
# SECTION 8 — Deep Learning: MLP
# =================================================================

cells.append(md(
'---\n'
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 20px 24px; border-radius: 12px; color: white;">\n'
'<h2 style="margin:0;">\U0001f9ec Section 8 \u2014 Deep Learning: Multi-Layer Perceptron</h2>\n'
'<p style="margin:4px 0 0 0; opacity:0.8;">\u23f1\ufe0f ~5 minutes \u2014 Neural network on fused features</p>\n'
'</div>\n'
'\n'
'We use a lightweight **MLP (Multi-Layer Perceptron)** on the fused feature vector. While architectures like LSTMs and Transformers can model sequential data, the MLP works well on our session-level aggregated features and trains in seconds on Colab\'s T4 GPU.\n'
'\n'
'> \U0001f4a1 See **Appendix A** at the end for an LSTM implementation you can explore after the workshop.'
))

cells.append(code(
'# ---- Define Multi-Task MLP ----\n'
'class DepressionMLP(nn.Module):\n'
'    def __init__(self, input_dim, hidden_dims=(256, 128, 64), dropout=0.3):\n'
'        super().__init__()\n'
'        layers = []\n'
'        prev_dim = input_dim\n'
'        for h_dim in hidden_dims:\n'
'            layers.extend([\n'
'                nn.Linear(prev_dim, h_dim),\n'
'                nn.BatchNorm1d(h_dim),\n'
'                nn.ReLU(),\n'
'                nn.Dropout(dropout),\n'
'            ])\n'
'            prev_dim = h_dim\n'
'        self.backbone = nn.Sequential(*layers)\n'
'        self.reg_head = nn.Linear(prev_dim, 1)\n'
'        self.cls_head = nn.Linear(prev_dim, 2)\n'
'\n'
'    def forward(self, x):\n'
'        features = self.backbone(x)\n'
'        reg_out = self.reg_head(features).squeeze(-1)\n'
'        cls_out = self.cls_head(features)\n'
'        return reg_out, cls_out\n'
'\n'
"X_tr_fused, X_dv_fused, X_te_fused = feature_sets['fused']\n"
'input_dim = X_tr_fused.shape[1]\n'
'model_mlp = DepressionMLP(input_dim, hidden_dims=(256, 128, 64), dropout=0.3).to(device)\n'
'\n'
'print(f"\U0001f9ec MLP Architecture:")\n'
'print(f"   Input dim:  {input_dim}")\n'
'print(f"   Parameters: {sum(p.numel() for p in model_mlp.parameters()):,}")\n'
'print(model_mlp)'
))

cells.append(code(
'%%time\n'
'# ---- Training Loop ----\n'
'train_dataset = TensorDataset(\n'
'    torch.FloatTensor(X_tr_fused),\n'
'    torch.FloatTensor(y_train_reg),\n'
'    torch.LongTensor(y_train_cls)\n'
')\n'
'dev_dataset = TensorDataset(\n'
'    torch.FloatTensor(X_dv_fused),\n'
'    torch.FloatTensor(y_dev_reg),\n'
'    torch.LongTensor(y_dev_cls)\n'
')\n'
'\n'
'train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)\n'
'\n'
'reg_criterion = nn.MSELoss()\n'
'class_weights = torch.FloatTensor(\n'
'    [1.0, (y_train_cls == 0).sum() / max((y_train_cls == 1).sum(), 1)]\n'
').to(device)\n'
'cls_criterion = nn.CrossEntropyLoss(weight=class_weights)\n'
'\n'
'optimizer = optim.Adam(model_mlp.parameters(), lr=1e-3, weight_decay=1e-4)\n'
'scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)\n'
'\n'
'EPOCHS = 100\n'
'ALPHA = 0.5\n'
"history = {'train_loss': [], 'dev_loss': [], 'dev_rmse': [], 'dev_ccc': [], 'dev_f1': []}\n"
"best_dev_loss = float('inf')\n"
'patience_counter = 0\n'
'PATIENCE = 20\n'
'\n'
'for epoch in range(EPOCHS):\n'
'    model_mlp.train()\n'
'    train_losses = []\n'
'    for X_batch, y_reg_batch, y_cls_batch in train_loader:\n'
'        X_batch = X_batch.to(device)\n'
'        y_reg_batch = y_reg_batch.to(device)\n'
'        y_cls_batch = y_cls_batch.to(device)\n'
'        optimizer.zero_grad()\n'
'        reg_pred, cls_pred = model_mlp(X_batch)\n'
'        loss = ALPHA * reg_criterion(reg_pred, y_reg_batch) + (1-ALPHA) * cls_criterion(cls_pred, y_cls_batch)\n'
'        loss.backward()\n'
'        optimizer.step()\n'
'        train_losses.append(loss.item())\n'
'\n'
'    model_mlp.eval()\n'
'    with torch.no_grad():\n'
'        X_dv_t = torch.FloatTensor(X_dv_fused).to(device)\n'
'        reg_pred_dv, cls_pred_dv = model_mlp(X_dv_t)\n'
'        reg_pred_np = reg_pred_dv.cpu().numpy()\n'
'        cls_pred_np = cls_pred_dv.argmax(dim=1).cpu().numpy()\n'
'\n'
'        dev_loss = (ALPHA * reg_criterion(reg_pred_dv, torch.FloatTensor(y_dev_reg).to(device)) +\n'
'                    (1-ALPHA) * cls_criterion(cls_pred_dv, torch.LongTensor(y_dev_cls).to(device))).item()\n'
'        dev_rmse = np.sqrt(mean_squared_error(y_dev_reg, np.clip(reg_pred_np, 0, 24)))\n'
'        dev_ccc = concordance_correlation_coefficient(y_dev_reg, np.clip(reg_pred_np, 0, 24))\n'
'        dev_f1 = f1_score(y_dev_cls, cls_pred_np, zero_division=0)\n'
'\n'
"    history['train_loss'].append(np.mean(train_losses))\n"
"    history['dev_loss'].append(dev_loss)\n"
"    history['dev_rmse'].append(dev_rmse)\n"
"    history['dev_ccc'].append(dev_ccc)\n"
"    history['dev_f1'].append(dev_f1)\n"
'    scheduler.step(dev_loss)\n'
'\n'
'    if dev_loss < best_dev_loss:\n'
'        best_dev_loss = dev_loss\n'
'        best_state = {k: v.cpu().clone() for k, v in model_mlp.state_dict().items()}\n'
'        patience_counter = 0\n'
'    else:\n'
'        patience_counter += 1\n'
'\n'
'    if patience_counter >= PATIENCE:\n'
'        print(f"\u23f9\ufe0f Early stopping at epoch {epoch+1}")\n'
'        break\n'
'\n'
'    if (epoch + 1) % 20 == 0:\n'
'        print(f"  Epoch {epoch+1:3d} | Train Loss: {np.mean(train_losses):.4f} | "\n'
'              f"Dev RMSE: {dev_rmse:.3f} | Dev CCC: {dev_ccc:.3f} | Dev F1: {dev_f1:.3f}")\n'
'\n'
'model_mlp.load_state_dict(best_state)\n'
"print(f\"\\n\u2705 Training complete! Best dev loss at epoch {len(history['train_loss']) - patience_counter}\")"
))

cells.append(md('### 8.1 \u2014 Training Curves'))

cells.append(code(
'fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))\n'
'\n'
'ax = axes[0]\n'
"ax.plot(history['train_loss'], label='Train', color=C['blue'], alpha=0.8)\n"
"ax.plot(history['dev_loss'], label='Dev', color=C['orange'], alpha=0.8)\n"
"ax.set_xlabel('Epoch')\n"
"ax.set_ylabel('Loss')\n"
"ax.set_title('(a) Training & Dev Loss')\n"
'ax.legend()\n'
'\n'
'ax = axes[1]\n'
"ax.plot(history['dev_rmse'], label='RMSE', color=C['red'], alpha=0.8)\n"
'ax2 = ax.twinx()\n'
"ax2.plot(history['dev_ccc'], label='CCC', color=C['green'], alpha=0.8)\n"
"ax.set_xlabel('Epoch')\n"
"ax.set_ylabel('RMSE', color=C['red'])\n"
"ax2.set_ylabel('CCC', color=C['green'])\n"
"ax.set_title('(b) Regression Metrics (Dev)')\n"
'lines1, labels1 = ax.get_legend_handles_labels()\n'
'lines2, labels2 = ax2.get_legend_handles_labels()\n'
"ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')\n"
'\n'
'ax = axes[2]\n'
"ax.plot(history['dev_f1'], label='Dev F1', color=C['purple'], alpha=0.8)\n"
"ax.set_xlabel('Epoch')\n"
"ax.set_ylabel('F1 Score')\n"
"ax.set_title('(c) Classification F1 (Dev)')\n"
'ax.legend()\n'
'\n'
'plt.tight_layout()\n'
'plt.show()'
))

cells.append(md('### 8.2 \u2014 MLP Evaluation'))

cells.append(code(
'model_mlp.eval()\n'
'with torch.no_grad():\n'
'    X_dv_t = torch.FloatTensor(X_dv_fused).to(device)\n'
'    reg_pred, cls_pred = model_mlp(X_dv_t)\n'
'    mlp_reg_pred = np.clip(reg_pred.cpu().numpy(), 0, 24)\n'
'    mlp_cls_pred = cls_pred.argmax(dim=1).cpu().numpy()\n'
'    mlp_cls_prob = torch.softmax(cls_pred, dim=1)[:, 1].cpu().numpy()\n'
'\n'
'mlp_reg_metrics = evaluate_regression(y_dev_reg, mlp_reg_pred)\n'
'print("\U0001f9ec MLP Regression Results (Dev):")\n'
'for k, v in mlp_reg_metrics.items():\n'
'    print(f"   {k}: {v:.3f}")\n'
'\n'
'print(f"\\n\U0001f9ec MLP Classification Results (Dev):")\n'
'print(f"   Accuracy:  {accuracy_score(y_dev_cls, mlp_cls_pred):.3f}")\n'
'print(f"   F1:        {f1_score(y_dev_cls, mlp_cls_pred, zero_division=0):.3f}")\n'
'print(f"   Precision: {precision_score(y_dev_cls, mlp_cls_pred, zero_division=0):.3f}")\n'
'print(f"   Recall:    {recall_score(y_dev_cls, mlp_cls_pred, zero_division=0):.3f}")\n'
'if len(np.unique(y_dev_cls)) > 1:\n'
'    print(f"   AUC-ROC:   {roc_auc_score(y_dev_cls, mlp_cls_prob):.3f}")'
))

# =================================================================
# SECTION 9 — Results & Discussion
# =================================================================

cells.append(md(
'---\n'
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 20px 24px; border-radius: 12px; color: white;">\n'
'<h2 style="margin:0;">\U0001f4ca Section 9 \u2014 Comprehensive Results & Discussion</h2>\n'
'<p style="margin:4px 0 0 0; opacity:0.8;">\u23f1\ufe0f ~4 minutes</p>\n'
'</div>'
))

cells.append(md('### 9.1 \u2014 Master Comparison'))

cells.append(code(
"mlp_reg_row = {'Model': 'MLP (DL)', 'Modality': 'fused',\n"
"               'RMSE': mlp_reg_metrics['RMSE'],\n"
"               'MAE': mlp_reg_metrics['MAE'],\n"
"               'CCC': mlp_reg_metrics['CCC']}\n"
'df_reg_all = pd.concat([df_reg_results, pd.DataFrame([mlp_reg_row])], ignore_index=True)\n'
'\n'
'mlp_cls_row = {\n'
"    'Model': 'MLP (DL)', 'Modality': 'fused',\n"
"    'Accuracy': accuracy_score(y_dev_cls, mlp_cls_pred),\n"
"    'F1': f1_score(y_dev_cls, mlp_cls_pred, zero_division=0),\n"
"    'Precision': precision_score(y_dev_cls, mlp_cls_pred, zero_division=0),\n"
"    'Recall': recall_score(y_dev_cls, mlp_cls_pred, zero_division=0),\n"
"    'AUC-ROC': roc_auc_score(y_dev_cls, mlp_cls_prob) if len(np.unique(y_dev_cls)) > 1 else 0,\n"
'}\n'
'df_cls_all = pd.concat([df_cls_results, pd.DataFrame([mlp_cls_row])], ignore_index=True)\n'
'\n'
"fused_reg = df_reg_all[df_reg_all['Modality'] == 'fused'].set_index('Model')[['RMSE', 'MAE', 'CCC']]\n"
'print("=" * 65)\n'
'print("\U0001f4c8  MASTER COMPARISON -- REGRESSION (Fused Features, Dev Set)")\n'
'print("=" * 65)\n'
'display(fused_reg.style\n'
"        .highlight_min(subset=['RMSE', 'MAE'], color='#C8E6C9')\n"
"        .highlight_max(subset=['CCC'], color='#C8E6C9')\n"
'        .format("{:.3f}"))\n'
'\n'
"fused_cls = df_cls_all[df_cls_all['Modality'] == 'fused'].set_index('Model')\n"
"fused_cls = fused_cls.drop(columns='Modality', errors='ignore')\n"
'print("\\n" + "=" * 65)\n'
'print("\U0001f3f7\ufe0f  MASTER COMPARISON -- CLASSIFICATION (Fused Features, Dev Set)")\n'
'print("=" * 65)\n'
'display(fused_cls.style\n'
"        .highlight_max(axis=0, color='#C8E6C9')\n"
'        .format("{:.3f}"))'
))

cells.append(md('### 9.2 \u2014 Modality Ablation Study'))

cells.append(code(
"ablation_data = df_reg_all[df_reg_all['Model'] == 'XGBoost']\n"
'\n'
'fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n'
'\n'
'ax = axes[0]\n'
"modality_order = [m for m in ['text', 'audio', 'visual', 'fused'] if m in ablation_data['Modality'].values]\n"
"rmse_vals = [ablation_data[ablation_data['Modality'] == m]['RMSE'].values[0] for m in modality_order]\n"
"colors = [C['blue'], C['orange'], C['green'], C['purple']][:len(modality_order)]\n"
"bars = ax.bar(modality_order, rmse_vals, color=colors, edgecolor='white', width=0.6)\n"
"ax.set_ylabel('RMSE (lower is better)')\n"
"ax.set_title('(a) Regression RMSE by Modality (XGBoost)')\n"
'for bar, val in zip(bars, rmse_vals):\n'
'    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,\n'
"            f'{val:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')\n"
'\n'
'ax = axes[1]\n'
"ccc_vals = [ablation_data[ablation_data['Modality'] == m]['CCC'].values[0] for m in modality_order]\n"
"bars = ax.bar(modality_order, ccc_vals, color=colors, edgecolor='white', width=0.6)\n"
"ax.set_ylabel('CCC (higher is better)')\n"
"ax.set_title('(b) Regression CCC by Modality (XGBoost)')\n"
'for bar, val in zip(bars, ccc_vals):\n'
'    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n'
"            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')\n"
'\n'
'plt.tight_layout()\n'
'plt.show()'
))

cells.append(md(
'### 9.3 \u2014 Discussion\n'
'\n'
'<div style="background: #e8f5e9; padding: 16px 20px; border-left: 5px solid #4CAF50; border-radius: 4px;">\n'
'\n'
'**Key Takeaways from our experiments:**\n'
'\n'
'1. **Multimodal fusion** generally outperforms any single modality, confirming that depression manifests across behavioral channels.\n'
'\n'
'2. **Text features** are typically the strongest single modality \u2014 language patterns carry rich information about mental state.\n'
'\n'
'3. **Classical ML models** (especially XGBoost) are competitive with or superior to the MLP on this dataset size (N=275). Deep learning shines with more data.\n'
'\n'
'4. **Class imbalance** is a real challenge \u2014 pay attention to F1 and PR-AUC rather than just accuracy.\n'
'\n'
'</div>\n'
'\n'
'---\n'
'\n'
'<div style="background: #fff3e0; padding: 16px 20px; border-left: 5px solid #FF9800; border-radius: 4px;">\n'
'\n'
'**\u26a0\ufe0f Methodological Pitfalls to Remember:**\n'
'\n'
'1. **Subject leakage:** Never split data at the utterance or segment level. Always split at the participant level.\n'
'\n'
'2. **Interviewer bias:** Models may learn to predict depression based on the interviewer\'s prompts.\n'
'\n'
'3. **Small dataset (N=275):** Results have high variance. Don\'t over-interpret small differences between models.\n'
'\n'
'4. **PHQ-8 is self-report:** It\'s not a clinical diagnosis \u2014 it\'s a screening tool with its own biases.\n'
'\n'
'</div>'
))

# =================================================================
# SECTION 10 — Ethics & Wrap-up
# =================================================================

cells.append(md(
'---\n'
'<div style="background: linear-gradient(135deg, #1a237e 0%, #4a148c 100%); padding: 20px 24px; border-radius: 12px; color: white;">\n'
'<h2 style="margin:0;">\u2696\ufe0f Section 10 \u2014 Ethics, Future Directions & References</h2>\n'
'<p style="margin:4px 0 0 0; opacity:0.8;">\u23f1\ufe0f ~2 minutes</p>\n'
'</div>'
))

cells.append(md(
'### Ethical Considerations in AI for Mental Health\n'
'\n'
'<div style="background: #fce4ec; padding: 16px 20px; border-left: 5px solid #e91e63; border-radius: 4px;">\n'
'\n'
'**\U0001f534 This technology raises critical ethical questions:**\n'
'\n'
'| Concern | Details |\n'
'|---|---|\n'
'| **Not a diagnostic tool** | Models like these are research tools, not FDA-approved diagnostics. They should **never** replace clinical judgment. |\n'
'| **Bias & fairness** | Models trained primarily on English-speaking, Western populations may not generalize. |\n'
'| **Privacy** | Audio/video data is deeply personal. Consent, data protection, and anonymization are paramount. |\n'
'| **Informed consent** | Participants must understand how their data will be used. |\n'
'| **Dual-use risk** | Surveillance, insurance discrimination, employment screening \u2014 automated mental health assessment can be misused. |\n'
'| **Transparency** | "Black box" models are problematic in clinical contexts. Explainability matters. |\n'
'\n'
'</div>\n'
'\n'
'### \u2705 Responsible Development Checklist\n'
'\n'
'- [ ] Is the model being used to **support** clinicians, not replace them?\n'
'- [ ] Have you tested for bias across demographic groups?\n'
'- [ ] Is participant data stored and processed with appropriate privacy safeguards?\n'
'- [ ] Are limitations clearly communicated to all stakeholders?\n'
'- [ ] Does the deployment context have human oversight?'
))

cells.append(md(
'### \U0001f52d Future Directions\n'
'\n'
'1. **Multimodal Transformers** \u2014 End-to-end models that jointly attend to text, audio, and visual features\n'
'2. **Cross-corpus generalization** \u2014 Test on datasets beyond E-DAIC to validate real-world applicability\n'
'3. **Explainable AI (XAI)** \u2014 Attention visualization, SHAP values, and feature attribution for clinical trust\n'
'4. **Longitudinal monitoring** \u2014 Track depression severity over time, not just single-session snapshots\n'
'5. **Multi-task learning** \u2014 Jointly predict depression, anxiety, and PTSD using the PCL-C labels available in E-DAIC\n'
'6. **Large Language Models** \u2014 Leverage GPT/LLaMA-family models for richer text understanding\n'
'\n'
'### \U0001f4da Key References\n'
'\n'
'1. Gratch et al. (2014). *The Distress Analysis Interview Corpus of human and computer interviews.* LREC.\n'
'2. Ringeval et al. (2019). *AVEC 2019 Workshop and Challenge.* ACM MM.\n'
'3. Kroenke et al. (2009). *The PHQ-8 as a measure of current depression.* J Affect Disord.\n'
'4. Cummins et al. (2015). *A review of depression and suicide risk assessment using speech analysis.* Speech Comm.\n'
'5. Muzammel et al. (2021). *End-to-end multimodal clinical depression recognition.* IEEE TAFFC.\n'
'6. Williamson et al. (2016). *Detecting depression using vocal, facial and semantic features.* AVEC.\n'
'7. He et al. (2022). *Deep learning for depression recognition from speech: A review.* arXiv.\n'
'8. Ray et al. (2019). *Multi-level attention network for depression detection.* AVEC Workshop.'
))

cells.append(md(
'### \U0001f3cb\ufe0f Take-Home Exercises\n'
'\n'
'Now that you\'ve completed the workshop, here are some challenges to deepen your understanding:\n'
'\n'
'1. **LSTM on Sequential Text** \u2014 Instead of mean-pooling SBERT embeddings, use per-utterance embeddings as a sequence and train a Bi-LSTM with attention (see Appendix A below)\n'
'2. **PTSD Prediction** \u2014 Use the `PCL-C (PTSD)` column as a target and compare which features predict PTSD vs. depression\n'
'3. **Hyperparameter Tuning** \u2014 Use GridSearchCV or Optuna to systematically tune model hyperparameters\n'
'4. **Subject Leakage Demo** \u2014 Intentionally introduce leakage and observe the inflated metrics (see Appendix B below)\n'
'5. **Feature Interpretation** \u2014 Use SHAP values (`shap` library) to explain individual predictions\n'
'6. **Gender Fairness** \u2014 Evaluate model performance separately for each gender \u2014 are there disparities?\n'
'\n'
'---\n'
'*Thank you for participating in this workshop! \U0001f389*'
))

# =================================================================
# APPENDIX A — LSTM (Take-Home)
# =================================================================

cells.append(md(
'---\n'
'## \U0001f4ce Appendix A \u2014 LSTM on Sequential Text Embeddings (Take-Home)\n'
'\n'
'This appendix shows how to model the **temporal sequence** of utterances using a Bi-LSTM with attention.'
))

cells.append(code(
'# ---- APPENDIX A: Bi-LSTM with Attention (Take-Home Exercise) ----\n'
'# Uncomment and run this section after the workshop\n'
'\n'
'# class AttentionLayer(nn.Module):\n'
'#     def __init__(self, hidden_dim):\n'
'#         super().__init__()\n'
'#         self.attention = nn.Linear(hidden_dim, 1)\n'
'#\n'
'#     def forward(self, lstm_output):\n'
'#         weights = torch.softmax(self.attention(lstm_output).squeeze(-1), dim=1)\n'
'#         context = torch.bmm(weights.unsqueeze(1), lstm_output).squeeze(1)\n'
'#         return context, weights\n'
'#\n'
'#\n'
'# class DepressionLSTM(nn.Module):\n'
'#     def __init__(self, input_dim=384, hidden_dim=128, num_layers=2, dropout=0.3):\n'
'#         super().__init__()\n'
'#         self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,\n'
'#                            batch_first=True, bidirectional=True, dropout=dropout)\n'
'#         self.attention = AttentionLayer(hidden_dim * 2)\n'
'#         self.reg_head = nn.Linear(hidden_dim * 2, 1)\n'
'#         self.cls_head = nn.Linear(hidden_dim * 2, 2)\n'
'#\n'
'#     def forward(self, x):\n'
'#         lstm_out, _ = self.lstm(x)\n'
'#         context, attn_weights = self.attention(lstm_out)\n'
'#         reg_out = self.reg_head(context).squeeze(-1)\n'
'#         cls_out = self.cls_head(context)\n'
'#         return reg_out, cls_out, attn_weights\n'
'\n'
'print("Appendix A: LSTM code is provided as comments.")\n'
'print("Uncomment the code above and adapt to your data to run it.")'
))

# =================================================================
# APPENDIX B — Subject Leakage Demo
# =================================================================

cells.append(md(
'## \U0001f4ce Appendix B \u2014 Subject Leakage Demonstration (Take-Home)\n'
'\n'
'This exercise demonstrates how subject leakage **artificially inflates** model performance.'
))

cells.append(code(
'# ---- APPENDIX B: Subject Leakage Demo ----\n'
'# WARNING: This code intentionally introduces a methodological flaw!\n'
'\n'
'from sklearn.model_selection import train_test_split\n'
'\n'
'# Correct approach (what we did): subject-level split\n'
'correct_rmse = np.sqrt(mean_squared_error(y_dev_reg, best_reg_pred_dev))\n'
'correct_ccc = concordance_correlation_coefficient(y_dev_reg, best_reg_pred_dev)\n'
'\n'
'# WRONG approach: random split (simulates leakage)\n'
"X_all_fused = np.vstack([feature_sets['fused'][0], feature_sets['fused'][1]])\n"
'y_all_reg = np.concatenate([y_train_reg, y_dev_reg])\n'
'\n'
'X_leak_train, X_leak_test, y_leak_train, y_leak_test = train_test_split(\n'
'    X_all_fused, y_all_reg, test_size=0.2, random_state=SEED\n'
')\n'
'\n'
'leaky_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=SEED, verbosity=0)\n'
'leaky_model.fit(X_leak_train, y_leak_train)\n'
'y_leak_pred = np.clip(leaky_model.predict(X_leak_test), 0, 24)\n'
'\n'
'leaky_rmse = np.sqrt(mean_squared_error(y_leak_test, y_leak_pred))\n'
'leaky_ccc = concordance_correlation_coefficient(y_leak_test, y_leak_pred)\n'
'\n'
'print("=" * 60)\n'
'print("\u26a0\ufe0f  SUBJECT LEAKAGE DEMONSTRATION")\n'
'print("=" * 60)\n'
"print(f\"\\n{'Metric':<10} {'Correct (Subject Split)':<25} {'WRONG (Random Split)':<25}\")\n"
'print("-" * 60)\n'
"print(f\"{'RMSE':<10} {correct_rmse:<25.3f} {leaky_rmse:<25.3f}\")\n"
"print(f\"{'CCC':<10} {correct_ccc:<25.3f} {leaky_ccc:<25.3f}\")\n"
"print(f\"\\n{'':>10} {'^ Real performance':<25} {'^ Inflated! DO NOT TRUST':<25}\")\n"
'print("\\nThe leaked model appears better because it memorized individual")\n'
'print("speaking patterns during training, not generalizable depression cues.")'
))

# =================================================================
# Assemble notebook
# =================================================================

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "toc_visible": True,
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        },
        "language_info": {
            "name": "python"
        },
        "accelerator": "GPU"
    },
    "cells": cells,
}


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "Depression_Characterization_EDAIC.ipynb")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    n_cells = len(cells)
    n_code = sum(1 for c in cells if c["cell_type"] == "code")
    n_md = sum(1 for c in cells if c["cell_type"] == "markdown")
    size_kb = os.path.getsize(output_path) / 1024

    print(f"Notebook generated: {output_path}")
    print(f"   Cells: {n_cells} total ({n_code} code, {n_md} markdown)")
    print(f"   Size:  {size_kb:.1f} KB")

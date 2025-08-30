#!/usr/bin/env python3
#in a shortcut: eeg -> cleaning signals -> epoching -> feature extraction (CSP) 
# -> classification (RandomForest)

import os
import glob #files, folders
import warnings
import numpy as np #calculations
import matplotlib.pyplot as plt #charts
import mne #eeg analysis
from mne.decoding import CSP
from sklearn.ensemble import RandomForestClassifier #machine learning
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import joblib #saving

# ----------------- CONFIG -----------------
DATA_DIR = "data" #where are files
OUT_FIG_DIR = "reports/figures"
OUT_MODEL_DIR = "models"
MODEL_FILE = os.path.join(OUT_MODEL_DIR, "csp_rf_model.joblib")

EVENT_NAMES = None
SFREQ_RESAMPLE = 128
TMIN, TMAX = 0.0, 2.0

# ----------------- HELPERS -----------------
def ensure_dirs(): #makes catalogs if not exist
    os.makedirs(OUT_FIG_DIR, exist_ok=True)
    os.makedirs(OUT_MODEL_DIR, exist_ok=True)

def find_edf_files(data_dir): #searches for .edf files
    files = glob.glob(os.path.join(data_dir, "**", "*.edf"), recursive=True)
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No .edf files found in {data_dir}")
    return files

def load_and_concatenate(edf_files): #loads and merges files
    raws = []
    print(f"Loading {len(edf_files)} EDF files...")
    for f in edf_files:
        print("  loading", f)
        raw = mne.io.read_raw_edf(f, preload=True, verbose=False)
        raws.append(raw)
    raw = mne.concatenate_raws(raws)
    return raw

def preprocess_raw(raw):
    raw.pick_types(eeg=True, meg=False, stim=False, eog=False)
    try:
        raw.set_montage("standard_1020") #mapping eceltrods
    except Exception:
        warnings.warn("Could not set standard_1020 montage")

    raw.filter(1., 40., fir_design='firwin', verbose=False) #usuwanie szumu
    raw.notch_filter(50, fir_design='firwin', verbose=False)  #usuwanie zaklocen

    ica = mne.preprocessing.ICA(n_components=15, random_state=42, verbose=False)
    ica.fit(raw) #jakie komponenty to artefakty
    ica.exclude = []

    eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name="Fp1.")
    ica.exclude = eog_indices
    ica.apply(raw)

    raw.resample(SFREQ_RESAMPLE, npad="auto") #zmniejsza czest. probkowania
    return raw

def extract_events(raw):
    events, event_id = mne.events_from_annotations(raw)
    print("Detected event_id mapping:", event_id)
    return events, event_id

def choose_event_names(event_id):
    if EVENT_NAMES:
        return EVENT_NAMES
    keys = list(event_id.keys())
    for pair in (("T1", "T2"), ("1", "2")):
        if pair[0] in keys and pair[1] in keys:
            return [pair[0], pair[1]]
    if len(keys) >= 2:
        return [keys[0], keys[1]]
    raise RuntimeError("Could not detect two event labels")

def make_epochs(raw, events, event_id, picks=None): #epoki = kwalaki sygnalu zwiazane z bodzcami
    ev_names = choose_event_names(event_id)
    epochs = mne.Epochs(raw, events, event_id=event_id,
                        tmin=TMIN, tmax=TMAX,
                        baseline=None,
                        preload=True, picks=picks, verbose=False)
    epochs_sel = mne.concatenate_epochs([epochs[name] for name in ev_names])
    
    epochs_sel.drop_bad(reject=dict(eeg=400e-6), flat=dict(eeg=1e-6))

    epochs_sel.plot_drop_log()
    plt.savefig(os.path.join(OUT_FIG_DIR, "drop_log.png"), dpi=200)
    plt.close()
    if len(epochs_sel) < 10:
        raise ValueError(f"Too few epochs remain after rejection: {len(epochs_sel)}. Cannot proceed with classification.")

    y = epochs_sel.events[:, -1]
    label_map = {event_id[name]: i for i, name in enumerate(ev_names)}
    y_mapped = np.array([label_map[eid] for eid in y])
    return epochs_sel, y_mapped, ev_names

def run_classification_csp(epochs, y, ev_names):
    X = epochs.get_data()
    csp = CSP(n_components=8, reg='ledoit_wolf', log=True, norm_trace=False)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    pipe = make_pipeline(csp, StandardScaler(), clf)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
    print("Cross-val mean accuracy:", np.mean(scores))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = pipe.score(X_test, y_test)
    print("Test accuracy:", acc)
    joblib.dump(pipe, MODEL_FILE)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ev_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion matrix (CSP + RandomForest)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "confusion_csp_rf.png"), dpi=200)
    plt.close()
    return pipe, scores

def plot_learning_curve(pipe, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    plt.figure(figsize=(6, 4))
    plt.plot(train_sizes, train_mean, 'o-', label='Train score')
    plt.plot(train_sizes, test_mean, 'o-', label='CV score')
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.legend()
    plt.title("Learning curve")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, "learning_curve.png"), dpi=200)
    plt.close()

# ----------------- MAIN -----------------
def main():
    ensure_dirs()
    files = find_edf_files(DATA_DIR)
    raw = load_and_concatenate(files)
    raw = preprocess_raw(raw)
    
    #plotes for verifications
    raw.plot(n_channels=6, title='Raw EEG snippet')
    plt.savefig(os.path.join(OUT_FIG_DIR, "raw_snippet.png"), dpi=200)
    plt.close()

    raw.plot_psd()
    plt.savefig(os.path.join(OUT_FIG_DIR, "avg_psd.jpg"), dpi=200)
    plt.close()

    raw.plot_psd(fmax=40, spatial_colors=False, average=False, show=False)
    plt.savefig(os.path.join(OUT_FIG_DIR, "spectrogram.jpg"), dpi=200)
    plt.close()
      
    events, event_id = extract_events(raw)
    epochs, y, ev_names = make_epochs(raw, events, event_id, picks='eeg')
    
    pipe, scores = run_classification_csp(epochs, y, ev_names)
    plot_learning_curve(pipe, epochs.get_data(), y)
    
    print("All done. Figures saved in:", OUT_FIG_DIR)
    print("Model saved to:", MODEL_FILE)

if __name__ == "__main__":
    main()

#czy mozna rozrocnic dwa rozne stany mozgu (po ptrzetrenowaniu...?)
#PSD & spektrogram - czy widac rytmy EEG (alfa 10hz, beta 20hza) - szum to slabe dane
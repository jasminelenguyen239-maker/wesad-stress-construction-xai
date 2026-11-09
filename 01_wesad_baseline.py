import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, f1_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

def load_wesad_subject(file_path):
    print(f"[*] Dang nap du lieu: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def extract_features(eda_w, bvp_w, temp_w, acc_w):
    # Lam phang thanh mang 1 chieu de tranh loi dinh dang 2D
    eda = eda_w.flatten()
    bvp = bvp_w.flatten()
    temp = temp_w.flatten()
    
    feats = {}
    
    # 1. EDA
    feats['eda_mean'] = float(np.mean(eda))
    feats['eda_std'] = float(np.std(eda))
    feats['eda_min'] = float(np.min(eda))
    feats['eda_max'] = float(np.max(eda))
    peaks, _ = find_peaks(eda, distance=4)
    feats['eda_peak_count'] = len(peaks)

    # 2. BVP
    feats['bvp_mean'] = float(np.mean(bvp))
    feats['bvp_std'] = float(np.std(bvp))
    bvp_peaks, _ = find_peaks(bvp, distance=32)
    feats['bvp_peak_rate'] = len(bvp_peaks) / (len(bvp) / 64.0)

    # 3. TEMP
    feats['temp_mean'] = float(np.mean(temp))
    feats['temp_std'] = float(np.std(temp))
    feats['temp_slope'] = float((temp[-1] - temp[0]) / len(temp))

    # 4. ACC
    acc_mag = np.sqrt(np.sum(acc_w**2, axis=1)).flatten()
    feats['acc_mag_mean'] = float(np.mean(acc_mag))
    feats['acc_mag_std'] = float(np.std(acc_mag))
    
    return feats

def process_wesad_wrist(data, window_sec=60, step_sec=10):
    wrist = data['signal']['wrist']
    labels = data['label'].flatten()
    
    # Dong bo nhan 700 Hz ve 4 Hz (tan so EDA)
    sub_labels = labels[::175]
    min_len = min(len(sub_labels), len(wrist['EDA']))
    sub_labels = sub_labels[:min_len]

    win_len = window_sec * 4
    step_len = step_sec * 4

    X_list, y_list = [], []
    for start in range(0, min_len - win_len, step_len):
        end = start + win_len
        win_labels = sub_labels[start:end]
        majority = pd.Series(win_labels).mode()[0]
        
        # 1: Baseline -> 0, 2: Stress -> 1
        if majority not in [1, 2]:
            continue

        s_sec, e_sec = start / 4.0, end / 4.0
        eda_w = wrist['EDA'][int(s_sec*4):int(e_sec*4)]
        bvp_w = wrist['BVP'][int(s_sec*64):int(e_sec*64)]
        temp_w = wrist['TEMP'][int(s_sec*4):int(e_sec*4)]
        acc_w = wrist['ACC'][int(s_sec*32):int(e_sec*32)]

        X_list.append(extract_features(eda_w, bvp_w, temp_w, acc_w))
        y_list.append(1 if majority == 2 else 0)

    return pd.DataFrame(X_list), np.array(y_list)

if __name__ == "__main__":
    # Duong dan tuong doi toi file S2
    DATA_PATH = os.path.join("WESAD", "S2", "S2.pkl")
    
    # 1. Trich xuat dac trung
    data = load_wesad_subject(DATA_PATH)
    X, y = process_wesad_wrist(data)
    print(f"[+] Tong so cua so trich xuat: {len(X)}")
    print(f"[+] Phan bo lop: Baseline (0): {sum(y==0)} | Stress (1): {sum(y==1)}")

    # 2. Chia tap train/test va huan luyen XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42, eval_metric='logloss')
    clf.fit(X_train, y_train)

    # 3. Danh gia ket qua
    y_pred = clf.predict(X_test)
    print("\n--- KET QUA PHAN LOAI STRESS ---")
    print(classification_report(y_test, y_pred, target_names=["Baseline", "Stress"]))
    print(f"Macro F1-Score: {f1_score(y_test, y_pred, average='macro'):.4f}")

    # 4. Giai thich mo hinh bang SHAP
    print("\n[*] Dang tinh toan gia tri SHAP...")
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_test)

    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("WESAD - SHAP Feature Importance (Stress Classification)", fontsize=12)
    plt.tight_layout()
    plt.savefig("wesad_shap_summary.png", dpi=300)
    print("[+] Da tao xong bieu do SHAP tai file: wesad_shap_summary.png")
    plt.show()

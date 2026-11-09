import os
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

def load_wesad_subject(file_path):
    print(f"[*] Dang nap du lieu: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def extract_advanced_features(eda_w, bvp_w, temp_w, acc_w):
    eda = eda_w.flatten()
    bvp = bvp_w.flatten()
    temp = temp_w.flatten()
    
    feats = {}
    
    # 1. ACC: Do lon chuyen dong va Do giat (Khu nhieu van dong)
    acc_mag = np.sqrt(np.sum(acc_w**2, axis=1)).flatten()
    feats['acc_mag_mean'] = float(np.mean(acc_mag))
    feats['acc_mag_std'] = float(np.std(acc_mag))
    feats['acc_jerk'] = float(np.std(np.diff(acc_mag))) if len(acc_mag) > 1 else 0.0

    # 2. EDA: Tach thanh phan Phasic bang High-pass filter
    # Giup lay phan ung kich thich dot ngot, giam anh huong do nhiet do
    b, a = butter(2, 0.05 / (4 / 2), btype='high')
    eda_phasic = filtfilt(b, a, eda)
    feats['eda_phasic_mean'] = float(np.mean(np.abs(eda_phasic)))
    feats['eda_phasic_max'] = float(np.max(eda_phasic))
    feats['eda_phasic_peaks'] = float(len(find_peaks(eda_phasic, height=0.01, distance=4)[0]))

    # 3. BVP & HRV: RMSSD (Chi so bien thien nhip tim)
    bvp_peaks, _ = find_peaks(bvp, distance=32)
    feats['bvp_peak_rate'] = float(len(bvp_peaks) / (len(bvp) / 64.0))
    if len(bvp_peaks) > 2:
        rr_intervals = np.diff(bvp_peaks) / 64.0 * 1000.0 # ms
        feats['hrv_rmssd'] = float(np.sqrt(np.mean(np.square(np.diff(rr_intervals)))))
    else:
        feats['hrv_rmssd'] = 0.0

    # 4. TEMP
    feats['temp_mean'] = float(np.mean(temp))
    feats['temp_slope'] = float((temp[-1] - temp[0]) / len(temp))

    return feats

def process_wesad_wrist(data, window_sec=60, step_sec=10):
    wrist = data['signal']['wrist']
    labels = data['label'].flatten()
    sub_labels = labels[::175][:min(len(labels[::175]), len(wrist['EDA']))]

    win_len = window_sec * 4
    step_len = step_sec * 4
    X_list, y_list = [], []

    for start in range(0, len(sub_labels) - win_len, step_len):
        end = start + win_len
        win_labels = sub_labels[start:end]
        majority = pd.Series(win_labels).mode()[0]
        if majority not in [1, 2]:
            continue

        s_sec, e_sec = start / 4.0, end / 4.0
        eda_w = wrist['EDA'][int(s_sec*4):int(e_sec*4)]
        bvp_w = wrist['BVP'][int(s_sec*64):int(e_sec*64)]
        temp_w = wrist['TEMP'][int(s_sec*4):int(e_sec*4)]
        acc_w = wrist['ACC'][int(s_sec*32):int(e_sec*32)]

        X_list.append(extract_advanced_features(eda_w, bvp_w, temp_w, acc_w))
        y_list.append(1 if majority == 2 else 0)

    return pd.DataFrame(X_list), np.array(y_list)

if __name__ == "__main__":
    DATA_PATH = os.path.join("WESAD", "S2", "S2.pkl")
    data = load_wesad_subject(DATA_PATH)
    X, y = process_wesad_wrist(data)
    
    # -------------------------------------------------------------
    # LOẠI BỎ BIẾN NHIỆT ĐỘ (KHÁNG NHIỄU MÔI TRƯỜNG CÔNG TRƯỜNG)
    # Ép mô hình dựa trên phản ứng thần kinh giao cảm (EDA, HRV) và vận động (ACC)
    # -------------------------------------------------------------
    construction_features = [col for col in X.columns if not col.startswith('temp_')]
    X_construction = X[construction_features]
    print(f"[*] Cac dac trung dua vao huan luyen: {construction_features}")

    # Chia train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X_construction, y, test_size=0.25, random_state=42, stratify=y
    )
    
    clf = xgb.XGBClassifier(
        n_estimators=100, 
        max_depth=4, 
        learning_rate=0.05, 
        random_state=42, 
        eval_metric='logloss'
    )
    clf.fit(X_train, y_train)

    print("\n--- KET QUA PHAN LOAI (CONSTRUCTION SETTING) ---")
    print(classification_report(y_test, clf.predict(X_test), target_names=["Baseline", "Stress"]))

    # Tinh toan SHAP
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer(X_test)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("Construction-Adapted Features (No Temp Bias) - SHAP Summary", fontsize=12)
    plt.tight_layout()
    # Luu thanh file anh moi, khong bi de len file cu
    plt.savefig("wesad_shap_construction_adapted.png", dpi=300)
    print("[+] Da xuat thanh cong anh moi: wesad_shap_construction_adapted.png")
    plt.show()

import pandas as pd
import numpy as np
import os
from src.scaler import MinMaxScaler

def run_preprocessing(config):
    # 1. Kaynak Dosyaların Varlığını Kontrol Et
    train_path = config["paths"]["train_path"]
    test_path = config["paths"]["test_path"]

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"HATA: 'Data' klasörü içinde trainDATA.csv veya testDATA.csv bulunamadı!")
        return

    # 2. Verileri Yükle
    X_train = pd.read_csv(train_path)
    X_test = pd.read_csv(test_path)

    # 3. Temizlik ve Özellik Mühendisliği (Aynı Mantık)
    X_train = X_train.drop('name', axis=1)
    X_test = X_test.drop('name', axis=1)

    y_train = X_train['selling_price']
    X_train = X_train.drop('selling_price', axis=1)
    y_test = X_test['selling_price']
    X_test = X_test.drop('selling_price', axis=1)

    current_year = 2026
    X_train['car_age'] = current_year - X_train['year']
    X_test['car_age'] = current_year - X_test['year']
    X_train = X_train.drop('year', axis=1)
    X_test = X_test.drop('year', axis=1)

    # Yeni Özellikleri Ekle (Hem Train hem Test için!)
    for df in [X_train, X_test]:
        # Etkileşim Terimi: Yaş ve KM arasındaki ilişki
        df["interaction"] = df["car_age"] * df["km_driven"]
        
        # Polinomsal Terim: Aracın yaşının karesi (Fiyat düşüşü doğrusal değildir)
        df["car_age_squared"] = df["car_age"] ** 2

    # NaN Önlemi: Eğer çarpım sonucu sonsuz değer oluşursa bunları temizle
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # 4. Kategorik Değişkenler ve One-Hot Encoding
    owner_mapping = {
    'First Owner': 5,
    'Second Owner': 4,
    'Third Owner': 3,
    'Fourth & Above Owner': 2,
    'Test Drive Car': 1
    }
    X_train['owner'] = X_train['owner'].map(owner_mapping)
    X_test['owner'] = X_test['owner'].map(owner_mapping)

    X_train['is_train'] = 1
    X_test['is_train'] = 0
    combined_df = pd.concat([X_train, X_test], axis=0)

    categorical_cols = ['fuel', 'seller_type', 'transmission']
    combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True).astype(float)

    X_train = combined_df[combined_df['is_train'] == 1].drop('is_train', axis=1)
    X_test = combined_df[combined_df['is_train'] == 0].drop('is_train', axis=1)

    y_train = np.log1p(y_train)
    y_test = np.log1p(y_test)

    # 5. Ölçeklendirme (Scaling)
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()

    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)

    y_train = y_scaler.fit_transform(y_train)
    y_test = y_scaler.transform(y_test) 

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    return X_train, y_train, X_test, y_test, y_scaler
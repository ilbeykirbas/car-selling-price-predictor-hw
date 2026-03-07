import pandas as pd
import numpy as np
import os

def run_preprocessing():
    # 1. Klasör Yollarını Ayarla
    # preprocessing.py'den bir üst dizine (root) çıkıp Data klasörüne odaklanıyoruz
    current_file_path = os.path.abspath(__file__)
    root_dir = os.path.dirname(os.path.dirname(current_file_path))
    data_dir = os.path.join(root_dir, 'Data')
    output_dir = os.path.join(data_dir, 'PreprocessedData')

    # 2. Kaynak Dosyaların Varlığını Kontrol Et
    train_path = os.path.join(data_dir, 'trainDATA.csv')
    test_path = os.path.join(data_dir, 'testDATA.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"HATA: 'Data' klasörü içinde trainDATA.csv veya testDATA.csv bulunamadı!")
        print(f"Lütfen dosyaların şu konumda olduğundan emin olun: {data_dir}")
        return

    # 3. Çıktı Klasörünü Otomatik Oluştur (Eğer yoksa)
    # Sadece PreprocessedData klasörünü oluşturuyoruz, Data klasörü zaten olmalı.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Yeni klasör oluşturuldu: {output_dir}")

    # 4. Verileri Yükle
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # 5. Temizlik ve Özellik Mühendisliği (Aynı Mantık)
    train_df = train_df.drop('name', axis=1)
    test_df = test_df.drop('name', axis=1)

    y_train = train_df['selling_price']
    train_df = train_df.drop('selling_price', axis=1)
    y_test = test_df['selling_price']
    test_df = test_df.drop('selling_price', axis=1)

    current_year = 2026
    train_df['car_age'] = current_year - train_df['year']
    test_df['car_age'] = current_year - test_df['year']
    train_df = train_df.drop('year', axis=1)
    test_df = test_df.drop('year', axis=1)

    # 6. Kategorik Değişkenler ve One-Hot Encoding
    train_df['is_train'] = 1
    test_df['is_train'] = 0
    combined_df = pd.concat([train_df, test_df], axis=0)

    categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
    combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True).astype(float)

    train_df = combined_df[combined_df['is_train'] == 1].drop('is_train', axis=1)
    test_df = combined_df[combined_df['is_train'] == 0].drop('is_train', axis=1)

    # 7. Ölçeklendirme (Scaling)
    cols_to_scale = train_df.columns
    train_mean = train_df[cols_to_scale].mean()
    train_std = train_df[cols_to_scale].std()
    train_std[train_std == 0] = 1 

    train_df[cols_to_scale] = (train_df[cols_to_scale] - train_mean) / train_std
    test_df[cols_to_scale] = (test_df[cols_to_scale] - train_mean) / train_std

    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_scaled = (y_train - y_mean) / y_std
    y_test_scaled = (y_test - y_mean) / y_std

    # 8. Kaydetme
    train_df.to_csv(os.path.join(output_dir, 'X_train_processed.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'X_test_processed.csv'), index=False)
    y_train_scaled.to_csv(os.path.join(output_dir, 'y_train_processed.csv'), index=False)
    y_test_scaled.to_csv(os.path.join(output_dir, 'y_test_processed.csv'), index=False)

    print("İşlem başarıyla tamamlandı. İşlenmiş dosyalar Data/PreprocessedData içine kaydedildi.")

if __name__ == "__main__":
    run_preprocessing()
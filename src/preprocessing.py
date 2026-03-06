import pandas as pd
import numpy as np

# 1. Verileri yükle (Dosya isimlerini düzelttim)
train_df = pd.read_csv('trainDATA.csv')
test_df = pd.read_csv('testDATA.csv')

# 2. İsim sütununu düşür (Sadece bir kez yapıyoruz)
train_df = train_df.drop('name', axis=1)
test_df = test_df.drop('name', axis=1)

# 3. Hedef değişkeni (y) ve özellikleri (X) ayır
y_train = train_df['selling_price']
train_df = train_df.drop('selling_price', axis=1)

y_test = test_df['selling_price']
test_df = test_df.drop('selling_price', axis=1)

# --- BURADAKİ TEKRAR EDEN BLOK SİLİNDİ ---

# 4. Özellik Mühendisliği
current_year = 2026
train_df['car_age'] = current_year - train_df['year']
test_df['car_age'] = current_year - test_df['year']
train_df = train_df.drop('year', axis=1)
test_df = test_df.drop('year', axis=1)

# 5. Kategorik Değişkenler ve Birleştirme
train_df['is_train'] = 1
test_df['is_train'] = 0
combined_df = pd.concat([train_df, test_df], axis=0)

# One-Hot Encoding
categorical_cols = ['fuel', 'seller_type', 'transmission', 'owner']
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

# Verileri tekrar ayır
train_df = combined_df[combined_df['is_train'] == 1].drop('is_train', axis=1)
test_df = combined_df[combined_df['is_train'] == 0].drop('is_train', axis=1)

# 6. Ölçeklendirme (Scaling)
cols_to_scale = train_df.columns
train_mean = train_df[cols_to_scale].mean()
train_std = train_df[cols_to_scale].std()

# Formülü uygula
train_df[cols_to_scale] = (train_df[cols_to_scale] - train_mean) / train_std
test_df[cols_to_scale] = (test_df[cols_to_scale] - train_mean) / train_std

# Hedef değişken ölçeklendirme
y_mean = y_train.mean()
y_std = y_train.std()
y_train_scaled = (y_train - y_mean) / y_std
y_test_scaled = (y_test - y_mean) / y_std

# 7. Kaydetme
train_df.to_csv('X_train_processed.csv', index=False)
test_df.to_csv('X_test_processed.csv', index=False)
y_train_scaled.to_csv('y_train_processed.csv', index=False)
y_test_scaled.to_csv('y_test_processed.csv', index=False)

print("İşlem başarıyla tamamlandı, dosyalar kaydedildi.")
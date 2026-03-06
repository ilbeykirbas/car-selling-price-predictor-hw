import pandas as pd
import numpy as np

# Verileri yükle
train_df = pd.read_csv('trainDATA.csv')
test_df = pd.read_csv('testDATA.csv')

# İsim sütununu düşür
train_df = train_df.drop('name', axis=1)
test_df = test_df.drop('name', axis=1)

# Hedef değişkeni (y) ve özellikleri (X) ayır (Şimdilik sadece train seti için)
y_train = train_df['selling_price']
train_df = train_df.drop('selling_price', axis=1)

y_test = test_df['selling_price']
test_df = test_df.drop('selling_price', axis=1)

#update test 2
def train_model(model, X_train, y_train_scaled):
    X_train_numpy=X_train.to_numpy()
    y_train_numpy=y_train_scaled.to_numpy()
       
    model.fit(X_train_numpy,y_train_numpy)

    return model
def train_model(model, X_train, y_train_scaled):
    model = model.fit(X_train, y_train_scaled)

    return model
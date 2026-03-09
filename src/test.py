from src.metrics import summary

def test_model(model, X_test, y_test, scaler):
    preds = model.predict(X_test)

    preds = scaler.inverse_transform(preds)
    y_test = scaler.inverse_transform(y_test)

    errors = summary(y_test, preds)

    print(errors)
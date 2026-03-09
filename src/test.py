from src.metrics import summary

def test_model(model, X_test, y_test_scaled):
    preds = model.predict(X_test)

    errors = summary(y_test_scaled, preds)

    print(errors)
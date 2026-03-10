import matplotlib.pyplot as plt
import numpy as np

def train_model(model, X_train, y_train):
    model = model.fit(X_train,y_train)

    # J vs iterasyon
    plt.plot(model.loss_history)

    # θ vs iterasyon
    weights_array = np.array(model.weight_history)  # shape: (n_iters, n_features+1)
    plt.plot(weights_array[:, 0])  # θ₀ (bias)
    plt.plot(weights_array[:, 1])  # θ₁ (ilk feature)

    return model
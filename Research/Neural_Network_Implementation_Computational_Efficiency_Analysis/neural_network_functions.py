
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def hidden_layer_forward_loop(x, W1, b1):
    """
    x : np.ndarray, shape (D,)
        One input observation.

    W1 : np.ndarray, shape (H, D)
        Input-to-hidden weights.

    b1 : np.ndarray, shape (H,)
        Hidden-layer biases.

    Returns
    z1 : np.ndarray, shape (H,)
        Hidden-layer pre-activations.

    a1 : np.ndarray, shape (H,)
        Hidden-layer activations.
    """
    hidden_units, input_features = W1.shape

    if x.shape != (input_features,):
        raise ValueError(
            f"x must have shape ({input_features},), "
            f"but received {x.shape}."
        )

    if b1.shape != (hidden_units,):
        raise ValueError(
            f"b1 must have shape ({hidden_units},), "
            f"but received {b1.shape}."
        )

    z1 = np.zeros(hidden_units, dtype=float)
    a1 = np.zeros(hidden_units, dtype=float)

    for i in range(hidden_units):
        z1[i] = float(b1[i])

        for j in range(input_features):
            z1[i] += float(W1[i, j]) * float(x[j])

        a1[i] = float(sigmoid(z1[i]))

    return z1, a1


def output_layer_forward_loop(a1, W2, b2):
    """
    a1 : np.ndarray, shape (H,)
        Hidden-layer activations.

    W2 : np.ndarray, shape (H,)
        Hidden-to-output weights.

    b2 : float
        Output-layer bias.

    Returns
    z2 : float
        Output pre-activation.

    y_hat : float
        Predicted probability.
    """
    if a1.shape != W2.shape:
        raise ValueError(
            f"a1 and W2 must have matching shapes, "
            f"but received {a1.shape} and {W2.shape}."
        )

    z2 = float(b2)

    for i in range(len(a1)):
        z2 += float(W2[i]) * float(a1[i])

    y_hat = float(sigmoid(z2))

    return z2, y_hat


def forward_propagation_loop(x, W1, b1, W2, b2):
    """
    Returns
    cache : dict
        Intermediate values required for backpropagation.
    """
    z1, a1 = hidden_layer_forward_loop(x, W1, b1)
    z2, y_hat = output_layer_forward_loop(a1, W2, b2)

    cache = {
        "x": x,
        "z1": z1,
        "a1": a1,
        "z2": z2,
        "y_hat": y_hat
    }

    return cache

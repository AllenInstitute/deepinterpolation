from tensorflow.keras import backend as K


def dummy_function(x):
    return x * x


def loss_selector(loss_type):
    if loss_type == "mean_squareroot_error":
        return mean_squareroot_error
    if loss_type == "annealed_loss":
        return annealed_loss
    else:
        return loss_type


def annealed_loss(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    local_power = 4
    final_loss = K.pow(K.abs(y_pred - y_true) + 0.00000001, local_power)
    return K.mean(final_loss, axis=-1)


def mean_squareroot_error(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    return K.mean(K.sqrt(K.abs(y_pred - y_true) + 0.00000001), axis=-1)

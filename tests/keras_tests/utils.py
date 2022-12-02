import matplotlib.pyplot as plt


def keras_plot_losses(history):
    print(history.history.keys())
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.show()


__all__ = [
    'keras_plot_losses',
]

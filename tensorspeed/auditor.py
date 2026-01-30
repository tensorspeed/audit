# tensorspeed/auditor.py
import matplotlib.pyplot as plt
import tensorflow as tf

class Audit:
    def __init__(self, model):
        self.model = model

    def plot_learning_curves(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Accuracy Audit')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Loss Audit (Check for Divergence)')
        plt.legend()
        plt.show()
        plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
        print("Learning curves saved as 'learning_curves.png'")

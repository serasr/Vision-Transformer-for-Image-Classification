import matplotlib.pyplot as plt

def plot_accuracy(results):
    plt.figure(figsize=(8,5))
    plt.plot(results['train_acc'], label='Train Accuracy')
    plt.plot(results['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.title('Training vs Validation Accuracy')
    plt.savefig("results/final_metrics.png")
    plt.show()

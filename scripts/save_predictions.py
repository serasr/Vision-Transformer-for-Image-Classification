import os
import matplotlib.pyplot as plt
import numpy as np
import torch

def save_predictions(model, dataloader, device, class_names, save_dir="results/", num_images=10):
    model.eval()
    model.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    images_so_far = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size(0)):
                if images_so_far >= num_images:
                    print(f"Saved {images_so_far} images to '{save_dir}'")
                    return
                images_so_far += 1

                img = inputs[j].cpu().permute(1, 2, 0).numpy()
                img = img * 0.5 + 0.5
                img = np.clip(img, 0, 1)

                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.axis('off')
                plt.title(f"Pred: {class_names[preds[j]]} | True: {class_names[labels[j]]}")

                save_path = os.path.join(save_dir, f"prediction_{images_so_far}.png")
                plt.savefig(save_path)
                plt.close()



save_predictions(
    model=vit_model,
    dataloader=test_dataloader,
    device=device,
    class_names=["ants", "bees"],
    save_dir="results/",
    num_images=10
)


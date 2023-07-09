
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from lenet import LeNet5Simulator
from tqdm import tqdm
from collections import defaultdict

import json
import os
import sys 



CLASSES_MODULE_PATH = "../../../"
WEIGHT_FILE_PATH = "."
MODELS_FOLDER = CLASSES_MODULE_PATH + 'models'

# appending a path
sys.path.append(CLASSES_MODULE_PATH) #CHANGE THIS LINE
from examples.pytorch.lenet.lenet import LeNet5

N_CLASSES = 10
N_INJECTABLE_LAYERS = 8
N_IMAGES = 10000
N_SIM_PER_IMAGE = 40


def main():
    model = LeNet5(N_CLASSES)
    model.load_state_dict(torch.load(os.path.join(WEIGHT_FILE_PATH,'lenet.pth')))
    model.eval()

    transf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    test_dataset = datasets.MNIST('data', train=False, transform=transf, download=True)

    test_images = np.random.choice(len(test_dataset), size=N_IMAGES, replace=False).astype(int)
    images_results = {}
    layer_results = {}

    prog_bar = tqdm(total=N_INJECTABLE_LAYERS * N_IMAGES * N_SIM_PER_IMAGE)

    for image_idx in test_images:
        img, label = test_dataset[image_idx]
        img = img.unsqueeze(0)
        output_vanilla = model(img)[0]
        pred_vanilla = output_vanilla.argmax(dim=1).item()
        images_results[image_idx] = {
            "label": label,
            "golden_prediction": pred_vanilla,
        }

    for injectable_layer_id in range(N_INJECTABLE_LAYERS):
        error_simuator_net = LeNet5Simulator(N_CLASSES, selected_layer=injectable_layer_id)
        with torch.no_grad():
            for name, _ in model.named_parameters():
                layer_name = name.split('.')[0]
                eval(f'error_simuator_net.{layer_name}.weight.copy_(model.{layer_name}.weight)')
                eval(f'error_simuator_net.{layer_name}.bias.copy_(model.{layer_name}.bias)')
        error_simuator_net.eval()

        images = {}
        for image_idx in test_images:
            label = images_results[image_idx]["label"]
            pred_vanilla = images_results[image_idx]["golden_prediction"] 
            images[int(image_idx)] = {
                "label": label,
                "golden_prediction": pred_vanilla,
                "results": defaultdict(int),
            }
            img, _ = test_dataset[image_idx]
            img = img.unsqueeze(0)
            for run in range(N_SIM_PER_IMAGE):
                output_corr = error_simuator_net(img)[0]
                # print(f'The shape is {img.shape}')
                pred = int(output_corr.argmax(dim=1).item())
                # print(f" Pred vs Label => ({pred},{label})")
                images[image_idx]["results"][pred] += 1

                prog_bar.update(1)
        layer_results[injectable_layer_id] = {
            "images": images
        }


    with open("lvf_campaign.json", "w") as f:
        json.dump(layer_results, f, indent=2)

    accuracy = sum(1 for v in images_results.values() if v["label"] == v["golden_prediction"]) / N_IMAGES
    print(f'Base accuracy: {accuracy}')

    fault_tol_preds = defaultdict(int)
    accurate_preds = defaultdict(int)
    for layer, layer_data in layer_results.items():
        for image_id, image_data in layer_data["images"].items():
            fault_tol_preds[layer] += image_data["results"][image_data["golden_prediction"]]
            accurate_preds[layer] += image_data["results"][image_data["label"]]

    print(fault_tol_preds)
    print(accurate_preds)

if __name__ == "__main__":
    main()
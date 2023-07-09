
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
N_SIM_PER_IMAGE = 5
N_DOMAIN_CLASSES = 8

DOM_CLASS_STEP = 100.0 / N_DOMAIN_CLASSES

def main():
    model = LeNet5(N_CLASSES)
    model.load_state_dict(torch.load(os.path.join(WEIGHT_FILE_PATH,'lenet.pth')))
    model.eval()

    transf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    test_dataset = datasets.MNIST('data', train=False, transform=transf, download=True)

    test_images = np.random.choice(len(test_dataset), size=N_IMAGES, replace=False).astype(int)
    images_results = {}
    exp_results = defaultdict(dict)

    prog_bar = tqdm(total=N_INJECTABLE_LAYERS * N_IMAGES * N_SIM_PER_IMAGE * N_DOMAIN_CLASSES)

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
        for dom_class_id in range(N_DOMAIN_CLASSES):

            pct_1 = dom_class_id * DOM_CLASS_STEP
            pct_2 = 100.0 - pct_1

            domain_class = {
                "in_range": [pct_1, pct_1 + DOM_CLASS_STEP],
                "out_of_range": [pct_2 - DOM_CLASS_STEP, pct_2]
            }
            error_simuator_net = LeNet5Simulator(N_CLASSES, selected_layer=injectable_layer_id, fixed_domain_class=domain_class)
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
            exp_results[dom_class_id][injectable_layer_id] = {
                "images": images
            }


    with open("domains_campaign.json", "w") as f:
        json.dump(exp_results, f, indent=2)

    accuracy = sum(1 for v in images_results.values() if v["label"] == v["golden_prediction"]) / N_IMAGES
    print(f'Base accuracy: {accuracy}')

    fault_tol_preds = defaultdict(int)
    accurate_preds = defaultdict(int)
    for dom_class, dom_class_data in exp_results.items():
        for layer, layer_data in dom_class_data.items():
            for image_id, image_data in layer_data["images"].items():
                fault_tol_preds[dom_class] += image_data["results"][image_data["golden_prediction"]]
                accurate_preds[dom_class] += image_data["results"][image_data["label"]]

    print(fault_tol_preds)
    print(accurate_preds)

if __name__ == "__main__":
    main()
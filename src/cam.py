import numpy as np
import cv2
import torch
import glob as glob

from torchvision import transforms
from torch.nn import functional as F
from torch import topk

from model import build_model

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

# Define computation device.
device = 'cpu'
# Class names.
class_names = ['fake', 'real']
# Initialize model, switch to eval model, load trained weights.
model = build_model(
    pretrained=False,
    fine_tune=False, 
    num_classes=2
).to(device)
model = model.eval()
model.load_state_dict(torch.load('../outputs/model.pth')['model_state_dict'])

# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
def returnCAM(feature_conv, weight_softmax, class_idx):
    # Generate the class activation maps upsample to 256x256.
    size_upsample = (256, 256)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight_softmax[idx].dot(feature_conv.reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

def show_cam(CAMs, width, height, orig_image, class_idx, save_name):
    for i, cam in enumerate(CAMs):
        heatmap = cv2.applyColorMap(cv2.resize(cam,(width, height)), cv2.COLORMAP_JET)
        result = heatmap * 0.5 + orig_image * 0.5
        # put class label text on the result
        cv2.putText(result, str(class_names[int(class_idx[i])]), (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow('CAM', result/255.)
        cv2.waitKey(0)
        cv2.imwrite(f"../outputs/CAM_{save_name}.jpg", result)

# Hook the feature extractor.
# https://github.com/zhoubolei/CAM/blob/master/pytorch_CAM.py
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())
model._modules.get('features').register_forward_hook(hook_feature)
# Get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-4].data.cpu().numpy())

# Define the transforms, resize => tensor => normalize.
transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize((256, 256)),
     transforms.ToTensor(),
     transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
        )
    ])

counter = 0
# Run for all the test images.
all_images = glob.glob('../input/test_images/*')
print(all_images[:5])
for image_path in all_images:
    # Read the image.
    image = cv2.imread(image_path)
    orig_image = image.copy()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = orig_image.shape
    # Apply the image transforms.
    image_tensor = transform(image)
    # Add batch dimension.
    image_tensor = image_tensor.unsqueeze(0)
    # Forward pass through model.
    outputs = model(image_tensor.to(device))
    # Get the softmax probabilities.
    probs = F.softmax(outputs).data.squeeze()
    # Get the class indices of top k probabilities.
    class_idx = topk(probs, 1)[1].int()
    
    # Generate class activation mapping for the top1 prediction.
    CAMs = returnCAM(features_blobs[0], weight_softmax, class_idx)
    # File name to save the resulting CAM image with.
    save_name = f"{image_path.split('/')[-1].split('.')[0]}"
    # Show and save the results.
    show_cam(CAMs, width, height, orig_image, class_idx, save_name)
    counter += 1
    print(f"Image: {counter}")
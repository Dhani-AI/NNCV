import numpy as np
from torchvision import transforms
import torch

# Constants from training
MEAN = [0.28689554, 0.32513303, 0.28389177]
STD = [0.18696375, 0.19017339, 0.18720214]

def preprocess(img):
    """preproces image:
    input is a PIL image.
    Output image should be pytorch tensor that is compatible with your model"""

    img = transforms.functional.resize(img, size=(640, 640), interpolation=transforms.InterpolationMode.BILINEAR)
    img = transforms.functional.pad(img, padding=[2, 2, 2, 2], fill=0, padding_mode='constant')

    trans = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
        ])
    
    img = trans(img)
    img = img.unsqueeze(0)

    return img

def postprocess(prediction, shape):
    """Post process prediction to mask:
    Input is the prediction tensor provided by your model, the original image size.
    Output should be numpy array with size [x,y,n], where x,y are the original size of the image and n is the class label per pixel.
    We expect n to return the training id as class labels. training id 255 will be ignored during evaluation."""
    
    # First upsample to 640x640 (matching training pipeline)
    upsampled_logits = torch.nn.functional.interpolate(
        prediction, 
        size=(640, 640),  # Match training size
        mode="bilinear",
        align_corners=False
    )

    m = torch.nn.Softmax(dim=1)
    prediction_soft = m(upsampled_logits)
    prediction_max = torch.argmax(prediction_soft, axis=1)

    # Remove padding (2 pixels each side)
    prediction_max = prediction_max[:, 2:-2, 2:-2]
    
    prediction = transforms.functional.resize(
        prediction_max, 
        size=shape, 
        interpolation=transforms.InterpolationMode.NEAREST)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()

    return prediction_numpy
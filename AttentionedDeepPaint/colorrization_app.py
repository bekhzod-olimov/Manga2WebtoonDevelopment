import json
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
import colorgram.colorgram as cgm
from preprocess import re_scale, save_image, make_colorgram_tensor, scale
from models import DeepUNetPaintGenerator
from utils import load_checkpoints
import sys
import torch
from PIL import Image
from torchvision import transforms
from glob import glob 
import random
import os

app = Flask(__name__)
CORS(app, resources={r"*": {"origins": "*"}})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = 'deepunetG_150.pth.tar'

model = DeepUNetPaintGenerator()
model = model.to(device)
load_checkpoints(generator, model, device_type=device.type)

for param in model.parameters():
    param.requires_grad = False

    
def get_rgb(colorgram_result):
    """
    from colorgram_result, result rgb value as tuple of (r,g,b)
    """
    color = colorgram_result.rgb
    return (color.r, color.g, color.b)


def crop_region(image):
    """
    from image, crop 4 region and return
    """
    width, height = image.size
    h1 = height // 4
    h2 = h1 + h1
    h3 = h2 + h1
    h4 = h3 + h1
    image1 = image.crop((0, 0, width, h1))
    image2 = image.crop((0, h1, width, h2))
    image3 = image.crop((0, h2, width, h3))
    image4 = image.crop((0, h3, width, h4))

    return (image1, image2, image3, image4)


def get_topk(color_info, k):
    colors = list(color_info.values())
    return list(map(lambda x: x[k], colors))

@app.route('/colordemo', methods=['POST'])
def anysegdemo():
    json_data = json.loads(request.data)
    if not os.path.exists(json_data['output_path']):
        os.mkdir(json_data['output_path'])
    topk = 4
    print(f"sytle:{json_data['style_path']}")
    print(f"image:{json_data['image_path']}")
    style = Image.open(json_data['style_path']).convert('RGB')
    style = transforms.Resize((512, 512))(style)
    style_pil = style

    image = Image.open(json_data['image_path']).convert('RGB')
    image_pil = transforms.Resize((512, 512))(image)
    transform = transforms.Compose(
        [transforms.Resize((512, 512)),
         transforms.ToTensor()])

    image = transform(image)
    image = scale(image)
    image = image.unsqueeze(0).to(device)

    to_pil = transforms.ToPILImage()

    images = list(crop_region(style))
    result = {}
    for i, img in enumerate(images, 1):
        colors = cgm.extract(img, topk + 1)
        result[str(i)] = {
            '%d' % i: get_rgb(colors[i])
            for i in range(1, topk + 1)
        }

    color_tensor = make_colorgram_tensor(result)
    color_tensor = color_tensor.unsqueeze(0).to(device)

    fakeB, _ = model(image, color_tensor)
    fakeB = fakeB.squeeze(0)
    fakeB = re_scale(fakeB.detach().cpu())
    fakeB = to_pil(fakeB)
    
    result_image = Image.new('RGB', (512 * 3, 512))
    result_image.paste(image_pil, (512 * 0, 0, 512 * 1, 512))
    result_image.paste(style_pil, (512 * 1, 0, 512 * 2, 512))
    result_image.paste(fakeB, (512 * 2, 0, 512 * 3, 512))
    image_pil.save(os.path.join(json_data['output_path'],'image_img.png'))
    style_pil.save(os.path.join(json_data['output_path'],'style_img.png'))
    fakeB.save(os.path.join(json_data['output_path'],'fake_img.png'))
    result_image.save(os.path.join(json_data['output_path'],'Merge_img.png'))
    result_dict ={}
    result_dict['image_path'] =os.path.join(json_data['output_path'],'image_img.png')
    result_dict['style_path'] =os.path.join(json_data['output_path'],'style_img.png')
    result_dict['fake_path'] =os.path.join(json_data['output_path'],'fake_img.png')
    result_dict['merge_path'] =os.path.join(json_data['output_path'],'Merge_img.png')
    return jsonify(result_dict)
    
if __name__ == "__main__":
    app.run(host = '0.0.0.0', debug = False, port = 8421, threaded=True)

   

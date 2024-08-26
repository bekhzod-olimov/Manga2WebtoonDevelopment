"""
For all datasets, extract top-4 color histogram and save it into json files
"""

import json
import os
import glob

from PIL import Image
from colorgram import colorgram as cgm

data_path = '../data'
out_path = '../data/colorgram'

img_files = glob.glob(os.path.join(data_path, 'train/*.png'))
img_files += glob.glob(os.path.join(data_path, 'val/*.png'))

# img_files = glob.glob(os.path.join(data_path, 'test/*.png'))

topk = 4


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

def w_crop_region(image):
    """
    from image, crop 4 region and return
    """
    width, height = image.size
    w1 = width // 4
    w2 = w1 + w1
    w3 = w2 + w1
    w4 = w3 + w1
    image1 = image.crop((0, 0, w1, height))
    image2 = image.crop((w1, 0, w2, height))
    image3 = image.crop((w2, 0, w3, height))
    image4 = image.crop((w3, 0, w4, height))

    return (image1, image2, image3, image4)


def get_topk(color_info, k):
    colors = list(color_info.values())
    return list(map(lambda x: x[k], colors))


for filename in img_files:
    image = Image.open(filename)
    width, height = image.size
    image = image.crop((0, 0, width // 2, height))

    image_id = filename.split('/')[-1][:-4]

    # get json
    out_file = os.path.join(out_path, '%s.json' % image_id)
    if os.path.exists(out_file):
        # for continuation
        print('Already processed %s' % image_id)
        continue
    print('processing %s...' % image_id)

    images = list(crop_region(image))
    result = {}
    for i, img in enumerate(images, 1):
        colors = []
        w_img = w_crop_region(img)
        test = {}
        for idx,cut_img in enumerate(w_img):
            color = cgm.extract(cut_img, 1)
            colors.extend(color)
            test[str(idx+1)] = get_rgb(colors[idx])
        result[str(i)] = test
    with open(out_file, 'w') as json_file:
        json_file.write(json.dumps(result))

    # except IndexError:
    #     print('Remove %s' % filename)
    #     os.remove(filename)

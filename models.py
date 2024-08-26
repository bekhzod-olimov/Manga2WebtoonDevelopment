import sys, os, cv2, torch, json, easyocr, requests, uuid, time, torchvision.transforms as transforms, numpy as np
import AttentionedDeepPaint.colorgram.colorgram as cgm
from AttentionedDeepPaint.preprocess import re_scale, make_colorgram_tensor, scale
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, _create_text_labels
from PIL import Image; from utils import load_colorization_model, load_det_seg_model
sys.path.append(os.getcwd())

################################################################# MODELS #################################################################

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COLORIZATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ColorizationModel:
    def __init__(self, device = "cuda:0", resize = (512, 512)):
        
        self.device, self.resize = device, resize
        self.style_model = load_colorization_model(checkpoint_path = f"./ckpts/colorization/best.tar", device = self.device)
        for param in self.style_model.parameters(): param.requires_grad = False

    def get_rgb(self, colorgram_result): return (colorgram_result.rgb.r, colorgram_result.rgb.g, colorgram_result.rgb.b)
    
    def crop_region(self, image):
        
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
    
    def style_color(self, line_art, style_img):
        
        transform_line = transforms.Compose([transforms.Resize(self.resize), transforms.ToTensor()])
        line_tensor = scale(transform_line(line_art)).unsqueeze(0).to(self.device)
        to_pil = transforms.ToPILImage()

        images = list(self.crop_region(style_img))
        result = {}
        for i, img in enumerate(images, 1):
            colors = cgm.extract(img, 5)
            result[str(i)] = {'%d' % i: self.get_rgb(colors[i]) for i in range(1, 5)}

        color_tensor = make_colorgram_tensor(result).unsqueeze(0).to(self.device)
        fakeB, _ = self.style_model(line_tensor, color_tensor)
        fakeB = to_pil(re_scale(fakeB.squeeze(0).detach().cpu())).resize(line_art.size)

        return fakeB

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ DETECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class DetectionModel():
    # def __init__(self, root = "/home/ubuntu/workspace/bekhzod/webtoon_dev"):
    #     sys.path.append(root)
    def __init__(self):
        cfg = get_cfg()
        cfg_save_path, weights_path = load_det_seg_model()
        cfg.merge_from_file(cfg_save_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATASETS.TEST = ("balloon",)
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)
        
    def detection(self, input_img):
        outputs = self.predictor(input_img)
        outputs = outputs["instances"].to("cpu")
        vis = Visualizer(input_img[:,:,::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        
        boxes, scores, classes = outputs.pred_boxes, outputs.scores, outputs.pred_classes.tolist()
        labels = _create_text_labels(classes, scores, vis.metadata.get("thing_classes", None))
        keypoints, masks, colors, alpha = None, None, None, 0.5

        v = vis.overlay_instances(masks = masks, boxes = boxes, labels = labels, alpha = alpha,
                                  keypoints = keypoints, assigned_colors = colors)
        
        array = v.get_image()[:, :, ::-1]
        overlayed = Image.fromarray(array).convert("RGB")
        
        boxes = vis._convert_boxes(boxes)
        areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        sorted_idxs = np.argsort(-areas).tolist()
        # Re-order overlapped instances in descending order.
        boxes = boxes[sorted_idxs] if boxes is not None else None
        labels = [labels[k] for k in sorted_idxs] if labels is not None else None
        
        pil_img = Image.fromarray(input_img)
        detections = [pil_img.crop((left, upper, right, lower)) for (left, upper, right, lower) in boxes]
            
        return overlayed, detections

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ OCR ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class OCRModel():
    def __init__(self):
        self.reader    = easyocr.Reader(["ch_sim", "en"])

    def readtext(self, im): return self.reader.readtext(im)

    def draw(self, im, threshold = 0.1):
        texts = []
        for bbox, text, score in self.readtext(im):
            if score > threshold:
                cv2.rectangle(im, tuple(map(int, bbox[0])), tuple(map(int, bbox[2])), (0, 255, 0), 3)
                texts.append(text)
                # cv2.putText(im, text, tuple(map(int, bbox[0])), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), 1)
        return im, "".join([f"{text} | "for text in texts])
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SEGMENTATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SegmentationModel():
    def __init__(self):
        self.device = "cuda:0"
        cfg = get_cfg()
        cfg_save_path, weights_path = load_det_seg_model(det_seg = "seg")
        cfg.merge_from_file(cfg_save_path)
        cfg.MODEL.WEIGHTS = weights_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.DATASETS.TEST = ("panel",)
        cfg.freeze()

        self.predictor = DefaultPredictor(cfg)
        
    def segpanel(self, input_img):
        outputs = self.predictor(input_img)
        
        v = Visualizer(input_img[:,:,::-1], metadata={}, scale=0.5, instance_mode=ColorMode.SEGMENTATION)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        array = v.get_image()
        overlayed = Image.fromarray(array).convert("RGB")
        
        pil_img = Image.fromarray(input_img)
        
        segments = []
        for pred_mask in outputs["instances"].to("cpu").pred_masks:
            mask = (pred_mask.numpy()*255).astype('uint8')
            pil_mask = Image.fromarray(mask)
            left, upper, right, lower = pil_mask.getbbox()

            cropped_image = pil_img.crop((left, upper, right, lower))
            cropped_mask = pil_mask.crop((left, upper, right, lower))

            segment_panel = Image.composite(cropped_image, Image.new("RGBA", cropped_image.size, (255, 255, 255, 0)), cropped_mask)
            segment_panel = segment_panel.convert("L")
            segments.append(segment_panel)
            
        return overlayed, segments
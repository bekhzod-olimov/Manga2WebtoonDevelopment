import os, torch, glob, cv2
from PIL import Image
from AttentionedDeepPaint.models import DeepUNetPaintGenerator

def load_det_seg_model(url = None, det_seg = "det"): 
    
    """
    
    This function gets several parameters and loads a detectron model.
    
    Parameters:
    
        checkpoint_dir  - path to the dir, str;
        url             - url to downloaded the pretrained model.
        
    Output:
    
        config and model pretrained weights paths.
    
    """
    checkpoint_dir = "ckpts/detection" if det_seg == "det" else "ckpts/segmentation"
    config_path, weight_path= f"{checkpoint_dir}/config.yaml", f"{checkpoint_dir}/best.pth"
    config_url  = "https://drive.google.com/file/d/1-VlFXySZXUUxlkDD8FHz64gypYu5Ewzr/view?usp=sharing" if det_seg == "det" else "https://drive.google.com/file/d/1xwUie5qlvLM0tqeL0HKkCodVcunYeD_0/view?usp=sharing"
    weight_url  = "https://drive.google.com/file/d/1jBYmteg1axZXlkI_vP18lFus6WKzrSYT/view?usp=sharing" if det_seg == "det" else "https://drive.google.com/file/d/1E9DvZTJAahpc0gZUrmIlgONwxjbMzZR5/view?usp=sharing"
    
    # Download from the checkpoint path
    if os.path.isfile(config_path) and os.path.isfile(weight_path): print("Detectron pretrained checkpoint is already downloaded!"); pass
    
    # If the checkpoint does not exist
    else:
        os.makedirs(checkpoint_dir, exist_ok = True)
        print("Pretrained checkpoint is not found!")
        print("Downloading the pretrained checkpoint...")
        
        # Get file ids
        config_id = config_url.split("/")[-2]
        weight_id = weight_url.split("/")[-2]
        
        # Download the checkpoint
        os.system(f"curl -L 'https://drive.usercontent.google.com/download?id={config_id}&confirm=xxx' -o {config_path}")
        os.system(f"curl -L 'https://drive.usercontent.google.com/download?id={weight_id}&confirm=xxx' -o {weight_path}")
    
    return config_path, weight_path

def load_colorization_model(checkpoint_path, optimizer = None, device = None):

    model = DeepUNetPaintGenerator().to(device)
    print("DeepUNetPaintGenerator model is constructed!")
    if os.path.isfile(checkpoint_path): print("Colorization pretrained checkpoint is already downloaded!"); pass
    else: 
        ckpt_url = "https://drive.google.com/file/d/1l3ZWqHjAwC7ES5t-zmgaHMomG4TY22SH/view?usp=sharing"
        ckpt_dir = checkpoint_path.split("/")[-2]
        os.makedirs(f"ckpts/{ckpt_dir}", exist_ok = True)
        ckpt_id = ckpt_url.split("/")[-2]
        os.system(f"curl -L 'https://drive.usercontent.google.com/download?id={ckpt_id}&confirm=xxx' -o {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location = device)
    model_state = checkpoint.get('model_state', None)
    optim_state = checkpoint.get('optimizer', None)

    model.load_state_dict(model_state)
    print("State dictionary is successfully loaded!")
    if optimizer is not None and optim_state is not None:
        optimizer.load_state_dict(optim_state)

    return model

def get_ims_captions(path):
        
    ims      = [Image.open(style_im).convert("RGB") for style_im in glob.glob(path)]
    captions = [f"Image #{i+1}" for i in range(len(ims))]

    return ims, captions



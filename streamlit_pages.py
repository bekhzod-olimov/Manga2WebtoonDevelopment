# Libraries
import cv2, numpy as np, streamlit as st
from streamlit_image_select import image_select
from PIL import Image; from utils import get_ims_captions
# Models
from models import SegmentationModel, DetectionModel, ColorizationModel, OCRModel

def segmentation_page(state):
    model = SegmentationModel()
    
    st.header("Upload a manga page or choose a page from the list")
    get_page = st.file_uploader('1', label_visibility='collapsed')
    ims_lst, image_captions = get_ims_captions(path = f"test_ims/segmentation/*.png")
    st.header("Please choose an input manga page:")
    page = image_select(label="", images = ims_lst, captions = image_captions)
    
    if get_page != None:
        try: page = cv2.imdecode(np.frombuffer(get_page.read(), np.uint8), cv2.IMREAD_COLOR)
        except Exception as e: print(e); st.write('Please upload another comic book page!')
            
    col1, col2 = st.columns(2)
    
    with col1: st.header("Uploaded Comic Book Page"); st.image(page)
    with col2: st.header("Segmented Comic Book Page")
        
    if 'seg_clicked' not in st.session_state: st.session_state.seg_clicked = False
        
    if 'seg_clicked_img' not in st.session_state: st.session_state.seg_clicked_img = None
        
    if 'segments' not in st.session_state: st.session_state.segments = None
        
    if 'overlayed' not in st.session_state: st.session_state.overlayed = None
    
    if st.button("Cut the uploaded comic book page"):
        overlayed, segments = model.segpanel(np.array(page))
        
        st.session_state.segments = segments
        st.session_state.overlayed = overlayed
        st.session_state.seg_clicked = True

    with col2:
        if st.session_state.overlayed != None: st.image(st.session_state.overlayed)
        else: st.write("Click 'Cut the uploaded comic book page' button to cut the original page")

    if not st.session_state.segments == None:
        st.header("Segmented Page")
        captions = [f"Segmented Page #{idx+1}" for idx in range(len(st.session_state.segments))]
        selected_seg = image_select(label = "", images = st.session_state.segments, captions = captions)
        st.image(selected_seg)
    
# Speech-balloon & Effect BBox Detection using Detectron2 trained model
def detection_page(state):
    model = DetectionModel()
    
    st.header("Upload a manga page or choose a page from the list")
    get_page = st.file_uploader('2', label_visibility='collapsed')
    st.header("Please choose an input manga page:")
    ims_lst, image_captions = get_ims_captions(path = f"test_ims/detection/*.png")
    
    page = image_select(label="", images = ims_lst, captions = image_captions)
    
    if get_page != None:
        try: page = cv2.imdecode(np.frombuffer(get_page.read(), np.uint8), cv2.IMREAD_COLOR)
        except: st.write('Please upload another comic book page!')
    
    col1, col2 = st.columns(2)
    
    with col1: st.header("Uploaded Comic Book Page"); st.image(page)
    with col2: st.header("Detected Speech Baloons")
    
    if 'det_clicked' not in st.session_state: st.session_state.det_clicked = False
    if 'det_clicked_img' not in st.session_state: st.session_state.det_clicked_img = None
    if 'detects' not in st.session_state: st.session_state.detects = None
    if 'overlayed_det' not in st.session_state: st.session_state.overlayed_det = None
    
    if st.button('Detect Speech Baloons'):
        overlayed, detects = model.detection(np.array(page))
        st.session_state.detects = detects
        st.session_state.overlayed_det = overlayed
        st.session_state.det_clicked = True
    
    with col2:
        if not st.session_state.overlayed_det == None: st.image(st.session_state.overlayed_det)
        else: st.write("Click 'Detect Speech Baloons'")
    
    if (st.session_state.detects != None and len(st.session_state.detects)) != 0:
        st.header("Detected Speech Baloons")
        captions = [f"Speech Balloon #{idx+1}" for idx in range(len(st.session_state.detects))]
        selected_det = image_select(label="", images = st.session_state.detects, captions = captions)
        st.image(selected_det)
    else: st.write("No speech baloon is detected!")
    
    
# Colorize lineart using trained Attentioned Deep Paint model
def colorization_page(state):
    model = ColorizationModel()
    
    ims_lst, image_captions = get_ims_captions(path = f"test_ims/colorization/*.png")
    sty_lst, style_captions = get_ims_captions(path = f"test_ims/styles/*.png")
    
    st.header("Upload a grayscale image or choose an image from the list")
    get_img = st.file_uploader('3', label_visibility='collapsed',)
    
    st.header("Please choose an input image:")
    input_img = image_select(label="", images = ims_lst, captions = image_captions)
    
    st.header("Please choose a neural style:")
    style_img = image_select(label="", images = sty_lst, captions = style_captions)
    
    if get_img != None:
        try: input_img = Image.open(get_img).convert("RGB")
        except: st.write('Please upload another image!')
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1: st.header("An Input Image");       st.image(input_img)

    with col2: st.header("A Neural Style Image"); st.image(style_img)

    with col3: st.header("A Colorized Image")
    
    with col4:
        btn1 = st.button('Colorize!')
        if btn1:
            with col3:
                with st.spinner('Colorizing... Please wait!'): result_img = model.style_color(input_img, style_img)
                st.image(result_img)
        else:
            with col3: st.write('Press "Colorize!" button')
        st.button("Reset", type = "primary")
    
def ocr_page(state):
    model = OCRModel()
    
    st.header("Please upload a speech balloon image or choose an image from the list below")
    get_img = st.file_uploader('4', label_visibility='collapsed')
    
    ims_lst, image_captions = get_ims_captions(path = f"test_ims/speech_baloons/*.jpg")
    
    if get_img != None:
        try: or_balloon = Image.open(get_img).convert("RGB")
        except: st.write('Please upload another image!')
    else:  or_balloon = image_select(label="", images = ims_lst, captions = image_captions)
        
    res, text = model.draw(np.array(or_balloon))
    
    col1, col2, col3 = st.columns(3)
    
    with col1: st.header("Original Image"); st.image(or_balloon)
    with col2: st.header("Detected Speech Balloon"); st.image(res)
    with col3: st.header("Text in the Speech Balloon"); st.write(text)
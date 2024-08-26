import pandas as pd
from PIL import Image,ImageDraw,ImageOps
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_option_menu import option_menu
import uuid
import json
import requests
import os 
import cv2
import numpy as np
st.set_page_config(layout="wide")

def check_imgsize(img):
    check = None
    if isinstance(img, Image.Image):
        check = True
        width, height = img.size
    elif isinstance(img, np.ndarray) and len(img.shape) == 3:
        check = False
        height , width, _ = img.shape

    if width > 700:
        new_width = 700
        new_height = int((height / width) * new_width)
        if check:
            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        else :
            img = cv2.resize(img,(new_width,new_height),interpolation = cv2.INTER_AREA)
    return img

def hex_to_rgb(hex):
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i:i+2], 16)
        rgb.append(decimal)

    return rgb

def mask_generate(data,img,bg_color):
    result = []
    for pts in data['polygon']:
        result.append(list(map(lambda x: [int(x[0] * img.shape[1]), int(x[1] * img.shape[0])], pts)))
    mask = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for i in result:
        cv2.fillPoly(mask, np.array([i]), color=(255, 255, 255))
    hsv_image = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    flat_list = [item for sublist in result for item in sublist]
    min_x = min(flat_list, key=lambda x: x[0])[0]
    min_y = min(flat_list, key=lambda x: x[1])[1]
    max_x = max(flat_list, key=lambda x: x[0])[0]
    max_y = max(flat_list, key=lambda x: x[1])[1]
    width = max_x - min_x
    height = max_y - min_y
    back_ground = np.full((height, width, 3),255, dtype=np.uint8)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 255, 255])

    white_mask = cv2.inRange(hsv_image, lower_white, upper_white)
    crop_mask =mask[min_y:max_y,min_x:max_x]
    masked = cv2.bitwise_and(img, mask)
    crop_img = masked[min_y:max_y,min_x:max_x]
    result_img = cv2.copyTo(crop_img,crop_mask,back_ground)
    
    mask[white_mask != 0] = hex_to_rgb(bg_color)

    dst = cv2.addWeighted(img, 0.8, mask, 1, 0)
    
    return dst,result_img

def page_Labeling(state):
    st.title('Labeling')
    st.subheader('*cut할 부분을 클릭해서 Labeling을 진행하세요.*')
    empty1,con1,empty2 = st.columns([0.01,1,0.01])
    con2,con3 = st.columns([0.5,0.5])
    with con1:
        bg_image = st.file_uploader("Background image:", type=["png", "jpg"])
        marker = st.radio("check your click",['Positive','Negative'])
        bg_color = st.color_picker("Background color : ", "#FF0000")
    with con2:
        if bg_image is not None:
            check_lst = []
            image = Image.open(bg_image)
            output_path = f'imgs/{bg_image.name}'
            image.save(output_path)
            image = check_imgsize(image)
            canvas_result = st_canvas(
                fill_color = '#FF0000' if marker == 'Positive' else '#0000FF',
                stroke_width=1,
                stroke_color= '#000000',
                background_image = image,
                update_streamlit=True,
                # width=600,
                # height=400,
                width=image.size[0],
                height=image.size[1],
                drawing_mode="point",
                point_display_radius=3,
                key="canvas",
                display_toolbar=True
            )
            with con3:
                if not canvas_result.json_data["objects"] == [] :
                        diction = {}
                        data_lst = []
                        data_dict = {}
                        point_lst = []
                        for data in canvas_result.json_data["objects"]:
                            check_lst.append(True if data["fill"]== '#FF0000' else False)
                            point_dict = {
                                "isPositive": True if data["fill"]== '#FF0000' else False,
                                "x": float(data["left"]/image.size[0]),
                                "y": float(data["top"]/image.size[1])
                                }
                            point_lst.append(point_dict)
                        if True not in check_lst:
                            st.write('잘못된 입력입니다. positive를 클릭후 이용해주세요')
                        else:
                            data_dict["imageId"] = str(uuid.uuid1())
                            data_dict["imgHeight"] = image.size[1]
                            data_dict["imgWidth"] = image.size[0]
                            data_dict["path"] = "/home/ubuntu/workspace/kwanwoo/webtoon/homepage/imgs"
                            data_dict["name"] = bg_image.name
                            data_dict["points"] = point_lst
                            diction["images"] = [data_dict]

                            response = requests.post("http://localhost:8420/AnySegdemo", data=json.dumps(diction))
                            result = json.loads(response.text)
                            new_img = cv2.imread(os.path.join(result['path'],result['name']))
                            new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
                            output,result_img = mask_generate(result,new_img,bg_color.replace('#',''))
                            cv2.imwrite(os.path.join(result['path'],'download_img.png'),result_img)
                            st.image(output)
                            # st.image(mask)
                            with open(os.path.join(result['path'],'download_img.png'), "rb") as file:
                                st.download_button(label="Download image",data=file,file_name="result.png",mime="image/png")
                            

def page_Inpainting(state):
    st.write('준비중')
    
    
def page_colorization(state):
    col1, col2 = st.columns(2)
    with col1:
        image_path = st.file_uploader("Upload sketch Image 🚀", type=["png","jpg","bmp","jpeg"])
    with col2:
        style_path = st.file_uploader("Upload sytle Image 🚀", type=["png","jpg","bmp","jpeg"])
    if image_path is not None and style_path is not None:
        output_path = f'imgs/{image_path.name.split(".")[0]}'
        with st.spinner("Working.. 💫"):
            image = cv2.imdecode(np.fromstring(image_path.read(), np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(output_path,'org.png'),image)
            
            style = cv2.imdecode(np.fromstring(style_path.read(), np.uint8), 1)
            style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(output_path,'style.png'),style)
            
            col3, col4, col5 = st.columns(3)
            with col3:
                st.success("Original Image")
                st.image(image)
            with col4:
                st.success("sytle Image")
                st.image(style)
            with col5:
                data_dict = {}
                data_dict['style_path'] = f'/home/ubuntu/workspace/kwanwoo/webtoon/homepage/{output_path}/style.png'
                data_dict['image_path'] = f'/home/ubuntu/workspace/kwanwoo/webtoon/homepage/{output_path}/org.png'
                data_dict['output_path'] = f'/home/ubuntu/workspace/kwanwoo/webtoon/homepage/{output_path}'
                response = requests.post("http://localhost:8421/colordemo", data=json.dumps(data_dict))
                result = json.loads(response.text)
                fake_img = cv2.imread(result['fake_path'])
                fake_img = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
                fake_img = cv2.resize(fake_img,(image.shape[1],image.shape[0]))
                st.success("Output Image")
                st.image(fake_img)
    else:
        st.warning('⚠ Please upload your Image! 😯')
import time
def main():
    # print_lst = ["Labeling", 'Inpainting','colorization']
    print_lst = ["Labeling", 'colorization']
    # st.write(st.session_state)
    if st.session_state.get('switch_button', False):
        # st.write(st.session_state)
        # time.sleep(10)
        st.session_state['menu_option'] = (st.session_state.get('menu_option',0) + 1) % 3
        manual_select = st.session_state['menu_option']
    else:
        manual_select = None
    with st.sidebar:
        selected = option_menu('webtoon', print_lst, 
            icons=['grid-1x2','brush','palette'], 
            menu_icon='book', manual_select=manual_select, key='state')
    if selected == "Labeling":
        page_Labeling(selected)
    elif selected == "Inpainting":
        page_Inpainting(selected)
    elif selected == "colorization":
        page_colorization(selected)
    # if st.button(f'Move to Next {((st.session_state.get("menu_option",0) + 1) % 3)}', key='switch_button'):
    #     st.write('ok')
    
main()



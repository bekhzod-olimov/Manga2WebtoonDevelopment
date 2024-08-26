# Libraries
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_pages import segmentation_page, detection_page, colorization_page, ocr_page
st.set_page_config(layout = "wide")

def main():
    print_lst = ["Segmentation", "Detection", "Colorization", "OCR"]
    if st.session_state.get('switch_button', False):
        st.session_state['menu_option'] = (st.session_state.get('menu_option',0) + 1) % 3
        manual_select = st.session_state['menu_option']
    else: manual_select = None
        
    with st.sidebar:
        selected = option_menu('Manga2Webtoon', print_lst,  icons=['grid-1x2','brush','palette', 'body-text'], 
                               menu_icon = "book", manual_select = manual_select, key = "state")
        
    if   selected == "Segmentation": segmentation_page(selected)

    elif selected == "Detection": detection_page(selected)
        
    elif selected == "Colorization": colorization_page(selected)
        
    elif selected == "OCR": ocr_page(selected)

main()
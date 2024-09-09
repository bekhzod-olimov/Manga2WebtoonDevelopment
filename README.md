# MANGA2WEBTOON
This repository contains a deep learning (DL)-based artificial intelligence (AI) complex system to convert Japanese comicbook (manga) pages into Korean Webtoon. The system uses various Computer Vision and multimodal tasks, such as:
* Semantic Segmentation              - [Detectron2](https://github.com/facebookresearch/detectron2) is used to cut original Japanese comicbook pages;
* Object Detection                   - [Detectron2](https://github.com/facebookresearch/detectron2) is used to detect speech baloons in the pages;
* Generative Adversarial Networks    - [Attentioned Deep Paint](https://github.com/ktaebum/AttentionedDeepPaint) is used to colorize grayscale manga images;
* Optical Character Recognition      - [EasyOCR](https://github.com/JaidedAI/EasyOCR) is used to read the contents of the speech balloons.

# These are the steps to use this repository:

1. Clone the repository:

`git clone https://github.com/bekhzod-olimov/Manga2WebtoonDevelopment.git`

`cd Manga2WebtoonDevelopment`

2. Create conda environment and activate it using the following script:
   
`conda create -n ENV_NAME python=3.10`

`conda activate ENV_NAME`

(if necessary) add the created virtual environment to the jupyter notebook (helps with debugging)

`python -m ipykernel install --user --name ENV_NAME --display-name ENV_NAME`

3. Install dependencies using the following script:

`pip install -r requirements.txt`

4. Train the models:

Train code is going to be available soon. All models in the system are trained using data that is collected and labelled by ourselves. Thus, the data is not publicly available yet. We trained all the models in the systems, such as Detectron2 and Attentioned Deep Paint. Regarding the EasyOCR model, it is used only for inference.

4. Run demo using the following script:

`streamlit run demo.py`

Please note that it will take considerable amount of time when the script is run for the first time. Because all tasks in the system (segmentation, detection, colorization) pretrained AI models' weights need to be downloaded.

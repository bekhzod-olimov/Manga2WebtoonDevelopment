# MANGA2WEBTOON
This repository contains a deep learning (DL)-based artificial intelligence (AI) complex system to convert Japanese comicbook (manga) pages into Korean Webtoon. The system uses various Computer Vision and multimodal tasks, such as:
* Semantic Segmentation              - [Detectron2](https://github.com/facebookresearch/detectron2) is used to cut original Japanese comicbook pages;
* Object Detection                   - [Detectron2](https://github.com/facebookresearch/detectron2) is used to detect speech baloons in the pages;
* Generative Adversarial Networks    - [Attentioned Deep Paint](https://github.com/ktaebum/AttentionedDeepPaint) is used to colorize grayscale manga images;
* Optical Character Recognition      - [EasyOCR](https://github.com/JaidedAI/EasyOCR) is used to read the contents of the speech balloons.

This repository is to develop an AI Model to transform Manga (Japanese Comic Pages) to Webtoon (Korean Animation).

These are the steps to use this repository:

1. Clone the repository:
`git clone https://github.com/bekhzod-olimov/Manga2WebtoonDevelopment.git`

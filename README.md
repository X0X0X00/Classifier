# Cat vs Dog Image Classifier with ResNet18 + Grad-CAM

This project builds a deep learning pipeline for binary image classification (cat vs dog) using PyTorch and ResNet18. It is specifically trained to distinguish between cat and dog photos only — images of other animals or objects may not be accurately recognized. The project also includes integrated Grad-CAM visualization, training monitoring, and a Gradio-powered web demo for easy interaction.

## Features

- **Model:** ResNet18 (customized for 2-class classification)
- **Training:** Full training loop with accuracy/loss curves and CSV logging
- **Visualization:** Grad-CAM heatmaps for interpretability
- **Deployment:** Gradio web interface for live image prediction
- **Auto-download dataset:** Filtered Cats and Dogs dataset (TensorFlow version)

## Demo Screenshot

![Training Curve](training_curve_resnet.png)

## Folder Structure
```
├───.gradio
│   └───flagged
│       ├───image
│       │   └───f26ab72748a62647b300
│       └───output
│           └───e117890c9dfc3bb3b604
├───data
│   └───cats_and_dogs_filtered
│       ├───train
│       │   ├───cats
│       │   └───dogs
│       └───validation
│           ├───cats
│           └───dogs
├───app.py
├───cat_dog_model_resnet18.pth
├───model.py
├───predict.py
├───requirements.txt
├───train.py
├───train_log_resnet.csv
├───train_resnet.py
├───train_curve_resnet.png
├───requirements.txt
├───utils.py
└───__pycache__
```

##  How to Run

### 1. Install dependencies (suggested environment: Python 3.9+)

```bash
pip install torch torchvision matplotlib gradio tqdm
```

### 2. Train the model (optional - model already provided)

```bash
python train_resnet.py
```

### 3. Run prediction (CLI)

```bash
python predict.py
```

### 4. Run the Gradio web interface

```bash
python app.py
```

Then open your browser:  
**Local:** http://127.0.0.1:7860  
**Public:** (auto-generated if `share=True` is set in `app.py`)

## Model Visualization (Grad-CAM)

We incorporate **Grad-CAM** to visualize what the model "sees" during classification.  
A heatmap overlay is generated for each prediction, highlighting the regions in the image that were most influential in the decision-making process.  
This improves **model transparency** and **user trust**.

## Dataset

We use the **`cats_and_dogs_filtered`** dataset from **TensorFlow's official repository**, which provides a clean and well-balanced set of labeled images for binary classification.

Our version contains:

- **24,349 training images** (cats and dogs, approximately balanced)  
- **1,000 validation images** for unbiased evaluation  

Compared to small toy datasets often used in tutorials, this dataset offers **significant scale and diversity**, enabling the model to learn robust visual features and generalize well to new cat and dog images.  
The dataset is **automatically downloaded** and organized by our `utils.py` script.

**Note:** This model is trained exclusively on cats and dogs. It may not provide accurate results for other animals or image categories.

## License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute this code for personal, educational, or commercial purposes, provided that proper attribution is given.

See the [LICENSE](./LICENSE) file for full license text.

MIT License

Copyright (c) 2025 [Zhenhao Zhang]

Permission is hereby granted, free of charge, to any person obtaining a copy...

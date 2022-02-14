#! /usr/bin/enc python
# -*- coding: utf-8 -*-
# author: Irving He 
# email: 1910646@tongji.edu.cn

# VIT
from transformers import ViTFeatureExtractor
from transformers import ViTForImageClassification
from PIL import Image
import requests


# Image Capture
url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

# Feature Extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# Model
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

# Inputs
inputs = feature_extractor(images=image, return_tensors="pt")

# Outputs
outputs = model(**inputs)
logits = outputs.logits

# Model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])
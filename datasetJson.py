import os
import cv2
import numpy as np
import json

dataset_dir = 'dataset'
output_json = 'annotations.json'

images = []
annotations = []
image_id = 0
annotation_id = 0

for folder in os.listdir(dataset_dir):
    folder_path = os.path.join(dataset_dir, folder)
    image_path = os.path.join(folder_path, 'image.jpg')

    img = cv2.imread(image_path)
    h, w, _ = img.shape

    images.append({
        'id': image_id,
        'file_name': f'{folder}/image.jpg',
        'height': h,
        'width': w
    })

    mask_dir = os.path.join(folder_path, 'masks')

    for mask_file in os.listdir(mask_dir):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Create binary mask
        ret, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask.astype(np.uint8)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) >= 6:  # At least 3 points
                segmentation.append(contour)

        if segmentation:
            area = int(mask.sum())
            annotations.append({
                'id': annotation_id,
                'image_id': image_id,
                'segmentation': segmentation,
                'category_id': 1,
                'area': area,
                'bbox': cv2.boundingRect(contours[0]),
                'iscrowd': 0
            })
            annotation_id += 1

    image_id += 1

coco_dict = {
    'images': images,
    'annotations': annotations,
    'categories': [{'id': 1, 'name': 'rock'}]  # Change class name as needed
}

with open(output_json, 'w') as f:
    json.dump(coco_dict, f)

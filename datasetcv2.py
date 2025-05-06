import subprocess
import json
import os
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
from scipy import ndimage


class RockAnnotator:
    def __init__(self):
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(device="cuda")
        self.predictor = SamPredictor(self.sam)

    def preprocess_image(self, image_path):
        """Preprocess image to enhance rock features"""
        image = cv2.imread(image_path)

        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Denoise
        denoised = cv2.fastNlMeansDenoising(gray)

        # Multi-scale edge detection
        edges = np.zeros_like(denoised)
        for sigma in [1, 2, 3]:  # Different scales for different sized rocks
            edges_temp = cv2.Canny(denoised, 50 * sigma, 150 * sigma)
            edges = cv2.bitwise_or(edges, edges_temp)

        # Dilate edges
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Process contours to rectangles
        rectangles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.2 < aspect_ratio < 5:  # Filter extreme aspect ratios
                    # Add a small padding to the rectangle
                    padding = 5
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2 * padding)
                    h = min(image.shape[0] - y, h + 2 * padding)
                    rectangles.append([x, y, x + w, y + h])

        return rectangles, image

    def create_labelme_json(self, image_path, rectangles):
        """Create LabelMe format JSON with rectangles"""
        image = cv2.imread(image_path)
        height, width = image.shape[:2]

        data = {
            "version": "4.5.7",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageHeight": height,
            "imageWidth": width
        }

        for rect in rectangles:
            shape = {
                "label": "rock",
                "points": [
                    [float(rect[0]), float(rect[1])],  # top-left
                    [float(rect[2]), float(rect[3])]  # bottom-right
                ],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {}
            }
            data["shapes"].append(shape)

        return data

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)

                # Get rectangles
                rectangles, image = self.preprocess_image(image_path)

                # Create export directory
                export_dir = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}.export")
                os.makedirs(export_dir, exist_ok=True)

                # Save image to export directory
                export_image_path = os.path.join(export_dir, filename)
                cv2.imwrite(export_image_path, image)

                # Create and save JSON
                json_data = self.create_labelme_json(image_path, rectangles)
                json_path = os.path.join(export_dir, f"{os.path.splitext(filename)[0]}.json")
                with open(json_path, 'w') as f:
                    json.dump(json_data, f, indent=2)

                # Save visualization
                debug_image = image.copy()
                for rect in rectangles:
                    cv2.rectangle(debug_image,
                                  (int(rect[0]), int(rect[1])),
                                  (int(rect[2]), int(rect[3])),
                                  (0, 255, 0), 2)
                cv2.imwrite(os.path.join(output_dir, f"debug_{filename}"), debug_image)

                # Refine with SAM
                self.refine_annotation(export_dir, output_dir)

    def refine_annotation(self, export_dir, output_dir):
        json_files = [f for f in os.listdir(export_dir) if f.endswith('.json')]
        if not json_files:
            print(f"No JSON file found in {export_dir}")
            return

        json_path = os.path.join(export_dir, json_files[0])

        with open(json_path, 'r') as f:
            data = json.load(f)

        image_path = os.path.join(export_dir, data['imagePath'])
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return

        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Set image in SAM predictor
        self.predictor.set_image(image_rgb)

        # Process each rectangle
        new_shapes = []
        for shape in data['shapes']:
            if shape['shape_type'] == 'rectangle':
                points = shape['points']
                box = [
                    points[0][0],  # x1
                    points[0][1],  # y1
                    points[1][0],  # x2
                    points[1][1]  # y2
                ]

                # Get SAM prediction
                masks, scores, _ = self.predictor.predict(
                    box=np.array(box),
                    multimask_output=True
                )

                # Use best mask
                best_mask = masks[scores.argmax()]

                # Convert mask to polygon
                contours = self._mask_to_contours(best_mask)

                # Add each contour as a new polygon shape
                for contour in contours:
                    new_shape = {
                        "label": "rock",
                        "points": contour.tolist(),
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    new_shapes.append(new_shape)

        # Update shapes in data
        data['shapes'] = new_shapes

        # Save refined annotation
        output_path = os.path.join(output_dir, os.path.basename(json_path))
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _mask_to_contours(self, mask):
        contours, _ = cv2.findContours(
            mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        simplified_contours = []
        for contour in contours:
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            simplified_contours.append(approx.reshape(-1, 2))

        return simplified_contours


def main():
    input_dir = "raw_images"
    output_dir = "refined_annotations"

    annotator = RockAnnotator()
    annotator.process_directory(input_dir, output_dir)

    print("Annotation process completed!")


if __name__ == "__main__":
    main()
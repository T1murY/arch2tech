import subprocess
import json
import os
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np


class RockAnnotator:
    def __init__(self):
        self.sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
        self.sam.to(device="cuda")
        self.predictor = SamPredictor(self.sam)

    def process_directory(self, input_dir, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        for filename in os.listdir(input_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(input_dir, filename)

                cmd = [
                    "labelmetk",
                    "ai-annotate-rectangles",
                    image_path,
                    #"--model", "sam",
                    "--texts", "rock",
                    "--score-threshold", "0.1",
                    "--iou-threshold", "0.5"
                ]
                subprocess.run(cmd)

                # Get the export directory path
                export_dir = os.path.join(input_dir, f"{os.path.splitext(filename)[0]}.export")
                if os.path.exists(export_dir):
                    self.refine_annotation(export_dir, output_dir)

    def generate_sam_prompts(self, image):
        """Generate automatic prompts for SAM using image analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Use adaptive thresholding to find potential rock regions
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2)

        # Find contours in the threshold image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Convert contours to bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter small regions
            if w > 20 and h > 20:
                boxes.append([x, y, x + w, y + h])

        return np.array(boxes)

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

        # Generate automatic SAM prompts
        auto_boxes = self.generate_sam_prompts(image)

        # Process both rectangle annotations and automatic prompts
        new_shapes = []
        processed_regions = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

        # First process manual annotations
        for shape in data['shapes']:
            if shape['shape_type'] == 'rectangle':
                points = shape['points']
                box = [
                    points[0][0],  # x1
                    points[0][1],  # y1
                    points[1][0],  # x2
                    points[1][1]  # y2
                ]

                masks, scores, _ = self.predictor.predict(
                    box=np.array(box),
                    multimask_output=True
                )

                best_mask = masks[scores.argmax()]
                processed_regions[best_mask] = 1

                contours = self._mask_to_contours(best_mask)
                for contour in contours:
                    new_shape = {
                        "label": "rock",
                        "points": contour.tolist(),
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    new_shapes.append(new_shape)

        # Then process automatic prompts for uncovered regions
        for box in auto_boxes:
            # Check if region is already covered
            roi = processed_regions[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            if np.mean(roi) < 0.5:  # Less than 50% overlap with existing annotations
                masks, scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=True
                )

                best_mask = masks[scores.argmax()]
                if scores.max() > 0.7:  # Confidence threshold
                    contours = self._mask_to_contours(best_mask)
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

        # Save refined annotation using original image filename
        original_filename = os.path.basename(export_dir).replace('.export', '.json')
        output_path = os.path.join(output_dir, original_filename)
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        # Copy the image to output directory
        output_image_path = os.path.join(output_dir, data['imagePath'])
        cv2.imwrite(output_image_path, image)

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
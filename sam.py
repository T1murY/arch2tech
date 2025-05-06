import torch
import cv2
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from segment_anything import SamAutomaticMaskGenerator
import svgwrite

def load_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def extract_contours(masks, image_shape):
    contours_list = []
    for mask_dict in masks:
        mask = mask_dict["segmentation"].astype(np.uint8)
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_list.extend(contours)
    return contours_list

def save_as_svg(contours, output_path):
    dwg = svgwrite.Drawing(output_path, profile='tiny')
    for contour in contours:
        points = [(float(p[0][0]), float(p[0][1])) for p in contour]
        if points:
            dwg.add(dwg.polyline(points, stroke='black', fill='none'))
    dwg.save()

def zoom(event, x, y, flags, param):
    global zoom_factor, overlay
    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            zoom_factor = min(zoom_factor + 0.1, 2.0)
        else:
            zoom_factor = max(zoom_factor - 0.1, 1.0)
        h, w = overlay.shape[:2]
        new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
        zoomed = cv2.resize(overlay, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        cv2.imshow("Rock Contours", zoomed)

def overlay_vector_on_image(img, contours):
    global overlay, zoom_factor
    overlay = img.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    zoom_factor = 1.0
    cv2.namedWindow("Rock Contours", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rock Contours", 1720, 1440)
    cv2.setMouseCallback("Rock Contours", mouse_callback)
    while True:
        cv2.imshow("Rock Contours", overlay)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'Esc' to exit
            break
    cv2.destroyAllWindows()

def mouse_callback(event, x, y, flags, param):
    global zoom_factor, img, img_display

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:  # Scroll up (zoom in)
            zoom_factor *= 1.2
        elif flags < 0:  # Scroll down (zoom out)
            zoom_factor = max(1.0, zoom_factor / 1.2)

        update_display(x, y)

def update_display(center_x, center_y):
    global zoom_factor, img, img_display

    # Get the dimensions of the original image
    h, w = img.shape[:2]

    # Calculate the size of the zoomed area
    zoom_w, zoom_h = int(w / zoom_factor), int(h / zoom_factor)

    # Compute the top-left corner of the zoomed area
    tl_x = max(0, center_x - zoom_w // 2)
    tl_y = max(0, center_y - zoom_h // 2)

    # Ensure the zoomed area is within the bounds of the image
    br_x = min(w, tl_x + zoom_w)
    br_y = min(h, tl_y + zoom_h)

    # Adjust top-left corner if the zoomed area exceeds the image dimensions
    tl_x = max(0, br_x - zoom_w)
    tl_y = max(0, br_y - zoom_h)

    # Ensure the width and height of the cropped area are valid
    if br_x > tl_x and br_y > tl_y:
        # Crop the zoomed area
        cropped = img[tl_y:br_y, tl_x:br_x]

        # Resize the cropped area back to the original display size
        img_display = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)


# Load SAM model
sam_checkpoint = "sam_vit_h_4b8939.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint).to(device)
predictor = SamPredictor(sam)
zoom_factor = 1.0
img = load_image("./imgs/1.jpg")
predictor.set_image(img)

mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(img)

contours = extract_contours(masks, img.shape)
save_as_svg(contours, "output.svg")
overlay_vector_on_image(img, contours)

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet101
import svgwrite


def load_model():
    model = deeplabv3_resnet101()
    model.eval()
    return model


def segment_rocks(image, model):
    transform = T.Compose([
        T.ToPILImage(),
        T.ToTensor()
    ])

    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)['out'][0]

    mask = output.argmax(0).byte().numpy()
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                      interpolation=cv2.INTER_NEAREST)  # Resize to original size
    return mask


def extract_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def save_as_svg(contours, output_path):
    dwg = svgwrite.Drawing(output_path, profile='tiny')
    for contour in contours:
        points = [(float(p[0][0]), float(p[0][1])) for p in contour]  # Convert to float
        if points:
            dwg.add(dwg.polyline(points, stroke='black', fill='none'))
    dwg.save()


def overlay_vector_on_image(image, contours):
    overlay = image.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)  # Green contours

    screen_res = (1920, 1080)  # Adjust to your screen resolution
    cv2.namedWindow("Rock Contours", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Rock Contours", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("Rock Contours", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image(image_path, output_svg):
    model = load_model()
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = segment_rocks(image, model)
    contours = extract_contours(mask)
    save_as_svg(contours, output_svg)

    overlay_vector_on_image(image, contours)
    print(f"SVG saved at: {output_svg}")

# Example usage
process_image('./imgs/1.jpg', './output_vectors/1.svg')

import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gradio as gr
from PIL import Image
import time

# Import SAM2 modules
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Configuration
MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"  # Update this path to your config file
CHECKPOINT_PATH = "fine_tuned_sam2_rock_18000.torch"  # Path to your fine-tuned model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Define a function to load the model
def load_model():
    try:
        # Load the base model first
        model = build_sam2(MODEL_CONFIG, DEVICE=DEVICE)

        # Load your fine-tuned weights
        if os.path.exists(CHECKPOINT_PATH):
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            print(f"Loaded fine-tuned model from {CHECKPOINT_PATH}")
        else:
            print(f"Warning: Fine-tuned model not found at {CHECKPOINT_PATH}")
            print("Using base model instead")

        # Create the predictor
        predictor = SAM2ImagePredictor(model)
        return predictor
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None


# Generate a grid of points to use as prompts for segmentation
def generate_grid_points(height, width, grid_size=20):
    points = []
    h_step = max(1, height // grid_size)
    w_step = max(1, width // grid_size)

    for h in range(h_step, height, h_step):
        for w in range(w_step, width, w_step):
            points.append([w, h])

    return np.array(points)


# Process the image and detect rocks
def detect_rocks(predictor, img):
    # Convert PIL Image to numpy array if needed
    if isinstance(img, Image.Image):
        img = np.array(img)

    # Ensure RGB format
    if img.shape[2] == 4:  # RGBA
        img = img[:, :, :3]

    # Resize image if it's too large
    max_size = 1024
    h, w = img.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h))

    # Set the image in the predictor
    predictor.set_image(img)

    # Generate grid points for prompting
    points = generate_grid_points(img.shape[0], img.shape[1], grid_size=15)

    # Add batch dimension
    points = points.reshape(-1, 1, 2)

    # Create labels (all foreground)
    labels = np.ones((points.shape[0], 1))

    # Get predictions
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        multimask_output=True,
    )

    return img, masks, scores


# Create a colored visualization of detected rocks
def create_visualization(image, masks, scores, threshold=0.6001, max_masks=50):
    # Sort masks by score
    sorted_indices = np.argsort(scores[:, 0])[::-1]

    # Take top N masks
    top_indices = sorted_indices[:max_masks]
    top_masks = masks[top_indices]
    top_scores = scores[top_indices, 0]

    # Create color overlay
    overlay = np.zeros_like(image, dtype=np.uint8)
    segments_overlay = np.zeros_like(image, dtype=np.uint8)

    # Get colormap
    colors = list(mcolors.TABLEAU_COLORS.values())

    # Combine masks with colors
    combined_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    for i, (mask, score) in enumerate(zip(top_masks, top_scores)):
        if score < threshold:
            continue

        color_idx = i % len(colors)
        color_rgb = mcolors.to_rgb(colors[color_idx])
        color_255 = [int(c * 255) for c in color_rgb]

        mask_area = mask[0].astype(bool)
        combined_mask[mask_area] = i + 1

        overlay[mask_area] = color_255
        segments_overlay[mask_area] = color_255

    # Blend the original image with the overlay
    alpha = 0.8
    blended = cv2.addWeighted(image, 1, overlay, alpha, 0)

    # Create separate visualization showing just the segments
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    axs[0].axis('off')

    axs[1].imshow(blended)
    axs[1].set_title("Rock Segments Overlay")
    axs[1].axis('off')

    axs[2].imshow(segments_overlay)
    axs[2].set_title("Rock Segments Only")
    axs[2].axis('off')

    plt.tight_layout()

    # Save figure to a file temporarily
    temp_file = f"temp_visualization_{time.time()}.png"
    plt.savefig(temp_file, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Read the file back
    result_image = cv2.imread(temp_file)
    result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    # Delete the temp file
    try:
        os.remove(temp_file)
    except:
        pass

    # Count detected rocks
    num_rocks = len([s for s in top_scores if s >= threshold])

    return result_image, combined_mask, num_rocks


# Define the Gradio interface function
def process_image(input_image, confidence_threshold=0.7):
    global predictor

    # Ensure the model is loaded
    if predictor is None:
        predictor = load_model()
        if predictor is None:
            return "Failed to load the model. Check console for errors.", None, "No rocks detected"

    # Process the image
    try:
        image, masks, scores = detect_rocks(predictor, input_image)

        # Create visualization
        result, segmentation_mask, num_rocks = create_visualization(
            image, masks, scores, threshold=confidence_threshold
        )

        rock_info = f"Detected {num_rocks} rock{'' if num_rocks == 1 else 's'}"
        for i in range(max(5, num_rocks)):
            rock_info += f"\nRock {i + 1} confidence: {scores[i][0]:.2f}"

        return result, segmentation_mask, rock_info

    except Exception as e:
        import traceback
        error_message = f"Error processing image: {str(e)}\n{traceback.format_exc()}"
        print(error_message)
        return input_image, None, f"Error: {str(e)}"


# Initialize the model
predictor = None

# Define the Gradio interface
with gr.Blocks(title="Rock Detector") as demo:
    gr.Markdown("# Rock Detector using SAM 2")
    gr.Markdown("Upload an image to detect and segment rocks.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input Image", type="pil")
            confidence = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=0.7,
                step=0.05,
                label="Confidence Threshold"
            )
            detect_button = gr.Button("Detect Rocks")

        with gr.Column():
            output_image = gr.Image(label="Rock Detection Results")
            segmentation_output = gr.Image(label="Segmentation Mask", visible=False)
            detection_info = gr.Textbox(label="Detection Information")

    detect_button.click(
        fn=process_image,
        inputs=[input_image, confidence],
        outputs=[output_image, segmentation_output, detection_info]
    )

    gr.Markdown("""
    ## How to use:
    1. Upload an image containing rocks
    2. Adjust the confidence threshold if needed
    3. Click "Detect Rocks"
    4. View the detected rock segments in the output

    The model has been fine-tuned specifically for rock detection.
    """)

# Launch the interface
if __name__ == "__main__":
    # Load the model at startup
    predictor = load_model()

    # Launch the interface
    demo.launch(share=False)  # set share=False if you don't want a public link
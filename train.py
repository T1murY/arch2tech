import os
import glob
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import hydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import torch
import torch.nn.utils


def create_dataset_from_rock_structure(dataset_path):
    """
    Create a dataset from the rock dataset structure:
    dataset/
    ├── 1.export/
    │   ├── image.jpg
    │   └── masks/
    │       ├── mask_0_rock.jpg
    │       ├── mask_1_rock.jpg
    │       └── ...
    """
    data_entries = []

    # Get all export folders
    export_folders = [f for f in os.listdir(dataset_path) if
                      os.path.isdir(os.path.join(dataset_path, f)) and "export" in f]

    for folder in export_folders:
        folder_path = os.path.join(dataset_path, folder)
        image_path = os.path.join(folder_path, "image.jpg")

        # Check if image exists
        if not os.path.exists(image_path):
            continue

        # Get all masks for this image
        masks_folder = os.path.join(folder_path, "masks")
        if not os.path.exists(masks_folder):
            continue

        mask_files = glob.glob(os.path.join(masks_folder, "*.jpg"))

        for mask_file in mask_files:
            data_entries.append({
                "image": image_path,
                "annotation": mask_file
            })

    # Create DataFrame
    data_df = pd.DataFrame(data_entries)

    # Split into train and test sets (80% train, 20% test)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(data_df, test_size=0.2, random_state=42)

    # Prepare the training data list
    train_data = []
    for index, row in train_df.iterrows():
        train_data.append({
            "image": row['image'],
            "annotation": row['annotation']
        })

    # Prepare the testing data list
    test_data = []
    for index, row in test_df.iterrows():
        test_data.append({
            "image": row['image'],
            "annotation": row['annotation']
        })

    return train_data, test_data
dataset_path = "./dataset"  # Change this to your dataset path
train_data, test_data = create_dataset_from_rock_structure(dataset_path)

print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")

def read_batch(data, visualize_data=False):
    # Select a random entry
    ent = data[np.random.randint(len(data))]

    # Get full paths
    img = cv2.imread(ent["image"])
    if img is None:
        print(f"Error: Could not read image from path {ent['image']}")
        return None, None, None, 0

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Read the mask
    mask = cv2.imread(ent["annotation"], cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error: Could not read mask from path {ent['annotation']}")
        return None, None, None, 0

    # For rock masks, we need to ensure they're binary (0 or 255)
    # Threshold to make sure it's binary
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Resize image and mask
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])  # Scaling factor
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))
    mask = cv2.resize(mask, (int(mask.shape[1] * r), int(mask.shape[0] * r)), interpolation=cv2.INTER_NEAREST)

    # Convert mask to binary format (0 or 1)
    binary_mask = (mask > 127).astype(np.uint8)

    # Erode the mask to avoid boundary points
    eroded_mask = cv2.erode(binary_mask, np.ones((5, 5), np.uint8), iterations=1)

    # Get coordinates inside the eroded mask and choose random points
    coords = np.argwhere(eroded_mask > 0)
    points = []

    if len(coords) > 0:
        # Select up to 3 points (or as many as available)
        num_points = min(3, len(coords))
        for _ in range(num_points):
            idx = np.random.randint(len(coords))
            yx = coords[idx]
            points.append([yx[1], yx[0]])  # Convert to x,y format

    points = np.array(points)

    if visualize_data and len(points) > 0:
        # Plotting the images and points
        plt.figure(figsize=(15, 5))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Rock Image')
        plt.imshow(img)
        plt.axis('off')

        # Segmentation Mask (binary_mask)
        plt.subplot(1, 3, 2)
        plt.title('Rock Mask')
        plt.imshow(binary_mask, cmap='gray')
        plt.axis('off')

        # Mask with Points
        plt.subplot(1, 3, 3)
        plt.title('Rock Mask with Points')
        plt.imshow(binary_mask, cmap='gray')

        # Plot points in different colors
        import matplotlib.colors as mcolors
        colors = list(mcolors.TABLEAU_COLORS.values())
        for i, point in enumerate(points):
            plt.scatter(point[0], point[1], c=colors[i % len(colors)], s=100, label=f'Point {i + 1}')

        plt.axis('off')
        plt.tight_layout()
        plt.show()

    # Format for model input
    binary_mask = np.expand_dims(binary_mask, axis=-1)  # Add channel dimension
    binary_mask = binary_mask.transpose((2, 0, 1))  # Change to CxHxW format

    if len(points) > 0:
        points = np.expand_dims(points, axis=1)  # Add dimension for compatibility

    # Return the image, binarized mask, points, and number of masks
    return img, binary_mask, points, 1  # Always 1 mask per image in this dataset


# Visualize a sample from the training data
if len(train_data) > 0:
    img, mask, points, num_masks = read_batch(train_data, visualize_data=True)



# Choose a model size

sam2_checkpoint = "sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
predictor = SAM2ImagePredictor(sam2_model)

# Train mask decoder and prompt encoder only
predictor.model.sam_mask_decoder.train(True)
predictor.model.sam_prompt_encoder.train(True)

# Configure optimizer
optimizer = torch.optim.AdamW(params=predictor.model.parameters(), lr=0.0001, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.2)

# Number of training steps
NO_OF_STEPS = 1000
FINE_TUNED_MODEL_NAME = "fine_tuned_sam2_rock"
accumulation_steps = 4

# Training loop
for step in range(1, NO_OF_STEPS + 1):
    with torch.cuda.amp.autocast():
        image, mask, input_point, num_masks = read_batch(train_data, visualize_data=False)

        # Skip if data loading failed or no points were found
        if image is None or mask is None or num_masks == 0 or input_point is None or len(input_point) == 0:
            continue

        input_label = np.ones((input_point.shape[0], 1))

        predictor.set_image(image)
        mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(input_point, input_label, box=None,
                                                                                mask_logits=None, normalize_coords=True)

        # Skip if prep_prompts failed
        if unnorm_coords is None or labels is None or unnorm_coords.shape[0] == 0 or labels.shape[0] == 0:
            continue

        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(unnorm_coords, labels), boxes=None, masks=None,
        )

        batched_mode = unnorm_coords.shape[0] > 1
        high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
        low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(
            image_embeddings=predictor._features["image_embed"][-1].unsqueeze(0),
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])

        gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
        prd_mask = torch.sigmoid(prd_masks[:, 0])

        # Calculate segmentation loss
        seg_loss = (-gt_mask * torch.log(prd_mask + 1e-6) - (1 - gt_mask) * torch.log(1 - prd_mask + 1e-6)).mean()

        # Calculate IoU scores
        inter = (gt_mask * (prd_mask > 0.5)).sum(1).sum(1)
        union = gt_mask.sum(1).sum(1) + (prd_mask > 0.5).sum(1).sum(1) - inter
        iou = inter / (union + 1e-6)

        score_loss = torch.abs(prd_scores[:, 0] - iou).mean()
        loss = seg_loss + score_loss * 0.05

        # Apply gradient accumulation
        loss = loss / accumulation_steps

    # Backward pass
    scaler.scale(loss).backward()

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(predictor.model.parameters(), max_norm=1.0)

    if step % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        predictor.model.zero_grad()

    # Update scheduler
    scheduler.step()

    # Save model periodically
    if step % 500 == 0:
        FINE_TUNED_MODEL = f"{FINE_TUNED_MODEL_NAME}_{step}.torch"
        torch.save(predictor.model.state_dict(), FINE_TUNED_MODEL)

    # Initialize mean IoU
    if step == 1:
        mean_iou = 0

    # Update mean IoU
    mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

    # Print progress
    if step % 100 == 0:
        print(f"Step {step}: Accuracy (IoU) = {mean_iou:.4f}, Loss = {loss.item():.4f}")


def read_image(image_path):
    """Read and preprocess a test image."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize if needed
    r = np.min([1024 / img.shape[1], 1024 / img.shape[0]])
    img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)))

    return img


def generate_grid_points(height, width, num_points=10):
    """Generate grid points to use for inference."""
    points = []
    h_step = height // (int(np.sqrt(num_points)) + 1)
    w_step = width // (int(np.sqrt(num_points)) + 1)

    for h in range(h_step, height, h_step):
        for w in range(w_step, width, w_step):
            points.append([w, h])

    return np.array(points)


# Select a test image
test_entry = random.choice(test_data)
image_path = test_entry['image']
mask_path = test_entry['annotation']

# Load the image
image = read_image(image_path)
if image is None:
    print(f"Failed to load image {image_path}")
else:
    # Generate grid points for inference
    points = generate_grid_points(image.shape[0], image.shape[1], num_points=16)
    points = np.expand_dims(points, axis=1)  # Add dimension for batch

    # Load the fine-tuned model
    FINE_TUNED_MODEL_WEIGHTS = f"{FINE_TUNED_MODEL_NAME}_3000.torch"  # Use the final model
    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")
    predictor = SAM2ImagePredictor(sam2_model)

    # Load the weights
    predictor.model.load_state_dict(torch.load(FINE_TUNED_MODEL_WEIGHTS))

    # Set model to evaluation mode
    predictor.model.eval()

    # Perform inference
    with torch.no_grad():
        predictor.set_image(image)
        masks, scores, logits = predictor.predict(
            point_coords=points,
            point_labels=np.ones([points.shape[0], 1])
        )

    # Process the predicted masks and sort by scores
    np_masks = np.array(masks[:, 0])
    np_scores = scores[:, 0]
    sorted_indices = np.argsort(np_scores)[::-1]  # Sort in descending order

    # Take the top 5 masks or less if fewer are available
    top_n = min(5, np_masks.shape[0])
    best_masks = np_masks[sorted_indices[:top_n]]
    best_scores = np_scores[sorted_indices[:top_n]]

    # Load the ground truth mask for comparison
    true_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    r = np.min([1024 / true_mask.shape[1], 1024 / true_mask.shape[0]])
    true_mask = cv2.resize(true_mask, (int(true_mask.shape[1] * r), int(true_mask.shape[0] * r)),
                           interpolation=cv2.INTER_NEAREST)
    true_mask = (true_mask > 127).astype(np.uint8)

    # Create a combined visualization
    plt.figure(figsize=(15, 10))

    # Original image
    plt.subplot(2, 3, 1)
    plt.title('Original Rock Image')
    plt.imshow(image)
    plt.axis('off')

    # Ground truth mask
    plt.subplot(2, 3, 2)
    plt.title('Ground Truth Mask')
    plt.imshow(true_mask, cmap='gray')
    plt.axis('off')

    # Show the best predicted masks
    for i in range(min(4, top_n)):
        plt.subplot(2, 3, i + 3)
        plt.title(f'Prediction {i + 1} (Score: {best_scores[i]:.2f})')

        # Create a color overlay
        overlay = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        overlay[best_masks[i] > 0.5] = [0, 255, 0]  # Green mask

        # Blend with original image
        alpha = 0.5
        blended = cv2.addWeighted(image, 1, overlay, alpha, 0)

        plt.imshow(blended)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Calculate IoU for the best mask
    if top_n > 0:
        pred_mask = best_masks[0] > 0.5
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        iou = intersection / union if union > 0 else 0
        print(f"Best mask IoU: {iou:.4f}")
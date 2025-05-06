import os


def rename_images(directory, extension=".jpg"):
    """
    Renames all images in the given directory to ascending numeric order.
    :param directory: Path to the folder containing images.
    :param extension: File extension to filter images (default: ".jpg").
    """
    images = [f for f in os.listdir(directory) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    images.sort()  # Sort files to maintain order

    for index, image in enumerate(images, start=1):
        old_path = os.path.join(directory, image)
        new_path = os.path.join(directory, f"{index}{extension}")
        os.rename(old_path, new_path)
        print(f"Renamed: {old_path} -> {new_path}")


if __name__ == "__main__":
    folder_path = "./imgs"
    rename_images(folder_path)

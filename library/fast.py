import os
import shutil
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from pathlib import Path
import sys
import tempfile
from slugify import slugify

# Add the parent directory to path to import captions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import captions

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def resize_image(image_path, output_path, size):
    """Resize an image while maintaining aspect ratio."""
    with Image.open(image_path) as img:
        width, height = img.size
        if width < height:
            new_width = size
            new_height = int((size/width) * height)
        else:
            new_height = size
            new_width = int((size/height) * width)
        print(f"resize {image_path} : {new_width}x{new_height}")
        img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
        img_resized.save(output_path)

def create_dataset(images, captions, destination_folder, size=1024):
    """
    Create a dataset with images and captions.
    
    Args:
        images: List of image file paths
        captions: List of corresponding captions
        destination_folder: Folder to save the dataset
        size: Size to resize images to
    """
    print("Creating dataset")
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    for index, image in enumerate(images):
        # Copy the image to the destination folder
        new_image_path = shutil.copy(image, destination_folder)

        # Skip if it's a caption text file
        ext = os.path.splitext(new_image_path)[-1].lower()
        if ext == '.txt':
            continue

        # Resize the image
        resize_image(new_image_path, new_image_path, size)

        # Create or use existing caption file
        original_caption = captions[index]
        image_file_name = os.path.basename(new_image_path)
        caption_file_name = os.path.splitext(image_file_name)[0] + ".txt"
        caption_path = os.path.join(destination_folder, caption_file_name)
        
        # If caption file exists, use it; otherwise create a new one
        if os.path.exists(caption_path):
            print(f"{caption_path} already exists. Using existing .txt file")
        else:
            print(f"{caption_path} create a .txt caption file")
            with open(caption_path, 'w') as file:
                file.write(original_caption)

    print(f"Dataset created at {destination_folder}")
    return destination_folder

@app.post("/caption_images/")
async def caption_images(
    model_name: str = Form(...),
    images: List[UploadFile] = File(...),
    size: int = Form(1024)
):
    # Create folder based on model name instead of UUID
    output_name = slugify(model_name)
    dataset_path = Path(f"datasets/{output_name}")
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded images to temporary files
    temp_image_paths = []
    for image in images:
        # Create a temporary file to store the image
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(image.filename)[1])
        temp_file.close()
        
        # Write image content to the temporary file
        with open(temp_file.name, "wb") as f:
            f.write(await image.read())
        
        temp_image_paths.append(temp_file.name)
    
    # Generate captions for each image
    image_captions = []
    for image_path in temp_image_paths:
        try:
            caption = captions.generate_caption(Path(image_path))
            image_captions.append(caption)
        except Exception as e:
            # Clean up temporary files
            for path in temp_image_paths:
                try:
                    os.unlink(path)
                except:
                    pass
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to generate caption: {str(e)}"}
            )
    
    # Create dataset with resized images and captions
    try:
        destination_folder = create_dataset(
            temp_image_paths, 
            image_captions, 
            str(dataset_path),
            size=size
        )
        
        # Clean up temporary files
        for path in temp_image_paths:
            try:
                os.unlink(path)
            except:
                pass
        
        return {
            "model_name": model_name,
            "dataset_path": destination_folder,
            "processed_files": len(image_captions)
        }
    
    except Exception as e:
        # Clean up temporary files
        for path in temp_image_paths:
            try:
                os.unlink(path)
            except:
                pass
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to process images: {str(e)}"}
        )

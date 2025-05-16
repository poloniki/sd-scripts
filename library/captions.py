import os
from openai import OpenAI
from pathlib import Path
import base64
from tqdm import tqdm
import re
import multiprocessing
from functools import partial

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def extract_caption_from_text(text):
    """Extract caption from different text formats."""
    # Try to find "Caption:" format
    if "Caption:" in text:
        return text.split("Caption:")[1].strip()
    
    # Try to find caption pattern in the response
    caption_match = re.search(r'caption:?(.*?)($|filename:)', text, re.IGNORECASE | re.DOTALL)
    if caption_match:
        return caption_match.group(1).strip()
    
    # If no specific format, just return the whole text as caption
    return text.strip()

def count_words(text):
    """Count words in a given text."""
    return len(text.split())

def generate_caption(image_path):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Encode the image
    base64_image = encode_image(image_path)
    
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"""
Create a concise, factual caption for this car image (maximum 30 words).

The caption should follow this structure: "A [color] <car_cheb> viewed from [angle], [lights status if visible], [number plate details], [wheel details], [location] [environmental context]."

For example: "A white <car_cheb> viewed from the rear three-quarter angle, with taillights off, visible rear license plate, black alloy wheels, parked in an outdoor lot on a cloudy day."

Important points to ONLY include:
- Car color
- Use "<car_cheb>" as the placeholder for the car brand (do not add any model name)
- View angle (front, rear, side, three-quarter, etc.)
- Lights status (on/off and which lights - headlights, taillights, etc.) ONLY if visible
- Number plate details (visible/not visible, front/rear, blacked-out, etc.)
- Wheel details (color, type if clearly visible)
- Location (road, parking lot, showroom, etc.)
- Environmental context (weather, time of day)

DO NOT include:
- Grille details
- Body styling opinions
- Interior features
- Any other car features not specifically listed above

Keep your description factual without styling opinions. Use a single, well-structured sentence that provides only the requested information efficiently.
"""
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/png;base64,{base64_image}"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "text"
            }
        },
        tools=[],
        store=True
    )
    
    # Extract the caption with robust handling
    try:
        # First try to get from output field
        if hasattr(response, 'output') and len(response.output) > 1:
            # Try to get from the second output item (index 1)
            text = response.output[1].content[0].text
        elif hasattr(response, 'output') and len(response.output) > 0:
            # If only one item in output, try that
            text = response.output[0].content[0].text
        elif hasattr(response, 'content') and len(response.content) > 0:
            # Try to get from content field
            text = response.content[0].text
        else:
            # Last resort - use string representation
            text = str(response)
            
        # Extract caption portion from text
        caption = extract_caption_from_text(text)
        
        # Check word count and warn if over limit
        word_count = count_words(caption)
        if word_count > 30:
            print(f"Warning: Caption for {image_path.name} has {word_count} words (exceeds 30 word limit)")
            
        return caption
    except Exception as e:
        # If we reach here, something went wrong with the response parsing
        raise Exception(f"Failed to parse response: {str(e)}, Response structure: {str(response)[:200]}...")

def process_single_image(image_path, captions_dir):
    """Process a single image and save its caption"""
    try:
        # Generate caption for the image
        caption = generate_caption(image_path)
        
        # Save caption to text file
        caption_file = captions_dir / f"{image_path.stem}.txt"
        with open(caption_file, "w") as f:
            f.write(caption)
        
        return image_path.name, True
    except Exception as e:
        return image_path.name, f"Error: {str(e)}"

def main():
    # Get the images directory
    images_dir = Path("images")
    
    # Create captions directory if it doesn't exist
    captions_dir = Path("captions")
    captions_dir.mkdir(exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in images_dir.glob("*") if f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    if not image_files:
        print("No images found in the images directory!")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Create a partial function with fixed captions_dir parameter
    process_func = partial(process_single_image, captions_dir=captions_dir)
    
    # Use multiprocessing with all available cores
    num_cores = multiprocessing.cpu_count()
    print(f"Processing with {num_cores} cores in parallel")
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=num_cores) as pool:
        # Process images in parallel with progress tracking
        results = []
        for result in tqdm(
            pool.imap_unordered(process_func, image_files),
            total=len(image_files),
            desc="Generating captions"
        ):
            results.append(result)
    
    # Check for any errors
    errors = [r for r in results if isinstance(r[1], str) and r[1].startswith("Error")]
    if errors:
        print(f"\n{len(errors)} images had errors:")
        for name, error in errors:
            print(f"- {name}: {error}")
    
    print("\nAll images processed!")

if __name__ == "__main__":
    main() 
import os
import ollama
from io import BytesIO
from PIL import Image

def perform_word_ocr(image_path):
    # 1. Verify the image file exists
    if not os.path.exists(image_path):
        print(f"Error: The file '{image_path}' could not be found.")
        return

    print(f"Loading image: {image_path}...")
    try:
        # 2. Open the image using PIL
        image = Image.open(image_path)
        
        # 3. Optimize image size for the Vision LLM
        # Moondream performs best when images are downscaled/upscaled close to standard sizes

        image.thumbnail((768, 768), Image.Resampling.LANCZOS)
        
        # 4. Convert the image to bytes for the Ollama API
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        
    except Exception as e:
        print(f"Failed to process image file: {e}")
        return

    # 5. Tailor the prompt for words and phrases
    model_name = 'moondream'
    prompt = (
        "Act as an advanced Handwritten Text Recognition (HTR) system. "
        "Carefully read the handwritten text, words, or characters in this image. "
        "Transcribe exactly what is written. "
        "Respond ONLY with the transcribed text. Do not add intro/outro text, "
        "do not say 'The word is', and do not include quotes unless they are written in the image."
    )
    
    print(f"Sending image to {model_name} for word recognition...")
    
    # 6. Request inference from Ollama
    try:
        response = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [image_bytes]
            }]
        )
        
        # 7. Output the transcription
        transcription = response['message']['content'].strip()
        print("\n==============================")
        print("    OCR TRANSCRIPTION RESULT   ")
        print("==============================")
        print(transcription)
        print("==============================\n")
        
    except Exception as e:
        print(f"\nError connecting to Ollama: {e}")
        print("Make sure the Ollama application is running and 'ollama pull moondream' was successful.")

if __name__ == "__main__":
    # Replace this with the path to your handwritten image file
    target_image = "LLM_approach\word_dog.png" 
    
    if not os.path.exists(target_image):
        print(f"'{target_image}' not found. Please place a handwritten image in this directory named '{target_image}' and run again.")
    else:
        perform_word_ocr(target_image)
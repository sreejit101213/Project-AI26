import os
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

script_dir = Path(__file__).resolve().parent
env_path = script_dir / '.env'
load_dotenv(dotenv_path=env_path)

HF_API_KEY = os.getenv("HF_TOKEN")

if HF_API_KEY:
    os.environ["HF_TOKEN"] = HF_API_KEY
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY

MODELS = [
    "black-forest-labs/FLUX.1-schnell",
    "stabilityai/stable-diffusion-xl-base-1.0",
    "runwayml/stable-diffusion-v1-5"
]

client = InferenceClient(api_key=HF_API_KEY)

def generate_image_from_text(prompt):
    for model in MODELS:
        try:
            print(f"Trying model: {model}...")
            image = client.text_to_image(prompt, model=model)
            return image
        except Exception as e:
            print(f"Failed: {model} | Error: {e}")
            print("Executing next...\n")
            continue
    return None

def post_process_image(image):
    image = ImageEnhance.Brightness(image).enhance(1.2)
    image = ImageEnhance.Contrast(image).enhance(1.3)
    return image.filter(ImageFilter.GaussianBlur(radius=2))

def main():
    print("Welcome to the Post-Processing Magic Workshop!")
    print("This program generates an image from text and applies post-processing effects.")
    print("Type 'quit', 'exit', 'q', or 'e' to exit\n")

    while True:
        user_input = input("Enter a description for the image: ")
        
        if user_input.strip().lower() in ["quit", "exit", "q", "e"]:
            print("Goodbye! Enjoy the experience your image I took 50 years to make with meticulous handwork to craft it. Hehe. Ok bye.")
            break

        if not user_input.strip():
            continue

        try:
            print("\nGenerating image...")
            image = generate_image_from_text(user_input)
            
            if image is None:
                print("Error: All models failed. Check your API Key or network.\n")
                continue

            print("Applying post-processing effects...\n")
            processed_image = post_process_image(image)
            processed_image.show()

            save_option = input("Do you want to save the processed image? (yes/no): ").strip().lower()
            if save_option in ["yes", "y"]:
                file_name = input("Enter a name for the image file (without extension): ").strip()
                if not file_name:
                    file_name = "output_image"
                
                output_dir = "Images"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filepath = os.path.join(output_dir, f"{file_name}_{timestamp}.png")
                
                processed_image.save(filepath)
                print(f"✅ Saved to folder: {filepath}\n")

            print("-" * 80 + "\n")
        except Exception as e:
            print(f"An error occurred: {e}\n")

if __name__ == "__main__":
    main()
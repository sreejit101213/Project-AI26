import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from PIL import Image, ImageEnhance, ImageFilter

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

def daylight_effect(image):
    image = ImageEnhance.Brightness(image).enhance(1.3)
    image = ImageEnhance.Contrast(image).enhance(1.1)
    return image.filter(ImageFilter.GaussianBlur(radius=1))

def night_mood_effect(image):
    image = ImageEnhance.Contrast(image).enhance(1.4)
    image = ImageEnhance.Brightness(image).enhance(0.9)
    return image.filter(ImageFilter.GaussianBlur(radius=0.5))

def main():
    print("--- AI Image Stylist: Your Signature Look ---")
    user_prompt = input("Enter a creative text description:\n")

    if not user_prompt.strip():
        print("Please enter a valid prompt.")
        return

    print("\nGenerating base image...")
    base_image = generate_image_from_text(user_prompt)

    if base_image is None:
        print("Error: All models failed. Check your API key or connection.")
        return

    print("Applying Daylight Edition effect...")
    daylight_img = daylight_effect(base_image)

    print("Applying Night Mood effect...")
    night_img = night_mood_effect(base_image)

    daylight_filename = f"{user_prompt}_daylight.png"
    night_filename = f"{user_prompt}_night.png"

    save_daylight = input(f"\nDo you want to save the Daylight image ({daylight_filename})? [yes/no]: ").strip().lower()
    if save_daylight in ["yes", "y"]:
        daylight_img.save(daylight_filename)
        print(f"✅ Saved: {daylight_filename}")

    save_night = input(f"Do you want to save the Night Mood image ({night_filename})? [yes/no]: ").strip().lower()
    if save_night in ["yes", "y"]:
        night_img.save(night_filename)
        print(f"✅ Saved: {night_filename}")

    print("\nDisplaying Daylight Edition...")
    daylight_img.show()

    print("Displaying Night Mood Edition...")
    night_img.show()

if __name__ == "__main__":
    main()
from flask import Flask, request, jsonify, render_template
from diffusers import StableDiffusionPipeline
import torch
import random
import os

app = Flask(__name__, template_folder="templates", static_folder="static")

# Load the pipeline (optimized for CPU)
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float32
).to("cpu")

# Directory for saving images
OUTPUT_DIR = "./static/images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate_images():
    try:
        # Get prompts from the request
        data = request.json
        prompts = data.get("prompts", [])
        
        if not prompts:
            return jsonify({"error": "No prompts provided"}), 400

        # Generate images for each prompt
        images = []
        for prompt in prompts:
            seed = random.randint(0, 100000)
            generator = torch.Generator("cpu").manual_seed(seed)
            image = pipe(prompt, generator=generator).images[0]

            # Save the image
            image_path = os.path.join(OUTPUT_DIR, f"output_{seed}.jpg")
            image.save(image_path)

            # Append the image URL to the list
            images.append(f"/static/images/output_{seed}.jpg")

        return jsonify({
            "message": "Images generated successfully!",
            "images": images
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)

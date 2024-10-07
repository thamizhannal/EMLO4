import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.catdog_classifier import DogBreedClassifier
from datamodules.catdog_datamodule import DogBreedDataModule
import argparse
from pathlib import Path

def load_and_preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

def inference(model, image_path, class_names):
    img_tensor = load_and_preprocess_image(image_path)
    img_tensor = img_tensor.to(model.device)

    model.eval()
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()

    predicted_label = class_names[predicted_class]
    confidence = probabilities[0][predicted_class].item()

    return predicted_label, confidence

def display_prediction(image_path, predicted_label, confidence, output_path):
    plt.figure(figsize=(10, 6))
    plt.axis('off')
    plt.title(f'Predicted: {predicted_label} (Confidence: {confidence:.2f})')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
def main(args):
    # Initialize DataModule to get class names
    data_module = DogBreedDataModule(data_dir="/app/data")
    data_module.prepare_data()
    data_module.setup(stage="fit")
    class_names = ["Beagle",
    "Boxer",
    "Bulldog",
    "Dachshund",
    "German_Shepherd",
    "Golden_Retriever",
    "Labrador_Retriever",
    "Poodle",
    "Rottweiler",
    "Yorkshire_Terrier"]
    num_classes = len(class_names)
    print("num classes:{}".format(num_classes))
    # Load the best model
    model = DogBreedClassifier.load_from_checkpoint(
        "/app/checkpoints/best-checkpoint-v1.ckpt",
        num_classes=num_classes
    )
    print("Model loaded sucessfully!")

    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(exist_ok=True, parents=True)
    image_files = list(input_folder.glob('*/*'))
    print("image_files={0}".format(len(image_files)))

    for image_file in image_files:
        if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            predicted_label, confidence = inference(model, image_file, class_names)    
            output_file = Path(output_folder).joinpath("predicted_"+image_file)
            print(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
            print("output_file:{0}".format(output_file))
            display_prediction(image_file, predicted_label, confidence, output_file)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Infer using trained Dog Breed Classifier")
    parser.add_argument("--input_folder", type=str, default="/app/data/extracted/valid", help="Path to input folder containing images")
    parser.add_argument("--output_folder", type=str, default="/app/data/output", help="Path to output folder for predictions")
    parser.add_argument("--ckpt_path", type=str, default="/app/checkpoints/best-checkpoint-v1.ckpt", help="Path to model checkpoint")
    args = parser.parse_args()
    main(args)

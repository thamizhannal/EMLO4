import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from models.catdog_classifier import DogBreedClassifier
from datamodules.catdog_datamodule import DogBreedDataModule

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

def display_prediction(image_path, predicted_label, confidence):
    img = Image.open(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {predicted_label} (Confidence: {confidence:.2f})')
    plt.show()

def main():
    # Initialize DataModule to get class names
    data_module = DogBreedDataModule(data_dir="../data")
    data_module.prepare_data()
    data_module.setup(stage="fit")
    class_names = data_module.train_dataset.classes
    num_classes = len(class_names)

    # Load the best model
    model = DogBreedClassifier.load_from_checkpoint(
        "checkpoints/best-checkpoint.ckpt",
        num_classes=num_classes
    )

    # Perform inference
    image_path = "../data/sample_image.jpg"  # Replace with your image path
    predicted_label, confidence = inference(model, image_path, class_names)

    # Display the result
    print(f"Predicted: {predicted_label} (Confidence: {confidence:.2f})")
    display_prediction(image_path, predicted_label, confidence)

if __name__ == "__main__":
    main()
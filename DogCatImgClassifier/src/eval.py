import lightning as L
from datamodules.catdog_datamodule import DogBreedDataModule
from models.catdog_classifier import DogBreedClassifier

def main():
    # Initialize DataModule
    data_module = DogBreedDataModule(data_dir="/app/data", batch_size=32, num_workers=4)
    data_module.prepare_data()
    data_module.setup(stage="test")

    # Get the number of classes
    num_classes = 10 #len(data_module.test_dataset.classes)

    # Load the best model
    model = DogBreedClassifier.load_from_checkpoint(
        "/app/checkpoints/best-checkpoint-v1.ckpt",
        num_classes=num_classes
    )

    # Initialize Trainer
    trainer = L.Trainer(accelerator="auto", devices=1 )#if L.pytorch.utilities.device_parser.parse_gpu_ids('auto') else None)

    # Evaluate the model
    results = trainer.test(model, datamodule=data_module)
    print(results)

if __name__ == "__main__":
    main()
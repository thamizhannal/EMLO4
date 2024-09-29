import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from datamodules.catdog_datamodule import DogBreedDataModule
from models.catdog_classifier import DogBreedClassifier

def main():
    # Set random seed for reproducibility
    L.seed_everything(42, workers=True)

    # Initialize DataModule
    data_module = DogBreedDataModule(data_dir="/app/data", batch_size=32, num_workers=2)
    data_module.prepare_data()
    data_module.setup()
    print("Initialize data module completed!")

    # Get the number of classes
    num_classes = len(data_module.train_dataset.classes)
    print("Num of the classes:{}", num_classes)

    # Initialize Model
    model = DogBreedClassifier(num_classes=num_classes, learning_rate=1e-3)
    print("Initialize DogBreedClassifier completed!")

    # Setup callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="best-checkpoint",
        save_top_k=1,
        verbose=True,
        monitor="val_loss",
        mode="min"
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min"
    )

    # Initialize Trainer
    trainer = L.Trainer(
        max_epochs=30,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator="cpu",
        devices=1, # if L.pytorch.utilities.device_parser.parse_gpu_ids('cpu') else None,
        logger=TensorBoardLogger("lightning_logs", name="dog_breed_classifier")
    )

    # Train the model
    trainer.fit(model, datamodule=data_module)

    # Test the model
    trainer.test(model, datamodule=data_module)

    print(f"Best model path: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()


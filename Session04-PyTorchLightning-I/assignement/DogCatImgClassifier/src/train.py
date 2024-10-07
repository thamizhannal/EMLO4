import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from datamodules.catdog_datamodule import DogBreedDataModule
from models.catdog_classifier import DogBreedClassifier

def main():
    # Set random seed for reproducibility
    L.seed_everything(42, workers=True)

    # Initialize DataModule
    data_module = DogBreedDataModule(data_dir="/app/data", batch_size=32, num_workers=4)
    data_module.prepare_data()
    data_module.setup(stage="fit")
    print("Initialize data module completed!")

    # Get the number of classes
    num_classes = 10 #len(data_module.train_dataset.classes)
    print("Num of the classes:{0}".format(num_classes))

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
    print("checkpoint callback completed!")

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode="min"
    )

    # Initialize Trainer
    trainer = L.Trainer(
        num_sanity_val_steps=0,
        log_every_n_steps=5,
        max_epochs=5,
        callbacks=[checkpoint_callback],
        accelerator="auto",
        devices= 1, #if L.pytorch.utilities.device_parser.parse_gpu_ids('cpu') else None,
        logger=TensorBoardLogger("dog_breed_classifier", name="dog_breed_classifier")
    )
    print("Initialize Trainer completed!")

    # Train the model
    trainer.fit(model, datamodule=data_module)
    print("Train.fit completed!")

    # Test the model
    trainer.test(model, datamodule=data_module)

    print(f"Best model path: {checkpoint_callback.best_model_path}")

if __name__ == "__main__":
    main()

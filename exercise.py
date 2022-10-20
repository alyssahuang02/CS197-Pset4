import math
import random
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

# Exercise 3: Imports for exercise 3
import os
import numpy as np
import logging
logging.getLogger().setLevel(logging.INFO)

# Importing wandb for exercise 1
import wandb
wandb.login()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_dataloader(is_train, batch_size, slice=5):
    full_dataset = torchvision.datasets.MNIST(
        root=".",
        train=is_train,
        transform=T.ToTensor(),
        download=True)
    sub_dataset = torch.utils.data.Subset(
        full_dataset,
        indices=range(0, len(full_dataset), slice))
    loader = torch.utils.data.DataLoader(
        dataset=sub_dataset,
        batch_size=batch_size,
        shuffle=is_train,
        pin_memory=True,
        num_workers=2)
    return loader


def get_model(dropout):
    "A simple model"
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28*28, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, 10)).to(DEVICE)
    return model


# Exercise 2: Create a wandb Table to log images, predictions, and labels
def log_image_table(images, predicted, labels):
    "Log a wandb.Table with (image, prediction, label)"
    table = wandb.Table(
        columns=["image", "prediction", "label"])
    for img, pred, label in \
            zip(images.to("cpu"),
                predicted.to("cpu"),
                labels.to("cpu")):
        table.add_data(
            wandb.Image(img[0].numpy()*255), pred, label)
    wandb.log({"predictions_table": table}, commit=False)


# Exercise 3: Making checkpoint saver class to save top_n models
class CheckpointSaver:
    def __init__(self, dirpath, decreasing=True, top_n=5):
        """
        dirpath: Directory path where to store all model weights 
        decreasing: If decreasing is `True`, then lower metric is better
        top_n: Total number of models to track based on validation metric value
        """
        if not os.path.exists(dirpath): os.makedirs(dirpath)
        self.dirpath = dirpath
        self.top_n = top_n 
        self.decreasing = decreasing
        self.top_model_paths = []
        self.best_metric_val = np.Inf if decreasing else -np.Inf
        
    def __call__(self, model, epoch, metric_val):
        model_path = os.path.join(self.dirpath, model.__class__.__name__ + f'_epoch{epoch}.pt')
        save = metric_val<self.best_metric_val if self.decreasing else metric_val>self.best_metric_val
        if save: 
            logging.info(f"Current metric value better than {metric_val} better than best {self.best_metric_val}, saving model at {model_path}, & logging model weights to W&B.")
            self.best_metric_val = metric_val
            torch.save(model.state_dict(), model_path)
            self.log_artifact(f'model-ckpt-epoch-{epoch}.pt', model_path, metric_val)
            self.top_model_paths.append({'path': model_path, 'score': metric_val})
            self.top_model_paths = sorted(self.top_model_paths, key=lambda o: o['score'], reverse=not self.decreasing)
        if len(self.top_model_paths)>self.top_n: 
            self.cleanup()
    
    def log_artifact(self, filename, model_path, metric_val):
        artifact = wandb.Artifact(filename, type='model', metadata={'Validation score': metric_val})
        artifact.add_file(model_path)
        wandb.run.log_artifact(artifact)        
    
    def cleanup(self):
        to_remove = self.top_model_paths[self.top_n:]
        logging.info(f"Removing extra models.. {to_remove}")
        for o in to_remove:
            os.remove(o['path'])
        self.top_model_paths = self.top_model_paths[:self.top_n]


def validate_model(
        model,
        valid_dl,
        loss_func,
        log_images=False,
        batch_idx=0):

    model.eval()
    val_loss = 0.
    with torch.inference_mode():
        correct = 0
        for i, (images, labels) in enumerate(valid_dl):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            val_loss += loss_func(outputs, labels).item() * labels.size(0)

            # Compute accuracy and accumulate
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()

            # Exercise 2: Log a batch of images in a table (always same batch_idx)
            if i == batch_idx and log_images:
                log_image_table(
                    images,
                    predicted,
                    labels)

    return val_loss / len(valid_dl.dataset), correct / len(valid_dl.dataset)

class Config():
    """Helper to convert a dictionary to a class"""
    def __init__(
        self,
        dict):
        "A simple config class"
        self.epochs = dict['epochs']
        self.batch_size = dict['batch_size']
        self.lr = dict['lr']
        self.dropout = dict['dropout']

# Exercise 4: Initializing sweep configuration
sweep_configuration = {
    "name": "Hyperparameter Sweep",
    "method": "grid",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "batch_size": {
            "values": [100,150,200]
        },
        "epochs": {
            "values": [5,10,15]
        },
        "lr": {
            "values": [1e-2, 1e-3, 1e-4]
        }
    }
}

def train():
    for _ in range(5):
        # Initalizing wandb for exercise 1
        wandb.init(project="cs197-pset4")
        # Exercise 4: Using sweep config values
        config_dict = {
            "epochs": wandb.config.epochs,
            "batch_size": wandb.config.batch_size,
            "lr": wandb.config.lr,
            "dropout": random.uniform(0.01, 0.80),
        }

        # Original config dict (commented out for the sweep)
        # config_dict = {
        #     "epochs": 10,
        #     "batch_size": 128,
        #     "lr": 1e-3,
        #     "dropout": random.uniform(0.01, 0.80),
        # }
        config = Config(config_dict)

        # Get the data
        train_dl = get_dataloader(
            is_train=True,
            batch_size=config.batch_size)
        valid_dl = get_dataloader(
            is_train=False,
            batch_size=2*config.batch_size)
        n_steps_per_epoch = \
            math.ceil(len(train_dl.dataset) / config.batch_size)
        
        # A simple MLP model
        model = get_model(config.dropout)

        # Make the loss and optimizer
        loss_func = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training
        example_ct = 0
        step_ct = 0

        # Exercise 3: Initializing CheckpointSaver
        checkpoint_saver = CheckpointSaver(dirpath='./model_weights', decreasing=True, top_n=3)
        for epoch in range(config.epochs):
            model.train()
            for step, (images, labels) in enumerate(train_dl):
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                outputs = model(images)
                train_loss = loss_func(outputs, labels)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
        
                example_ct += len(images)
                step_ct += 1

            val_loss, accuracy = validate_model(
                model,
                valid_dl,
                loss_func,
                log_images=(epoch == (config.epochs-1)))
            
            # Exercise 3: Updating model val loss
            checkpoint_saver(model, epoch, val_loss)

            print(f"Train Loss: {train_loss:.3f}, \
                Valid Loss: {val_loss:3f}, \
                Accuracy: {accuracy:.2f}")
            
            # Logging train loss, val loss, and accuracy in wandb for exercise 1
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "accuracy": accuracy})
        
        # Finishing wandb task for exercise 1
        wandb.finish()


if __name__ == "__main__":
    # Exercise 1-3: We just ran train() but we want to comment it out for Exercise 4
    # train()
    # Exercise 4: Initializing and running sweep
    sweep_id = wandb.sweep(sweep_configuration, project="cs197-pset4")
    wandb.agent(sweep_id, function=train)   
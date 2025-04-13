import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path

class TabularClassifiers(nn.Module):
    def __init__(
        self, 
        model: nn.Module,
        model_param, 
        columns
    ) -> None:
        super(TabularClassifiers, self).__init__()
        self.model = model(model_param, columns)
        if not hasattr(model_param, 'threshold'):
            model_param.threshold = 0.5  # Default threshold for binary classification
        self.model_param = model_param
        self.state_dict_path = model_param.state_dict_path

        if model_param.loss == 'BCE':
            self.criterion = nn.BCELoss()

        if model_param.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr= model_param.learning_rate,
            )

    def fit(self, train_loader):
        for epoch in range(self.model_param.epochs):
            self.model.train()
            running_loss = 0.0
            for _ , (features, labels) in enumerate(train_loader):
                outputs = self.model(features)
                labels = labels.view(-1, 1).float()
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.model_param.epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Save model
        torch.save(self.model.state_dict(), "heart_disease_model.pkl")
        print("Model saved as heart_disease_model.pkl")

    def load_model_from_dict_state(self):
        """Load the model from the specified path.

        Args:
            path (str): Path to the model file.
        """
        if self.state_dict_path:
            if self.state_dict_path.endswith('.pkl'):
                self.model.load_state_dict(torch.load(Path(self.state_dict_path), map_location='cpu', weights_only=True))
            else:
                raise ValueError("Invalid file format. Expected .pkl file.")	

    def compute_loss(self, data_loader: DataLoader, dataset_name: str = "Test"):
        """Compute the accuracy of the model on the given data loader.

        Args:
            data_loader (DataLoader): A DataLoader object containing the dataset to evaluate the model on.
            dataset_name (str): A string specifying the name of the dataset (e.g., "Validation", "Test").
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in data_loader:
                outputs = self.model(features)
                predicted = (outputs > self.model_param.threshold).float()
                labels = labels.view(-1, 1).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        if total > 0:
            print(f"{dataset_name} Accuracy: {accuracy * 100:.2f}%")
        else:
            print("Undefined (no samples in dataset)")
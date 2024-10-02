import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class TabularClassifiers(nn.Module):
    def __init__(
        self, 
        model,
        model_param, 
        columns
    ) -> None:
        super(TabularClassifiers).__init__()
        self.model = model(model_param, columns)
        self.model_param = model_param

        if model_param.loss == 'BCE':
            self.criterion = nn.BCELoss

        if model_param.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr= model_param.learning_rate,
            )

    def forward(self, train_loader):
        for epoch in range(self.model_param.epochs):
            self.model.train()
            running_loss = 0.0
            for _ , (features, labels) in enumerate(train_loader):
                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            print(f"Epoch [{epoch+1}/{self.model_param.epochs}], Loss: {running_loss/len(train_loader):.4f}")

        # Save model
        torch.save(self.model.state_dict(), "heart_disease_model.pth")
        print("Model saved as heart_disease_model.pth")


    def compute_loss(self, data_loader: DataLoader):
        """_summary_

        Args:
            data_loader (DataLoader): _description_
        """        
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for features, labels in data_loader:
                outputs = self.model(features)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
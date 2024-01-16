import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Load dataset from CSV
data = pd.read_csv("first_1000_rows.csv")  # Replace with your actual CSV file path

# Ensure that your CSV file has columns named 'text' and 'class'
X = data['text']
y = data['class']

# Tokenize text using CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X)

# Convert labels to numerical format
label_mapping = {'non-suicide': 0, 'suicide': 1}
y = y.map(label_mapping)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define a simple neural network model
class SimpleTextClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleTextClassifier, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return torch.sigmoid(self.fc(x))

# Define model and optimizer
input_size = X_train.shape[1]
output_size = 1  # Binary classification
model = SimpleTextClassifier(input_size, output_size)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Convert to PyTorch tensors
X_train_tensor = torch.Tensor(X_train.toarray())
y_train_tensor = torch.Tensor(y_train.values)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Training loop
num_epochs = 10

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs.squeeze(), batch_y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch + 1}/{num_epochs}, Average Loss: {average_loss}')

print("Training complete!")

# Evaluation
model.eval()

with torch.no_grad():
    # Convert test data to PyTorch tensors
    X_test_tensor = torch.Tensor(X_test.toarray())
    y_test_tensor = torch.Tensor(y_test.values)

    # Create DataLoader for test set
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    correct_predictions = 0
    total_samples = 0

    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        predictions = (outputs > 0.5).float()  # Assuming a threshold of 0.5 for binary classification

        correct_predictions += (predictions == batch_y).sum().item()
        total_samples += batch_y.size(0)

accuracy = correct_predictions / total_samples
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Switch back to evaluation mode
model.eval()

while True:
    user_input = input("Enter a text (type 'exit' to end): ")

    if user_input.lower() == 'exit':
        break

    # Tokenize the input using the same vectorizer used during training
    user_input_vectorized = vectorizer.transform([user_input])

    # Convert to PyTorch tensor
    user_input_tensor = torch.Tensor(user_input_vectorized.toarray())

    # Make a prediction
    with torch.no_grad():
        output = model(user_input_tensor)
        prediction = 'suicide' if output.item() > 0.5 else 'non-suicide'

    print(f'Model Prediction: {prediction}')

# DL- Developing a Neural Network Classification Model using Transfer Learning

## AIM
To develop an image classification model using transfer learning with VGG19 architecture for the given dataset.

## Problem Statement and Dataset
The problem statement for this experiment is to develop an image classification model that can accurately distinguish between 'defect' and 'notdefect' semiconductor chip images. This is a binary classification task, where the goal is to leverage transfer learning using a pre-trained VGG19 model to effectively classify new, unseen chip images.




## DESIGN STEPS
### STEP 1: 

Data Loading and Preprocessing: Load the chip image dataset, apply necessary transformations like resizing and converting to tensors, and create data loaders for efficient batch processing. This step also includes visualizing sample images and checking dataset statistics.

### STEP 2: 

Model Setup for Transfer Learning: Load a pre-trained VGG19 model, modify its final classification layer to match the binary classification task (defect/not defect), freeze the convolutional layers to retain pre-trained features, and define the loss function and optimizer.

### STEP 3: 
Model Training: Train the modified VGG19 model using the prepared training data. The training process will involve iterating through epochs, calculating loss, performing backpropagation, and updating the model's weights. Training and validation loss will be tracked and plotted.


### STEP 4: 
Model Evaluation and Reporting: Evaluate the trained model's performance on the test dataset. This includes calculating and displaying the test accuracy, generating and visualizing a confusion matrix, and printing a detailed classification report.


### STEP 5: 

Single Image Prediction: Demonstrate the model's predictive capability by performing inference on individual images from the test dataset and displaying the actual and predicted labels along with the image.






## PROGRAM

### Name:Sanchita Sandeep

### Register Number:212224240142

```python
def train_model(model, train_loader,test_loader,num_epochs=10):
  train_losses = []
  val_losses = []
  model.train()
  for epoch in range(num_epochs):
    # Training phase
    running_loss= 0.0
    for images , labels in train_loader:
      images, labels = images.to(device), labels.to(device)
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels.unsqueeze(1).float())

      loss.backward()
      optimizer.step()
      running_loss += loss.item()

    epoch_train_loss = running_loss / len(train_loader) # Calculate average training loss for the epoch
    train_losses.append(epoch_train_loss)

    # Validation phase
    model.eval()
    epoch_val_loss = 0.0 # Initialize validation loss for the current epoch
    with torch.no_grad():
      for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        epoch_val_loss += loss.item()

    epoch_val_loss = epoch_val_loss / len(test_loader) # Calculate average validation loss for the epoch
    val_losses.append(epoch_val_loss)
    model.train() # Set model back to training mode

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

  # Plot training and validation loss (outside the epoch loop)
  print("Name:  Sanchita Sandeep      ")
  print("Register Number:  212224240142`      ")
  plt.figure(figsize=(8, 6))
  plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
  plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training and Validation Loss')
  plt.legend()
  plt.show()


```

### OUTPUT

## Training Loss, Validation Loss Vs Iteration Plot
<img width="998" height="729" alt="Screenshot 2026-02-24 111958" src="https://github.com/user-attachments/assets/6143f7d7-93b6-48b5-b9f4-e1b738f8a431" />



## Confusion Matrix

<img width="864" height="749" alt="Screenshot 2026-02-24 111914" src="https://github.com/user-attachments/assets/366b4cf4-4c84-4acb-9665-2b07065e72b7" />


## Classification Report

<img width="591" height="261" alt="Screenshot 2026-02-24 111924" src="https://github.com/user-attachments/assets/4539ae21-7567-4b33-8918-5d2a13fb8f5c" />

### New Sample Data Prediction
<img width="583" height="493" alt="Screenshot 2026-02-24 111933" src="https://github.com/user-attachments/assets/69ad7dad-ffbb-4a83-abc0-d6576f34186e" />


## RESULT
The images are classified successfully

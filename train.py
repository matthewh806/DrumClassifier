import torch
import torch.nn.functional as F
import torch.nn as nn
'''
Runs the training loop for num_epochs using the model & training_dl
'''
def training(model, train_dl, num_epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, 
                                                    steps_per_epoch=int(len(train_dl)),
                                                    epochs=num_epochs,
                                                    anneal_strategy="linear")
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_prediction = 0
        total_prediction = 0

        for i, data in enumerate(train_dl):
            inputs, labels = data[0].to("cpu"), data[1].to("cpu")

            # Normalise inputs
            inputs_mean, inputs_std = inputs.mean(), inputs.std()
            inputs = (inputs - inputs_mean) / inputs_std

            # Zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            _, pred = torch.max(outputs, 1)

            correct_prediction += (pred == labels).sum().item()
            total_prediction += pred.shape[0]

            if i % 10 == 0:
                print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/10))
        
        num_batches = len(train_dl)
        avg_loss = running_loss / num_batches
        acc = correct_prediction / total_prediction
        print(f'Epoch: {epoch}, Loss: {avg_loss:.2f}, Accuracy: {acc:.2f}')

    print("Finished Training")
            
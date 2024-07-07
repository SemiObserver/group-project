import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Load AG News dataset
dataset = load_dataset("ag_news")
num_epochs = 6

URL = ['bert-base-uncased','albert-base-v2', 'microsoft/deberta-base']
model_name = ['BERT', 'albert-v2', 'deberta-v2']

for index in range(len(URL)):
    # Load pre-trained tokenizer and model for sequence classification
    tokenizer = AutoTokenizer.from_pretrained(URL[index])
    model = AutoModelForSequenceClassification.from_pretrained(URL[index], num_labels=4)  # Adjust num_labels for your specific task

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

    # Tokenize datasets
    tokenized_train = dataset['train'].map(tokenize_function, batched=True)
    tokenized_test = dataset['test'].map(tokenize_function, batched=True)

    # Use only half of the dataset
    train_size = len(tokenized_train) // 2
    test_size = len(tokenized_test)

    train_indices = list(range(train_size))
    test_indices = list(range(test_size))

    tokenized_train = Subset(tokenized_train, train_indices)
    tokenized_test = Subset(tokenized_test, test_indices)

    # Set the format of the datasets to PyTorch tensors
    tokenized_train.dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    tokenized_test.dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Set up data loaders
    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=16)
    eval_dataloader = DataLoader(tokenized_test, batch_size=32)

    # Set up optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=num_training_steps)

    # Initialize lists to store metrics
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    train_f1_scores = []
    learning_rates = []

    # Training loop
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        epoch_preds = []
        epoch_labels = []

        for batch_idx, batch in enumerate(tqdm(train_dataloader)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            epoch_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())

            # Print notification after each batch
#             print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_dataloader)} completed")

        # Calculate metrics for the epoch
        epoch_loss = np.mean(epoch_losses)
        epoch_accuracy = accuracy_score(epoch_labels, epoch_preds)
        epoch_f1 = f1_score(epoch_labels, epoch_preds, average='weighted')
        current_lr = optimizer.param_groups[0]['lr']

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        train_f1_scores.append(epoch_f1)
        learning_rates.append(current_lr)

        # Validation
        model.eval()
        val_epoch_losses = []
        val_epoch_preds = []
        val_epoch_labels = []

        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss = outputs.loss
                val_preds = outputs.logits.argmax(dim=-1)

                val_epoch_losses.append(val_loss.item())
                val_epoch_preds.extend(val_preds.cpu().numpy())
                val_epoch_labels.extend(labels.cpu().numpy())

        val_epoch_loss = np.mean(val_epoch_losses)
        val_epoch_accuracy = accuracy_score(val_epoch_labels, val_epoch_preds)

        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)

        # Print notification after each epoch
        print(f"Epoch {epoch+1}/{num_epochs} completed. Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}, F1 Score: {epoch_f1:.4f}")
    savedir = model_name[index] + ' ' + str(num_epochs) + 'num_epoch.pth'
    torch.save(model.state_dict(), savedir)
    # Final evaluation
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = outputs.logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate final metrics
    test_accuracy = accuracy_score(all_labels, all_preds)
    test_f1 = f1_score(all_labels, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test F1 Score: {test_f1:.4f}")

    # Plot loss curve
    plt.figure(figsize=(10, 6))
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, train_losses, '.-', label='Training Loss')
    plt.plot(epochs, val_losses, '.-', label='Validation Loss')
    plt.title('Training and Validation Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.xticks(epochs)
    plt.savefig(str(model_name[index])+'loss_curve.png')
    plt.close()

    # Plot accuracy curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accuracies, '.-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, '.-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.xticks(epochs)
    plt.savefig(str(model_name[index])+'accuracy_curve.png')
    plt.close()

    # Plot F1 score curve
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_f1_scores, '.-')
    plt.title('Training F1 Score vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.xticks(epochs)
    plt.savefig(str(model_name[index])+'f1_score_curve.png')
    plt.close()

    # Plot learning rate curve
    plt.figure(figsize=(10, 6))
    plt.scatter(epochs, learning_rates)
    plt.title('Learning Rate vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.xticks(epochs)
    plt.savefig(str(model_name[index])+'learning_rate_curve.png')
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(str(model_name[index])+'confusion_matrix.png')
    plt.close()
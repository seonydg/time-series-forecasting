import torch

def train(model, optimizer, train_dataloader, device):
    model.train()
    for idx, batch in enumerate(train_dataloader):
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)
        optimizer.zero_grad()

        logit = model(x)
        loss = torch.nn.functional.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()

        if idx % 50 == 0:
            print(f'Train Epoch: {e} [{idx*len(x)}/{len(train_dataloader.dataset)}({idx*len(x)/len(train_dataloader.dataset)*100:.6f}%)] \tLoss: {loss.item():.6f}')

def evaluate(model, val_dataloader, device):
    model.eval()
    corrects, total, total_loss = 0, 0, 0

    for batch in val_dataloader:
        x, y = batch.text.to(device), batch.label.to(device)
        y.data.sub_(1)

        logit = model(x)
        loss = torch.nn.functional.cross_entropy(logit, y, reduction='sum')
        total += y.size(0)
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    
    avg_loss = total_loss / len(val_dataloader.dataset)
    avg_accuracy = corrects / total

    return avg_loss, avg_accuracy
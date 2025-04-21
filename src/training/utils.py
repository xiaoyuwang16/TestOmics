class EarlyStopping:
    def __init__(self, patience, delta):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_loss):
        if self.best_score is None:
            self.best_score = current_loss
        elif current_loss > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if current_loss < self.best_score - self.delta:
                self.best_score = current_loss
            self.counter = 0
        return self.early_stop

def validate_model(model, test_loader, criterion, device):
    model.eval()
    total_valid_loss = 0.0
    total_samples = 0
    total_mae = 0.0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch['x'].to(device)
            targets = batch['y'].to(device)
            
            outputs = model(inputs)
            valid_loss = criterion(outputs.squeeze(), targets.squeeze())
            mae = torch.abs(outputs - targets).sum().item()
            
            total_valid_loss += valid_loss.item() * targets.size(0)
            total_mae += mae
            total_samples += targets.size(0)
    
    mean_valid_loss = total_valid_loss / total_samples
    mean_mae = total_mae / total_samples
    
    return mean_valid_loss, mean_mae
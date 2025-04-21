import torch
import torch.nn as nn
import torch.optim as optim
from src.training.utils import EarlyStopping

def trainer(train_loader, test_loader, model, lr, L2, num_epochs, patience, delta):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=L2)
    early_stopping = EarlyStopping(patience=patience, delta=delta)
    
    best_mae = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_mae = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                val_mae += torch.abs(outputs - batch_y).mean().item()
        
        val_mae /= len(test_loader)
        
        if val_mae < best_mae:
            best_mae = val_mae
            best_model_state = model.state_dict()
        
        if early_stopping(val_mae):
            break
            
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val MAE: {val_mae:.4f}')
    
    return best_model_state, train_loss, best_mae
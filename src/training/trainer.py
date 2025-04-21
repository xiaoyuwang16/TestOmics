import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def trainer(train_loader, test_loader, model, config):
    writer = SummaryWriter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config['lr'], 
        weight_decay=config['L2']
    )
    
    early_stopper = EarlyStopping(
        patience=config['patience'], 
        delta=config['delta']
    )
    
    best_mae = float('inf')
    best_model_state = None
    
    for epoch in range(config['num_epochs']):
        model.train()
        loss_record = []
        
        for batch in tqdm(train_loader):
            inputs = batch['x'].to(device)
            labels = batch['y'].to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.squeeze())
            loss.backward()
            optimizer.step()
            
            loss_record.append(loss.detach().item())
            
        mean_train_loss = sum(loss_record) / len(loss_record)
        
        mean_valid_loss, mean_mae = validate_model(
            model, test_loader, criterion, device
        )
        
        writer.add_scalars('Loss', {
            'train': mean_train_loss,
            'valid': mean_valid_loss
        }, epoch)
        writer.add_scalar('Mae/valid', mean_mae, epoch)
        
        if mean_mae < best_mae:
            best_mae = mean_mae
            best_model_state = model.state_dict()
            
        if early_stopper(mean_mae):
            break
            
    writer.close()
    return best_model_state, mean_train_loss, best_mae
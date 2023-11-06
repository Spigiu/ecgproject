import torch
import torch.nn as nn
from tqdm import tqdm


def train(model, device, train_loader, validation_loader, epochs,
          lr=0.01, weight_decay=0, criterion=nn.MSELoss()  , betas=(0.95, 0.999), verbose= True):
    
    #loss_fn= nn.MSELoss()
   
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=betas)
    train_loss, validation_loss = [], []
   
    with tqdm(range(epochs), unit='epoch') as tepochs:
        tepochs.set_description('Training')
       
        for epoch in tepochs:
        
            model.train()
            running_loss = 0
            correct, total = 0, 0
            for data in train_loader:
                #data, target = data.to(device), target.to(device)
                
                
                data = data.to(device)
                
               
                
                output = model(data) 
                
                
                optimizer.zero_grad()
                loss = criterion(output, data)  
                #loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                tepochs.set_postfix(loss=loss.item())
                running_loss += loss.item()
                
                #model.input_size = data.size(2)
              
            train_loss.append(running_loss / len(train_loader))
           
 # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for val_data in validation_loader:
                    val_data = val_data.to(device)
                  
                    
                    val_output = model(val_data)
                    val_loss += criterion(val_output, val_data).item()

            validation_loss.append(val_loss / len(validation_loader))

    return train_loss, validation_loss

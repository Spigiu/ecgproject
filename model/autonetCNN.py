import torch
import torch.nn as nn

class AutoencoderCNN(nn.Module):
    def __init__(self):
        super(AutoencoderCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size = 3, stride=1, padding=1),
           
           
         
            
            nn.Conv1d(16, 32, kernel_size= 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
           # nn.Sigmoid(),
            
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
          #  nn.Sigmoid(),
            
            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv1d(256 ,512,kernel_size= 3, stride=1, padding=1),
           # nn.Sigmoid(),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(0.5),
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(512,256 ,kernel_size= 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.Conv1d(256, 128, kernel_size= 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
            nn.ConvTranspose1d(128, 64, kernel_size= 3, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
           
            
            nn.ConvTranspose1d(64, 64, kernel_size= 3, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
            
            nn.ConvTranspose1d(64, 32, kernel_size= 3, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True), 
           # nn.Sigmoid(),
            
            nn.ConvTranspose1d(32, 16, kernel_size= 3, stride=1, padding=1, output_padding=0),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            
           
            nn.ConvTranspose1d(16, 1, kernel_size= 3, stride=1, padding=1, output_padding=0),
         # nn.Sigmoid(),
        )

    def forward(self, x):
       # x = x.squeeze(0)  # Rimuovi la dimensione aggiuntiva
    
        #x = torch.flatten(x, 1).unsqueeze(0)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

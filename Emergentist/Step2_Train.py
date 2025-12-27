
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import time
import os

class Custom3DDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y

def periodic_padding(x, pad_width):

    # Pad end dimension with periodic values
    x = torch.cat([x, x[:, :, :pad_width, :, :]], dim=2)  # Depth
    x = torch.cat([x, x[:, :, :, :pad_width, :]], dim=3)  # Height
    x = torch.cat([x, x[:, :, :, :, :pad_width]], dim=4)  # Width
    return x

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, stride = 2):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.emb_size = emb_size
        self.stride = stride
        self.conv = nn.Conv3d(in_channels, emb_size, kernel_size=patch_size, stride=stride, padding=0)
        
    def forward(self, x):
        pad_width = self.patch_size[0] - 1
        x = periodic_padding(x, pad_width)
        
        # Extract patches using convolution
        x = self.conv(x)  # [B, emb_size, D', H', W']
        x = x.flatten(2)  # [B, emb_size, N_patches]
        x = x.transpose(1, 2)  # [B, N_patches, emb_size]
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_size, num_heads, dropout = 0.1, ff_hidden_mult = 4):
        super(TransformerEncoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(emb_size)
        self.mha = nn.MultiheadAttention(emb_size, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(emb_size)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, ff_hidden_mult * emb_size),
            nn.ReLU(),
            nn.Linear(ff_hidden_mult * emb_size, emb_size)
        )
        self.dropout_layer = nn.Dropout(dropout)
        
    def forward(self, x):
        # Adjust dropout rate dynamically based on training or evaluation mode
        if self.training == False:
            self.mha.dropout = 0.0
            self.dropout_layer.p = 0.0

        attn_out, _ = self.mha(x, x, x)
        x = x + self.dropout_layer(attn_out)
        x = self.ln1(x)
        ff_out = self.ff(x)
        x = x + self.dropout_layer(ff_out)
        x = self.ln2(x)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, in_channels, patch_size, emb_size, num_layers, num_heads, num_patches, num_classes, dropout = 0.1, cutoff = 2):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, emb_size))
        self.dropout_layer = nn.Dropout(dropout)  # Dropout before the final head
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(emb_size=emb_size, num_heads=num_heads, dropout=dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(emb_size)
        self.head = nn.Linear(emb_size, num_classes, bias=True)
        self.cutoff = cutoff
        self._init_weights()

    def _init_layer(self, layer):
        if isinstance(layer, nn.Linear):
            nn.init.trunc_normal_(layer.weight, std=0.02)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.LayerNorm):
            nn.init.zeros_(layer.bias)
            nn.init.ones_(layer.weight)
            
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_layer)

    def forward(self, x):
        # Adjust dropout rate dynamically based on training or evaluation mode
        if self.training == False:
            self.dropout_layer.p = 0.0
            

        x = self.patch_embedding(x) # [B, N_patches, emb_size]
        # outputs['patch_embedding'] = x.clone()

        B, N, E = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, emb_size]
        x = torch.cat((cls_tokens, x), dim=1) # [B, N_patches+1, emb_size]
        x = x + self.pos_embedding
        x = self.dropout_layer(x)
        # outputs['pos_embedding'] = x.clone()

        x = x.transpose(0, 1) # (sequence_length, batch_size, embed_dim)
        #print(x.size())

        small_cubics_indices = []
        r = self.cutoff
        for i in range(0, 16, r):
            for j in range(0, 16, r):
                for k in range(0, 16, r):
                    # Initialize a list to store indices of 8 rows for this 2x2x2 cubic
                    small_cubic_index = [0]
                    # Iterate over the rows in this 2x2x2 cubic
                    for di in range(r):
                        for dj in range(r):
                            for dk in range(r):
                                # Calculate the index in the original 16x16x16 grid
                                index = ((i + di) % 16) * 16 * 16 + ((j + dj) % 16) * 16 + ((k + dk) % 16) + 1
                                # Append the index to the small cubic index list
                                small_cubic_index.append(index)
                    # Append the 2x2x2 cubic index to the list of small cubic indices
                    small_cubics_indices.append(small_cubic_index)
        
        #print(small_cubics_indices)
        for listIndex in small_cubics_indices:
            x_seg = x[listIndex]
            #print(x_seg)
            #print(listIndex)
            for layer in self.encoder_layers:
                x_seg = layer(x_seg)
            x[listIndex] = x_seg

        # for layer in self.encoder_layers:
        #     x = layer(x)
            # outputs[f'encoder_layer_{i}'] = x.clone()
        
        #cls_output = x[:, 0] # [B, emb_size]
        #out = self.regressor(cls_output) # [B, num_classes]

        x = x.transpose(0, 1)
        x = self.norm(x)
        # outputs['norm'] = x.clone()
        
        x = x[:, 0]  # Get the representation of the CLS token
        x = self.head(x)
        # outputs['head'] = x.clone()
        return x

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs=25):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 20)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        # for name, param in model.named_parameters():
                        #     if 'mha' in name and param.grad is not None:
                        #         print(f"Layer: {name} | Gradients: {param.grad}")
                 
                running_loss += loss.item() * inputs.size(0)
                
                if phase == 'train':
                    print(f'Batch {i+1}/{len(dataloaders[phase])} Loss: {loss.item():.7f}')
                    if i % 10 == 0:
                        end_time = time.time()
                        with open('output.txt', 'a') as file:
                            file.write(f'Time {end_time - start_time}: Batch {i+1}/{len(dataloaders[phase])} Loss: {loss.item():.7f}' + '\n')
            
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            end_time = time.time()
            with open('output.txt', 'a') as file:
                file.write(f'Time {end_time - start_time}: {phase} Loss: {epoch_loss:.7f}' + '\n')
        
        # Step the scheduler after each epoch
        scheduler.step()

        torch.save(model.state_dict(), f'./models/vision_transformer_model_epoch_{epoch}.pth')
    
    return model

def test_model(model, test_loader, criterion):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            print(criterion(targets, outputs))

if __name__ == "__main__":

    start_time = time.time()
    if os.path.exists('output.txt'):
        os.remove('output.txt')
        
    save_dir = './models'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)



    # Load Data
    data_train = torch.from_numpy(np.load('../data_train.npy')).to(torch.float32)
    data_val = torch.from_numpy(np.load('../data_valid.npy')).to(torch.float32)
    # data_test = torch.from_numpy(np.load('../data_test.npy')).to(torch.float32)
    targets_train = torch.from_numpy(np.load('../targets_train_norm.npy')).unsqueeze(1).to(torch.float32)
    targets_val = torch.from_numpy(np.load('../targets_valid_norm.npy')).unsqueeze(1).to(torch.float32)
    # targets_test = torch.from_numpy(np.load('../targets_test_norm.npy')).unsqueeze(1).to(torch.float32)
    train_dataset = Custom3DDataset(data_train, targets_train)
    val_dataset = Custom3DDataset(data_val, targets_val)
    # test_dataset = Custom3DDataset(data_test, targets_test)
    del data_train, data_val, targets_train, targets_val

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=32, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=32, shuffle=False)
    }

    # Parameters
    in_channels = 3
    patch_size = (2,2,2)
    emb_size = 64
    num_layers = 2
    num_heads = 2
    num_patches = 16*16*16
    num_classes = 1
    dropout_rate = 0.1
    cutoff = 4

    model = VisionTransformer(in_channels, patch_size, emb_size, num_layers, num_heads, num_patches, num_classes, dropout_rate, cutoff)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    # StepLR scheduler: Decreases the learning rate by gamma every step_size epochs
    scheduler = StepLR(optimizer, step_size = 20, gamma = 0.1)

    model = train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs = 40)
    torch.save(model.state_dict(), 'vision_transformer_model.pth')

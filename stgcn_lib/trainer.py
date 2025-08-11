# stgcn_project/stgcn_lib/trainer.py

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import glob

class Trainer:
    # 构造函数不再需要l_norm
    def __init__(self, model, config, dataloader):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")
        
        # 模型已经包含了l_norm作为buffer，可以直接移动到device
        self.model = model.to(self.device)
        self.dataloader = dataloader
        
        if config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])
        
        self.loss_fn = nn.MSELoss()
        
        self.checkpoint_dir = config['checkpoint']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = 0

        if config['checkpoint']['load_checkpoint']:
            self._load_checkpoint()

    def _load_checkpoint(self):
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'stgcn-*.pth'))
        if not checkpoints:
            print("No checkpoint found, starting from scratch.")
            return

        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        # 确保加载到正确的设备
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {self.start_epoch}")


    def _save_checkpoint(self, epoch):
        save_path = os.path.join(self.checkpoint_dir, f'stgcn-epoch-{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        print(f"Checkpoint saved to {save_path}")

    def train(self):
        print(f"Starting training on {self.device}...")
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            self.model.train()
            total_loss = 0
            
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            # 在循环中将数据移动到目标设备
            for X_batch, Y_batch in progress_bar:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.loss_fn(output, Y_batch)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.dataloader)
            print(f"Epoch {epoch+1} | Average Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.config['checkpoint']['save_every_epochs'] == 0:
                self._save_checkpoint(epoch)

        print("Training finished.")
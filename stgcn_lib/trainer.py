# stgcn_project/stgcn_lib/trainer.py

import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import os
import glob

class Trainer:
    # 构造函数现在接收训练和验证的dataloader
    def __init__(self, model, config, train_loader, val_loader):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        if config['training']['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['training']['learning_rate'])

        self.loss_fn = nn.MSELoss()

        self.checkpoint_dir = config['checkpoint']['save_dir']
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.start_epoch = 0

        if config['checkpoint']['load_checkpoint']:
            self._load_checkpoint()

    # ... _load_checkpoint 和 _save_checkpoint 方法保持不变 ...
    def _load_checkpoint(self):
        # ... (与之前版本相同)
        checkpoints = glob.glob(os.path.join(self.checkpoint_dir, 'stgcn-*.pth'))
        if not checkpoints:
            print("No checkpoint found, starting from scratch.")
            return

        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming training from epoch {self.start_epoch}")

    def _save_checkpoint(self, epoch):
        # ... (与之前版本相同)
        save_path = os.path.join(self.checkpoint_dir, f'stgcn-epoch-{epoch}.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
        print(f"Checkpoint saved to {save_path}")


    def _run_validation(self):
        """执行验证循环"""
        self.model.eval() # 设置为评估模式
        total_val_loss = 0
        with torch.no_grad(): # 禁用梯度计算
            for X_batch, Y_batch in self.val_loader:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)
                output = self.model(X_batch)
                loss = self.loss_fn(output, Y_batch)
                total_val_loss += loss.item()
        return total_val_loss / len(self.val_loader)

    def train(self):
        print(f"Starting training on {self.device}...")
        for epoch in range(self.start_epoch, self.config['training']['epochs']):
            self.model.train() # 设置为训练模式
            total_train_loss = 0

            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']} [Training]")
            for X_batch, Y_batch in progress_bar:
                X_batch, Y_batch = X_batch.to(self.device), Y_batch.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.loss_fn(output, Y_batch)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()
                progress_bar.set_postfix(train_loss=loss.item())

            avg_train_loss = total_train_loss / len(self.train_loader)
            avg_val_loss = self._run_validation()

            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}")

            if (epoch + 1) % self.config['checkpoint']['save_every_epochs'] == 0:
                self._save_checkpoint(epoch)

        print("Training finished.")
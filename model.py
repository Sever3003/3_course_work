import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


notes_dim = 100
feature_dim = 3
latent_dim = 100

class Encoder(nn.Module):
    def __init__(self, notes_dim=notes_dim, feature_dim=feature_dim, latent_dim=latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(notes_dim * feature_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3_mu = nn.Linear(100, latent_dim)
        self.fc3_logvar = nn.Linear(100, latent_dim)
        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = x.reshape(-1, notes_dim * feature_dim)
        h1 = F.relu(self.bn1(self.fc1(x)))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        return self.fc3_mu(h2), self.fc3_logvar(h2)

class Decoder(nn.Module):
    def __init__(self, notes_dim=notes_dim, feature_dim=feature_dim, latent_dim=latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, notes_dim * feature_dim)
        self.bn1 = nn.BatchNorm1d(200)
        self.bn2 = nn.BatchNorm1d(100)

    def forward(self, z):
        h1 = F.relu(self.bn1(self.fc1(z)))
        h2 = F.relu(self.bn2(self.fc2(h1)))
        output = torch.sigmoid(self.fc3(h2)).view(-1, notes_dim, feature_dim)
        
        pitch = output[:, :, 0].mul(127)
        pitch = torch.round(pitch)
        
        output = torch.cat((pitch.unsqueeze(2), output[:, :, 1:]), dim=2)
        
        return output


class VAE(nn.Module):
    def __init__(self, notes_dim = notes_dim, feature_dim = feature_dim, latent_dim = latent_dim):
        super(VAE, self).__init__()
        
        self.encoder = Encoder(notes_dim=notes_dim, feature_dim=feature_dim, latent_dim=latent_dim)
        self.decoder = Decoder(notes_dim=notes_dim, feature_dim=feature_dim, latent_dim=latent_dim)
        
    def forward(self, x):
        
        mu, log_var = self.encoder.forward(x)
        std = log_var.mul(0.5).exp_()
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mu)
        
        return self.decoder.forward(z), mu, log_var

        
    def loss_function(self, x, recon_x, mu, log_var):
        pitch_diff = recon_x[:, 1:, 0] - recon_x[:, :-1, 0]
        x_pitch_diff = x[:, 1:, 0] - x[:, :-1, 0]
        pitch_diff_loss = F.mse_loss(pitch_diff, x_pitch_diff, reduction='sum')
    
        pitch_loss = F.mse_loss(recon_x[:, :, 0], x[:, :, 0], reduction='sum')
        step_loss = F.mse_loss(recon_x[:, :, 1], x[:, :, 1], reduction='sum')
        duration_loss = F.mse_loss(recon_x[:, :, 2], x[:, :, 2], reduction='sum')
        kl_loss = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return 100*kl_loss + pitch_loss + step_loss + duration_loss + pitch_diff_loss
    
    def train_step(self, x, optimizer):
        self.train()
        optimizer.zero_grad()
        recon_x, mu, log_var = self.forward(x)
        loss = self.loss_function(x, recon_x, mu, log_var)
        loss.backward()
        optimizer.step()
        return loss.item()
        
    def train_model(self, train_loader, optimizer, epochs=10):
        for epoch in range(epochs):
            train_loss = 0
            for batch_idx, data in enumerate(train_loader):
                loss = self.train_step(data, optimizer)
                train_loss += loss
            
            print(f'Эпоха {epoch + 1}, Средние потери: {train_loss / len(train_loader.dataset)}')
        
    def generate_song(self, num_songs=10): # поменяем на z из распределения
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_songs, latent_dim)
            generated_songs = self.decoder(z)
        return generated_songs
    
class Discriminator(nn.Module):
    def __init__(self, notes_dim=notes_dim, feature_dim=feature_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(notes_dim * feature_dim, 100)
        self.fc2 = nn.Linear(100, 1)
        self.bn1 = nn.BatchNorm1d(100)

    def forward(self, x):
        x = x.view(-1, notes_dim * feature_dim)
        h1 = F.relu(self.bn1(self.fc1(x)))
        return torch.sigmoid(self.fc2(h1))
    
class VAEGAN(nn.Module):
    def __init__(self, notes_dim = notes_dim, feature_dim = feature_dim, latent_dim = latent_dim):
        super(VAEGAN, self).__init__()
        
        self.vae = VAE(notes_dim=notes_dim, feature_dim=feature_dim, latent_dim=latent_dim)
        self.discriminator = Discriminator(notes_dim=notes_dim, feature_dim=feature_dim)
    
    def forward(self, x):
        recon_x, mu, log_var = self.vae(x)
        return recon_x, mu, log_var
    
    def discriminator_loss(self, real_output, fake_output):
        real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
        fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        total_loss = (real_loss + fake_loss) / 2
        return total_loss
    
    def generator_loss(self, fake_output):
        return F.binary_cross_entropy(fake_output, torch.ones_like(fake_output))
    
    def train_step(self, x, optimizer_G, optimizer_D):
        # Обучение дискриминатора
        optimizer_D.zero_grad()
        
        real_output = self.discriminator(x)
        
        z = torch.randn(x.size(0), latent_dim)
        with torch.no_grad():
            fake_x = self.vae.decoder(z)
        fake_output = self.discriminator(fake_x)
        
        d_loss = self.discriminator_loss(real_output, fake_output)
        d_loss.backward()
        optimizer_D.step()
        
        # Обучение генератора (VAE)
        optimizer_G.zero_grad()
        recon_x, mu, log_var = self.vae(x)
        g_loss = self.vae.loss_function(x, recon_x, mu, log_var)
        
        fake_x = self.vae.decoder(z)
        fake_output = self.discriminator(fake_x)
        g_loss += self.generator_loss(fake_output)
        
        g_loss.backward()
        optimizer_G.step()
        
        return d_loss.item(), g_loss.item()

    def generate_song(self, num_songs=10):
        return self.vae.generate_song(num_songs)
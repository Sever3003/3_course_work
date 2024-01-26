import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class VAE(nn.Module):
    def __init__(self, inp_dim, latent_dim):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, inp_dim//2),
            nn.ReLU(),
            nn.Linear(inp_dim//2, latent_dim * 2)  # mu and log(sigma)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, inp_dim//2),
            nn.ReLU(),
            nn.Linear(inp_dim//2, inp_dim)
        )
        self.latent_dim = latent_dim
        self.inp_dim = inp_dim

    def forward(self, x):
        x_flatten = x.view(-1, self.inp_dim)
        mu, log_sigma = torch.split(self.encoder(x_flatten), self.latent_dim, dim=1)
        kl_loss = self.kl_loss(mu, log_sigma)
        sample = torch.exp(log_sigma) * torch.randn_like(log_sigma) + mu
        out = self.decoder(sample)
        return out.view(x.shape), kl_loss
    
    def kl_loss(self, mu, log_sigma):
        return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    
    def sample_vae(self, num_samples):
        # Генерация случайных латентных векторов
        latent_vectors = torch.randn(num_samples, self.latent_dim)

        # Преобразование латентных векторов в сэмплы
        samples = self.decoder(latent_vectors).view(num_samples, 100, 5)
        return samples
    

class VAEGAN(nn.Module):
    def __init__(self, inp_dim, notes_dim, latent_dim):
        super().__init__()
        
        # Энкодер: преобразует входные данные в параметры латентного пространства (mu и log(sigma))
        self.encoder = nn.Sequential(
            nn.Linear(inp_dim, inp_dim//2),
            nn.ReLU(),
            nn.Linear(inp_dim//2, latent_dim * 2)  # mu and log(sigma)
        )

        # Декодер: преобразует латентные переменные обратно в пространство исходных данных
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, inp_dim//2),
            nn.ReLU(),
            nn.Linear(inp_dim//2, inp_dim),
        )

        # Дискриминатор (или генератор): для оценки реалистичности сгенерированных данных
        self.discriminator = nn.Sequential(
            nn.Linear(inp_dim, inp_dim//2),
            nn.LeakyReLU(0.2),
            nn.Linear(inp_dim//2, inp_dim//4),
            nn.LeakyReLU(0.2),
            nn.Linear(inp_dim//4, 1),
            nn.Sigmoid()
        )

        self.latent_dim = latent_dim
        self.inp_dim = inp_dim
        self.notes_dim = notes_dim

    def encode(self, x):
        # возвращает параметры латентного распределения
        mu_logsigma = self.encoder(x)
        mu, logsigma = mu_logsigma.chunk(2, dim=1)
        return mu, logsigma

    def reparameterize(self, mu, logsigma):
        std = torch.exp(logsigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logsigma = self.encode(x)
        z = self.reparameterize(mu, logsigma)
        recon_x = self.decoder(z)
        return recon_x, mu, logsigma
    
    def kl_loss(self, mu, log_sigma):
        return -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

    def predict_next_note(self, x):
        # Предсказание следующей ноты
        # Здесь x - входные данные, которые могут быть текущей нотой или последовательностью нот
        mu, logsigma = self.encode(x)
        z = self.reparameterize(mu, logsigma)
        next_note = self.decoder(z)
        return next_note
    
    def discriminator_loss(self, real_output, fake_output):
        # Функция потерь для дискриминатора
        real_loss = F.binary_cross_entropy(real_output, torch.ones_like(real_output))
        fake_loss = F.binary_cross_entropy(fake_output, torch.zeros_like(fake_output))
        return real_loss + fake_loss
    
    def generate_song(self):
        self.eval()
        # Генерация случайного шума в латентном пространстве
        z = torch.randn(self.latent_dim)
        with torch.no_grad():
            generated_song = self.decoder(z)
        
        # Масштабирование и округление до диапазона 0-127 для первого столбца
        generated_song = generated_song.view(self.notes_dim, self.inp_dim // self.notes_dim)
        generated_song[:, 0] = (generated_song[:, 0] * 127).round()
 
        return generated_song
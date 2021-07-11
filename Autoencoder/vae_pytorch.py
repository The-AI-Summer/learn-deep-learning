import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F


def loss_function(recon_x, x, mu, logvar) -> Variable:
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= BATCH_SIZE * 784
    return BCE + KLD



class VAE():
    def __init__(self):
            super(VAE, self).__init__()
            # ENCODER
            self.fc1 = nn.Linear(784, 400)
            self.relu = nn.ReLU()
            self.fc21 = nn.Linear(400, 20)  # mu layer
            self.fc22 = nn.Linear(400, 20)  # logvariance layer
            # DECODER
            self.fc3 = nn.Linear(20, 400)
            self.fc4 = nn.Linear(400, 784)
            self.sigmoid = nn.Sigmoid()

    def encode(self, x: Variable) -> (Variable, Variable):
            h1 = self.relu(self.fc1(x))
            return self.fc21(h1), self.fc22(h1)
    def decode(self, z: Variable) -> Variable:
            h3 = self.relu(self.fc3(z))
            return self.sigmoid(self.fc4(h3))
    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:
            #mu :  mean matrix
            #logvar :  variance matrix
            if self.training:
                std = logvar.mul(0.5).exp_()  # type: Variable
                eps = Variable(std.data.new(std.size()).normal_())
                return eps.mul(std).add_(mu)
            else:
                return mu

model= VAE()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        
def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]

for epoch in range(1, EPOCHS + 1):
    train(epoch)
    test(epoch)
import torch
import torch.nn as nn

from pytorch_model_summary import summary


class Net(nn.Module):

    def __init__(self, input_size: int = 4, hidden_size: int = 50, num_layers: int = 1, out_size: int = 1) -> None:
        
        super(Net, self).__init__()
        
        # A questa rete sono passati 3 parametri:
        # - Il numero di features attese per l'input del layer LSTM.
        # - Il numero di celle interne al layer LSTM.
        # - Il numero di layer LSTM impilati, uno dopo l'altro.
        # - Il numero di features in output dal layer LSTM.
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size

        # Aggiungo strato costituito di celle LSTM.
        self.lstm = nn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size, 
                            num_layers = self.num_layers, 
                            batch_first = True)

        # Aggiungo uno strato fully-connected.
        self.linear = nn.Linear(self.hidden_size, 
                                self.out_size)
        
    def reset_state(self, batch_size: int) -> None:
        
        # Inizializzo hidden e cell state iniziale: h0 e c0.
        # - Calcolo batch_size dalla len() della sequenza.
        self.hidden = (torch.zeros( self.num_layers, batch_size, self.hidden_size),
                       torch.zeros( self.num_layers, batch_size, self.hidden_size))

    def forward(self, sequence : torch.Tensor) -> torch.Tensor:
        batch_size = sequence.size(0)
        
        # Inizializzo h0 e c0
        self.reset_state(batch_size)
        
        # Calcolo la dimensione da passare alla rete LSTM:
        # - Con batch_first a True, e' attesto (batch_size, sequence_length, input_size)
        # - batch_size la deduco dalla len() dei dati in ingresso.
        # - input_size e' nota.
        # - sequence_length la ottengo indirettamente (-1) dalle altre dimensioni.        
        
        x = sequence.view(len(sequence), -1, self.input_size)
        x, _ = self.lstm(x)
        x = self.linear(x[:, -1, :])
        return x

    
if __name__ == '__main__':
    
    # Crea l'oggetto che rappresenta la rete.
    # Fornisce le classi.
    n = Net(input_size=4)
    
    # Salva i parametri addestrati della rete.
    torch.save(n.state_dict(), './out/model_state_dict.pth')
    
    # Salva l'intero modello.
    torch.save(n, './out/model.pth')
    
    # # Stampa informazioni generali sul modello.
    print(n)

    # Stampa i parametri addestrabili.
    for name, param in n.named_parameters():
        if param.requires_grad:
            print(name, param.data)

    # Stampa un recap del modello.
    print(summary(n, torch.ones(size=[1, 40, 4])))
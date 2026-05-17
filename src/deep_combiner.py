import torch
import torch.nn as nn
import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.features import seed_everything
seed_everything(42)
class DynamicAlphaCombiner(nn.Module):
    def __init__(self, num_alphas):
        super(DynamicAlphaCombiner, self).__init__()
        
        # LSTM reads the sequence of alpha values and encodes market context
        self.context_lstm = nn.LSTM(input_size=num_alphas, hidden_size=16, batch_first=True)
        
        # Attention layer computes a dynamic weight for each alpha signal
        self.attention_scoring = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, num_alphas),
            nn.Softmax(dim=-1)  # Weights sum to 1 across all alphas
        )

    def forward(self, alphas_seq):
        # alphas_seq shape: [Batch, Time_1_to_T, Alpha_1_to_K]
        
        lstm_out, (h_n, _) = self.context_lstm(alphas_seq)

        context_vector = h_n[-1]
        
        # Compute dynamic attention weights from the context vector
        # attention_weights shape: [Batch, Alpha_1_to_K]
        attention_weights = self.attention_scoring(context_vector)
        
        # Take alpha values at the current timestep T
        current_alphas = alphas_seq[:, -1, :]
        
        # Weighted sum: multiply each alpha by its attention weight and sum
        mega_alpha_dynamic = torch.sum(attention_weights * current_alphas, dim=1)
        
        return mega_alpha_dynamic, attention_weights
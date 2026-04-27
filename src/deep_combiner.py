import torch
import torch.nn as nn

class DynamicAlphaCombiner(nn.Module):
    def __init__(self, num_alphas):
        super(DynamicAlphaCombiner, self).__init__()
        
        # Mạng đọc trạng thái thị trường (Context)
        self.context_lstm = nn.LSTM(input_size=num_alphas, hidden_size=16, batch_first=True)
        
        # Lớp tính toán trọng số Attention cho từng Alpha (Từ 1 đến num_alphas)
        self.attention_scoring = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, num_alphas),
            nn.Softmax(dim=-1) # Đảm bảo tổng trọng số = 1
        )

    def forward(self, alphas_seq):
        # alphas_seq có dạng: [Batch, Time_1_to_T, Alpha_1_to_K]
        
        # 1. Trích xuất bối cảnh thời gian
        lstm_out, _ = self.context_lstm(alphas_seq)
        
        # Lấy trạng thái ở bước thời gian cuối cùng T
        _, (h_n, _) = self.context_lstm(alphas_seq)
        context_vector = h_n[-1]
        
        # 2. Tính toán trọng số động
        # attention_weights có dạng: [Batch, Alpha_1_to_K]
        attention_weights = self.attention_scoring(context_vector)
        
        # Lấy giá trị Alphas ở ngày hiện tại T
        current_alphas = alphas_seq[:, -1, :]
        
        # 3. Tổ hợp: Nhân từng Alpha với trọng số tương ứng rồi cộng lại
        mega_alpha_dynamic = torch.sum(attention_weights * current_alphas, dim=1)
        
        return mega_alpha_dynamic, attention_weights
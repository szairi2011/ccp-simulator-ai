# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: .ccp_simulator_venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# This code is suggested by Tabnine using the following prompt:
# Prompt: "I need to build a custom transformer model to predict an output sequence from an input one. Provide a concise Pytorch codebase to train a decoder transformer model on a custom dataset. Don't use a pre-trained model, we need to build all the transformer layers from scratch using Pytorch API."


# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# %%
# Define the custom tokenizer
class Tokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.token_dict = {i: chr(97 + i) for i in range(vocab_size)}
        self.inv_token_dict = {v: k for k, v in self.token_dict.items()}

    def encode(self, text):
        return [self.token_dict[c] for c in text]

    def decode(self, encoded_text):
        return ''.join([self.inv_token_dict[i] for i in encoded_text])

# %%
# Define the custom dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text = self.data[idx][0]
        target_text = self.data[idx][1]

        input_ids = self.tokenizer.encode(input_text)
        target_ids = self.tokenizer.encode(target_text)

        if len(input_ids) < self.max_len:
            input_ids = input_ids + [0] * (self.max_len - len(input_ids))

        if len(target_ids) < self.max_len:
            target_ids = target_ids + [0] * (self.max_len - len(target_ids))

        return {
            'input_ids': torch.tensor(input_ids),
            'target_ids': torch.tensor(target_ids)
        }

# %%
# Define the positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


# %%
# Define the custom transformer model
class CustomTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, n_heads, dropout_rate):
        super(CustomTransformer, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.dropout_rate = dropout_rate

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len)

        self.transformer_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, n_heads, dropout_rate)
            for _ in range(n_layers)
        ])

        self.linear_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask=None):
        input_embeds = self.embedding(input_ids)
        input_embeds = self.pos_encoding(input_embeds)

        for layer in self.transformer_layers:
            input_embeds = layer(input_embeds, attention_mask)

        output = self.linear_out(input_embeds)
        return output

# %%
# Define the positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# %%
# Define the transformer layer
class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super(TransformerLayer, self).__init__()

        self.multihead_attention = MultiheadAttention(d_model, n_heads, dropout_rate)
        self.ffn = FeedForward(d_model, dropout_rate)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_embeds, attention_mask):
        attn_output, _ = self.multihead_attention(input_embeds, input_embeds, input_embeds, attention_mask)
        attn_output = self.dropout(attn_output)
        out1 = self.norm1(attn_output + input_embeds)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout(ffn_output)
        out2 = self.norm2(ffn_output + out1)

        return out2

# %%
# Define the multihead attention
class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_rate):
        super(MultiheadAttention, self).__init__()

        self.n_heads = n_heads
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value, attention_mask):
        batch_size, seq_len, _ = query.size()

        query = self.q_proj(query)
        key = self.k_proj(key)
        value = self.v_proj(value)

        query = query.view(batch_size, seq_len, self.n_heads, -1)
        key = key.view(batch_size, seq_len, self.n_heads, -1)
        value = value.view(batch_size, seq_len, self.n_heads, -1)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_model // self.n_heads)

        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e9)

        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        output = torch.matmul(attention_probs, value)
        output = output.view(batch_size, seq_len, self.n_heads * self.d_model)
        output = self.out_proj(output)

        return output, attention_probs

# %%
# Define the feedforward network
class FeedForward(nn.Module):
    def __init__(self, d_model, dropout_rate):
        super(FeedForward, __init__).__init__()

        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(-1, x.size(-1))
        x = self.linear2(torch.relu(self.linear1(x)))
        x = self.dropout(x)
        return x.view(x.size(0), x.size(1))

# %%
# Define the training loop
def train(model, optimizer, data_loader, device, epochs):
    for epoch in range(epochs):
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)

            optimizer.zero_grad()

            output = model(input_ids)
            loss = nn.CrossEntropyLoss()(output, target_ids)
            loss.backward()
            optimizer.step()

# %%
# Define the main function
def main():
    # Define the hyperparameters
    input_dim = 100
    hidden_dim = 512
    output_dim = 100
    n_layers = 6
    n_heads = 8
    dropout_rate = 0.1
    max_len = 50
    batch_size = 32
    lr = 0.001
    epochs = 10

    # Define the tokenizer
    tokenizer = Tokenizer(input_dim)

    # Define the custom dataset
    data = [
        ('input_text_1', 'target_text_1'),
        ('input_text_2', 'target_text_2'),
        # Add more data here
    ]
    dataset = CustomDataset(data, tokenizer, max_len)

    # Define the data loader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the transformer model
    model = CustomTransformer(input_dim, hidden_dim, output_dim, n_layers, n_heads, dropout_rate).to(device)

    print("The model info:", model)

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Train the model
    # train(model, optimizer, data_loader, device, epochs)

# %%
if __name__ == "__main__":
    main()

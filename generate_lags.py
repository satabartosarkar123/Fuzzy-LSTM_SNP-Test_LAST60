import json
import os
import copy

BASE_DIR = '/Users/satabarto/Research/FINAL LSTM SNP ALL LAST 60/Final'
PURE_LSTM_DIR = os.path.join(BASE_DIR, 'Pure_LSTM')

# Datasets and their pure LSTM base notebook names
DATASETS = [
    ('dow_jones', 'Pure_LSTM_dow_jones - test_60.ipynb'),
    ('lake_erie', 'Pure_LSTM_lake_erie - test_60.ipynb'),
    ('milk_production', 'Pure_LSTM_milk_production - test_60.ipynb'),
    ('sp500', 'Pure_LSTM_sp500 - test_60.ipynb')
]

LAGS = {
    'LSTM_SNP': [1],
    'BiLSTM_SNP': [1],
    'Attention_SNP': [1, 5, 10, 20, 30]
}

SNP_CELL_CODE = """class LSTMSNPCell(nn.Module):
    \"\"\"
    LSTM-SNP Cell: gates r, c, o (hard sigmoid) and generated spikes a (tanh).
    u(t) = r(t)*u(t-1) - c(t)*a(t)
    h(t) = o(t)*a(t)
    \"\"\"
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel = nn.Linear(input_size, hidden_size * 4, bias=False)
        self.recurrent_kernel = nn.Linear(hidden_size, hidden_size * 4, bias=False)
        self.bias = nn.Parameter(torch.zeros(hidden_size * 4))

        nn.init.xavier_uniform_(self.kernel.weight)
        nn.init.orthogonal_(self.recurrent_kernel.weight)
        
        # Initialize consumption gate bias to 1.0 (Eq requirement)
        with torch.no_grad():
            self.bias.data[hidden_size:2 * hidden_size] = 1.0

    def hard_sigmoid(self, x):
        return torch.clamp(0.2 * x + 0.5, 0.0, 1.0)

    def forward(self, x, u_tm1):
        z = self.kernel(x) + self.recurrent_kernel(u_tm1) + self.bias
        z0, z1, z2, z3 = z.chunk(4, dim=-1)

        r = self.hard_sigmoid(z0)   # reset
        c = self.hard_sigmoid(z1)   # consumption
        o = self.hard_sigmoid(z2)   # output/generation
        a = torch.tanh(z3)          # generated spikes

        u = r * u_tm1 - c * a
        h = o * a
        return h, u
"""

MODELS = {
    'LSTM_SNP': {
        'name': 'SequenceLSTMSNP',
        'code': f"""import torch
import torch.nn as nn
import torch.nn.functional as F

{SNP_CELL_CODE}

class SequenceLSTMSNP(nn.Module):
    \"\"\"
    LSTM-SNP for time series forecasting over a sequence.
    Stateful: maintains hidden state across sequence steps.
    \"\"\"
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = LSTMSNPCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.u = None  # state

    def reset_states(self, batch_size, device):
        self.u = torch.zeros(batch_size, self.hidden_dim, device=device)
        
    def detach_states(self):
        if self.u is not None:
            self.u = self.u.detach()

    def forward(self, x):
        if self.u is None or self.u.size(0) != x.size(0) or self.u.device != x.device:
            self.reset_states(x.size(0), x.device)
            
        seq_len = x.size(1)
        h = None
        for t in range(seq_len):
            h, self.u = self.cell(x[:, t, :], self.u)
            
        out = self.fc(h)
        return out

def build_model(input_dim, units):
    return SequenceLSTMSNP(input_dim=input_dim, hidden_dim=units, output_dim=1)
"""
    },
    'BiLSTM_SNP': {
        'name': 'SequenceBiLSTMSNP',
        'code': f"""import torch
import torch.nn as nn
import torch.nn.functional as F

{SNP_CELL_CODE}

class SequenceBiLSTMSNP(nn.Module):
    \"\"\"
    Bidirectional LSTM-SNP for time series forecasting over a sequence.
    \"\"\"
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell_fw = LSTMSNPCell(input_dim, hidden_dim)
        self.cell_bw = LSTMSNPCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.u_fw = None
        self.u_bw = None

    def reset_states(self, batch_size, device):
        self.u_fw = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.u_bw = torch.zeros(batch_size, self.hidden_dim, device=device)
        self.u = self.u_fw # For compatibility with training loop that resets self.u
        
    def detach_states(self):
        if self.u_fw is not None:
            self.u_fw = self.u_fw.detach()
        if self.u_bw is not None:
            self.u_bw = self.u_bw.detach()
        self.u = self.u_fw

    def forward(self, x):
        if self.u_fw is None or self.u_fw.size(0) != x.size(0) or self.u_fw.device != x.device:
            self.reset_states(x.size(0), x.device)
            
        seq_len = x.size(1)
        
        h_fw = None
        for t in range(seq_len):
            h_fw, self.u_fw = self.cell_fw(x[:, t, :], self.u_fw)
            
        h_bw = None
        for t in range(seq_len - 1, -1, -1):
            h_bw, self.u_bw = self.cell_bw(x[:, t, :], self.u_bw)
            
        h_concat = torch.cat([h_fw, h_bw], dim=-1)
        out = self.fc(h_concat)
        return out

def build_model(input_dim, units):
    return SequenceBiLSTMSNP(input_dim=input_dim, hidden_dim=units, output_dim=1)
"""
    },
    'Attention_SNP': {
        'name': 'SequenceAttentionLSTMSNP',
        'code': f"""import torch
import torch.nn as nn
import torch.nn.functional as F

{SNP_CELL_CODE}

class SequenceAttentionLSTMSNP(nn.Module):
    \"\"\"
    LSTM-SNP with Temporal Attention for time series forecasting over a sequence.
    \"\"\"
    def __init__(self, input_dim=1, hidden_dim=8, output_dim=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = LSTMSNPCell(input_dim, hidden_dim)
        self.attention = nn.Linear(hidden_dim, 1)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.u = None  # state

    def reset_states(self, batch_size, device):
        self.u = torch.zeros(batch_size, self.hidden_dim, device=device)
        
    def detach_states(self):
        if self.u is not None:
            self.u = self.u.detach()

    def forward(self, x):
        if self.u is None or self.u.size(0) != x.size(0) or self.u.device != x.device:
            self.reset_states(x.size(0), x.device)
            
        seq_len = x.size(1)
        h_states = []
        for t in range(seq_len):
            h, self.u = self.cell(x[:, t, :], self.u)
            h_states.append(h.unsqueeze(1))
            
        lstm_out = torch.cat(h_states, dim=1) # (batch, seq_len, hidden_dim)
        
        # Self attention over sequence
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1) # (batch, seq_len, 1)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1) # (batch, hidden_dim)
        
        out = self.fc(context_vector)
        return out

def build_model(input_dim, units):
    return SequenceAttentionLSTMSNP(input_dim=input_dim, hidden_dim=units, output_dim=1)
"""
    }
}


def modify_cell(cell, replacements):
    # Returns true if modified
    modified = False
    new_source = []
    for line in cell.get('source', []):
        new_line = line
        for k, v in replacements.items():
            if k in new_line:
                new_line = new_line.replace(k, v)
                modified = True
        new_source.append(new_line)
    if modified:
        cell['source'] = new_source
    return modified


for model_name, model_info in MODELS.items():
    for lag in LAGS[model_name]:
        out_dir = os.path.join(BASE_DIR, f"{model_name}", f"lag_{lag}")
        os.makedirs(out_dir, exist_ok=True)
        
        for ds_name, nb_name in DATASETS:
            in_path = os.path.join(PURE_LSTM_DIR, nb_name)
            with open(in_path, 'r', encoding='utf-8') as f:
                nb = json.load(f)
            
            # Find the pure LSTM cell (the one containing 'class PureLSTM(nn.Module):')
            for cell in nb['cells']:
                if cell.get('cell_type') == 'code':
                    src = "".join(cell.get('source', []))
                    if "class PureLSTM(nn.Module):" in src:
                        # Replace the entire cell source with the new model code
                        cell['source'] = [line + "\n" for line in model_info['code'].split("\n")][:-1]
            
            # Replacements
            replacements = {
                'def timeseries_to_supervised(data, lag=1):': f'def timeseries_to_supervised(data, lag={lag}):',
                'supervised = timeseries_to_supervised(diff_values, 1)': f'supervised = timeseries_to_supervised(diff_values, {lag})',
                'X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))': 'X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))',
                'view(1, 1, len(X_raw))': 'view(1, len(X_raw), 1)',
                'view(1, 1, len(X))': 'view(1, len(X), 1)',
                'build_model(input_dim=1, units=8)': 'build_model(input_dim=1, units=8)', # Keep as 1 since our feature dimension is 1
                'Pure_LSTM\\\\checkpoints_': f'{model_name}\\\\lag_{lag}\\\\checkpoints_',
                'PureLSTM': model_info['name'],
                '# Pure LSTM': f'# {model_name} (Lag {lag})',
                'Pure LSTM': f'{model_name}',
                'NOISE_LEVELS = [0.005, 0.05, 0.10, 0.15]  # 0.5%, 5%, 10%, 15%': 'NOISE_LEVELS = [0.005, 0.05, 0.10]  # 0.5%, 5%, 10%',
                'model.u = model.u.detach()': 'model.detach_states()',
                'model.u = model.detach_states() or model.u  # see note below': 'model.detach_states()',
                'This notebook implements a **standard LSTM** (Hochreiter & Schmidhuber, 1997) for time series': f'This notebook implements a **{model_name}** cell utilizing Spiking Neural P system components for time series',
                'forecasting. Uses identical hyperparameters and training protocol as LSTM-SNP for fair comparison.': 'forecasting. Uses hard-sigmoid gates and a custom `LSTMSNPCell` underlying architecture.'
            }
            
            for cell in nb['cells']:
                modify_cell(cell, replacements)
                # clear outputs
                if cell.get('cell_type') == 'code':
                    cell['outputs'] = []
                    cell['execution_count'] = None
                    
            out_nb_name = f"{model_name}_lag{lag}_{ds_name}.ipynb"
            out_path = os.path.join(out_dir, out_nb_name)
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
            print(f"Generated {out_path}")

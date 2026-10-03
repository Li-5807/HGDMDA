import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, accuracy_score, recall_score, precision_score)
from sklearn.metrics.pairwise import cosine_similarity
import random
import os
import copy



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def graph_diffusion(A, alpha=0.10, num_iter=30, epsilon=1e-6):
    A = np.array(A, dtype=np.float64)
    assert A.shape[0] == A.shape[1], \
        f"graph_diffusion 只接受方阵，当前形状 {A.shape}"

    D_row = np.diag(np.sum(A, axis=1) + epsilon)
    D_col = np.diag(np.sum(A, axis=0) + epsilon)
    D_row_inv_sqrt = np.linalg.inv(np.sqrt(D_row))
    D_col_inv_sqrt = np.linalg.inv(np.sqrt(D_col))

    T = D_row_inv_sqrt @ A @ D_col_inv_sqrt
    TT = T @ T.T

    A_diff = np.zeros_like(A)
    M = T.copy()
    for i in range(num_iter):
        w = alpha * (1 - alpha) ** i
        A_diff += w * M
        M = TT @ M

    return A_diff

def normalized_laplacian(A):
    A = np.array(A, dtype=np.float64)
    assert A.shape[0] == A.shape[1], \
        f"normalized_laplacian 只接受方阵，当前形状 {A.shape}"
    D = np.diag(np.sum(A, axis=1) + 1e-6)
    D_inv_sqrt = np.linalg.inv(np.sqrt(D))
    L = D_inv_sqrt @ A @ D_inv_sqrt
    return L



def build_joint_block(A_dd, A_mm, A_dm, n_drugs, n_mutations):
    n_total = n_drugs + n_mutations
    A_full = np.zeros((n_total, n_total), dtype=np.float64)

    A_full[:n_drugs, :n_drugs] = A_dd
    A_full[n_drugs:, n_drugs:] = A_mm
    A_full[:n_drugs, n_drugs:] = A_dm
    A_full[n_drugs:, :n_drugs] = A_dm.T

    return A_full



class HGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_relations=3, dropout=0.3):
        super().__init__()
        self.num_relations = num_relations
        self.weights = nn.ModuleList([
            nn.Linear(in_dim, out_dim, bias=False) for _ in range(num_relations)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, L_list):
        X_relations = []
        for i in range(self.num_relations):
            X_conv = torch.mm(L_list[i], X)
            X_conv = self.weights[i](X_conv)
            X_conv = self.dropout(F.relu(X_conv))
            X_relations.append(X_conv)
        return X_relations


class AttentionAggregation(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim // 4),
            nn.Tanh(),
            nn.Linear(hidden_dim // 4, 1, bias=False)
        )

    def forward(self, X_relations):
        attention_scores = [self.mlp(X_r) for X_r in X_relations]
        attention_scores = torch.stack(attention_scores, dim=0)
        attention_weights = F.softmax(attention_scores, dim=0)
        X_aggregated = sum(w * x for w, x in zip(attention_weights, X_relations))
        return X_aggregated


class MGDModel(nn.Module):
    def __init__(self, num_drugs, num_mutations, hidden_dim=96,
                 num_classes=2, dropout=0.3, device='cpu'):
        super().__init__()
        self.num_drugs = num_drugs
        self.num_mutations = num_mutations
        self.device = device
        self.dropout = nn.Dropout(dropout)

        self.drug_embed = nn.Embedding(num_drugs, hidden_dim)
        self.mutation_embed = nn.Embedding(num_mutations, hidden_dim)

        self.hgcn1 = HGCNLayer(hidden_dim, hidden_dim, dropout=dropout)
        self.hgcn2 = HGCNLayer(hidden_dim, hidden_dim, dropout=dropout)

        self.attention1 = AttentionAggregation(hidden_dim, hidden_dim)
        self.attention2 = AttentionAggregation(hidden_dim, hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, drug_idx, mutation_idx, L_dd, L_mm, L_dm):
        X_d = self.drug_embed(torch.arange(self.num_drugs, device=self.device))
        X_m = self.mutation_embed(torch.arange(self.num_mutations, device=self.device))
        X = torch.cat([X_d, X_m], dim=0)

        L_list = [L_dd, L_mm, L_dm]

        X_relations1 = self.hgcn1(X, L_list)
        X1 = self.attention1(X_relations1)

        X_relations2 = self.hgcn2(X1, L_list)
        X2 = self.attention2(X_relations2)

        drug_feats = X2[drug_idx]
        mutation_feats = X2[mutation_idx + self.num_drugs]
        pair_feats = torch.cat([drug_feats, mutation_feats], dim=1)
        return self.classifier(pair_feats)



def evaluate(model, data):
    model.eval()
    with torch.no_grad():
        outputs = model(data['drug_idx'], data['mutation_idx'],
                        data['L_dd'], data['L_mm'], data['L_dm'])
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        y_true = data['labels'].cpu().numpy()
        auc = roc_auc_score(y_true, probs)
        aupr = average_precision_score(y_true, probs)
    return auc, aupr


def train_and_evaluate(model, train_data, val_data, test_data,
                       optimizer, criterion, epochs=120, patience=20):

    best_val_auc = -np.inf
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_data['drug_idx'], train_data['mutation_idx'],
                        train_data['L_dd'], train_data['L_mm'], train_data['L_dm'])
        loss = criterion(outputs, train_data['labels'])
        loss.backward()
        optimizer.step()

        val_auc, val_aupr = evaluate(model, val_data)


        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stopping at epoch {epoch+1}, best val AUC={best_val_auc:.4f}")
                break


    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        test_outputs = model(test_data['drug_idx'], test_data['mutation_idx'],
                             test_data['L_dd'], test_data['L_mm'], test_data['L_dm'])
        probs = F.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
        preds = test_outputs.argmax(dim=1).cpu().numpy()
        y_true = test_data['labels'].cpu().numpy()

        metrics = {
            'AUC': roc_auc_score(y_true, probs),
            'AUPR': average_precision_score(y_true, probs),
            'F1': f1_score(y_true, preds),
            'Accuracy': accuracy_score(y_true, preds),
            'Recall': recall_score(y_true, preds),
            'Precision': precision_score(y_true, preds, zero_division=0)
        }
    return metrics, y_true.tolist(), probs.tolist()


def prepare_data(df, drug_to_idx, mutation_to_idx):
    drug_idx = torch.LongTensor([drug_to_idx[d] for d in df['Drug']])
    mutation_idx = torch.LongTensor([mutation_to_idx[m] - len(drug_to_idx)
                                     for m in df['Mutation']])
    labels = torch.tensor(((df['Association'].values + 1) // 2).astype(np.int64))
    return {
        'drug_idx': drug_idx,
        'mutation_idx': mutation_idx,
        'labels': labels
    }


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

data = pd.read_csv('association_2173_M1224_K949.csv')
drugs = sorted(data['Drug'].unique())
mutations = sorted(data['Mutation'].unique())

drug_to_idx = {d: i for i, d in enumerate(drugs)}
mutation_to_idx = {m: i + len(drugs) for i, m in enumerate(mutations)}
n_drugs = len(drugs)
n_mutations = len(mutations)
num_nodes = n_drugs + n_mutations

N_REPEATS = 100
N_FOLDS = 5
VAL_RATIO = 0.15     
PATIENCE = 20         

all_results = []
all_test_labels = []
all_test_probs = []

for repeat in range(1, N_REPEATS + 1):
    print(f"\n========== Repeat {repeat}/{N_REPEATS} ==========")
    seed = 42 + repeat * 10
    set_seed(seed)

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(kf.split(data), 1):
        print(f" Fold {fold}/{N_FOLDS}")

        train_df_full = data.iloc[train_idx]
        test_df = data.iloc[test_idx]

        train_df, val_df = train_test_split(
            train_df_full, test_size=VAL_RATIO, random_state=seed
        )

        A_dm = np.zeros((n_drugs, n_mutations), dtype=np.float64)
        for _, row in train_df.iterrows():
            d = drug_to_idx[row['Drug']]
            m = mutation_to_idx[row['Mutation']] - n_drugs
            A_dm[d, m] = 1.0 if row['Association'] > 0 else 0.0

        A_dm_for_sim = A_dm + 1e-6

        drug_sim = cosine_similarity(A_dm_for_sim)
        A_dd = (drug_sim > 0.6).astype(np.float64)
        np.fill_diagonal(A_dd, 1.0)

        mutation_sim = cosine_similarity(A_dm_for_sim.T)
        A_mm = (mutation_sim > 0.6).astype(np.float64)
        np.fill_diagonal(A_mm, 1.0)


        A_full = build_joint_block(A_dd, A_mm, A_dm, n_drugs, n_mutations)


        A_full_diff = graph_diffusion(A_full, alpha=0.10, num_iter=30)

        L_dd = normalized_laplacian(A_full_diff[:n_drugs, :n_drugs])
        L_mm = normalized_laplacian(A_full_diff[n_drugs:, n_drugs:])
        L_dm = normalized_laplacian(A_full_diff)

        L_dd_full = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        L_dd_full[:n_drugs, :n_drugs] = L_dd

        L_mm_full = np.zeros((num_nodes, num_nodes), dtype=np.float64)
        L_mm_full[n_drugs:, n_drugs:] = L_mm


        train_data = prepare_data(train_df, drug_to_idx, mutation_to_idx)
        val_data = prepare_data(val_df, drug_to_idx, mutation_to_idx)
        test_data = prepare_data(test_df, drug_to_idx, mutation_to_idx)

        L_dict = {
            'L_dd': torch.FloatTensor(L_dd_full).to(device),
            'L_mm': torch.FloatTensor(L_mm_full).to(device),
            'L_dm': torch.FloatTensor(L_dm).to(device)
        }
        train_data.update(L_dict)
        val_data.update(L_dict)
        test_data.update(L_dict)

        for k in ['drug_idx', 'mutation_idx', 'labels']:
            train_data[k] = train_data[k].to(device)
            val_data[k] = val_data[k].to(device)
            test_data[k] = test_data[k].to(device)

        model = MGDModel(n_drugs, n_mutations, device=device).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        metrics, fold_labels, fold_probs = train_and_evaluate(
            model, train_data, val_data, test_data,
            optimizer, criterion, epochs=120, patience=PATIENCE
        )

        metrics['repeat'] = repeat
        metrics['fold'] = fold
        all_results.append(metrics)

        all_test_labels.extend(fold_labels)
        all_test_probs.extend(fold_probs)

        print(f" → {metrics}")



print("\n" + "=" * 60)
print("All 500 experiments finished. Computing final statistics...")

df_all = pd.DataFrame(all_results)
metric_cols = ['AUC', 'AUPR', 'F1', 'Accuracy', 'Recall', 'Precision']

final_stats = pd.DataFrame({
    'Metric': metric_cols,
    'Mean': df_all[metric_cols].mean().values,
    'Std': df_all[metric_cols].std().values
})

print("\nFinal Results over 100 repeats × 5-fold (500 models):")
print(final_stats.round(4))

final_stats.to_csv('hgdmda_joint_diffusion_results.csv', index=False)
print("Results saved to 'hgdmda_joint_diffusion_results.csv'")

pred_df = pd.DataFrame({
    'true_label': all_test_labels,
    'pred_prob': all_test_probs
})
pred_df.to_csv('hgdmda_joint_diffusion_predictions_500folds.csv', index=False)
print("PR and ROC curve data saved to 'hgdmda_joint_diffusion_predictions_500folds.csv'")
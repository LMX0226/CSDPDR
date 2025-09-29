import copy
import timeit
import argparse
import warnings
import sys
import os

import torch
import torch.optim as optim
import pandas as pd
import dgl

from model import AEModel
from data import *

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='B-dataset')
    parser.add_argument('--random_seed', type=int, default=1234)
    parser.add_argument('--neighbor', type=int, default=5)
    parser.add_argument('--negative_rate', type=float, default=1.0)

    parser.add_argument('--epochs', type=int, default=800)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)

    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--num_out_heads', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--in_drop', type=float, default=0.2)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--norm', type=str, default=None)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--activation', type=str, default='prelu')

    args = parser.parse_args()
    args.data_dir = '../data/' + args.dataset + '/'
    args.GAE_data_dir = 'Embedding/' + args.dataset + '/'
    args.result_dir = 'Embedding/' + args.dataset + '/'
    os.makedirs(args.result_dir, exist_ok=True)

    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)

    data = get_data(args)
    args.drug_number = data['drug_number']
    args.disease_number = data['disease_number']
    args.protein_number = data['protein_number']

    data = data_processing(data, args)
    drug_graph, disease_graph, protein_graph, data = dgl_similarity_graph(data, args)

    drug_graph = drug_graph.to(device)
    disease_graph = disease_graph.to(device)
    protein_graph = protein_graph.to(device)

    drug_feature = torch.as_tensor(data['drs'], dtype=torch.float32, device=device)
    disease_feature = torch.as_tensor(data['dis'], dtype=torch.float32, device=device)
    protein_feature = torch.as_tensor(data['prs'], dtype=torch.float32, device=device)

    start = timeit.default_timer()
    print('Dataset:', args.dataset)

    entity = ['drug', 'disease', 'protein']

    for item in entity:
        entity_graph = eval(f'{item}_graph')
        entity_feature = eval(f'{item}_feature')
        entity_num = eval(f'args.{item}_number') 
        in_dim = entity_num

        model = AEModel(args, in_dim=in_dim).to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_loss = float('inf')
        best_epoch = 0
        best_model = None

        for epoch in range(args.epochs):
            model.train()
            loss, loss_dict, enc_rep, recon = model(entity_graph, entity_feature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            elapsed = timeit.default_timer() - start
            val = float(loss.item())
            print('\t'.join(map(str, [epoch + 1, round(elapsed, 2), round(val, 6)])))

            if val < best_loss:
                best_loss = val
                best_epoch = epoch + 1
                best_model = copy.deepcopy(model)

        print('Entity:', item, ';\tBest epoch:', best_epoch, ';\tBest loss:', best_loss)

        with torch.no_grad():
            rep = best_model.embed(entity_graph, entity_feature)
        rep = rep.detach().cpu().numpy()
        pd.DataFrame(rep).to_csv(os.path.join(args.result_dir, f'{item}.csv'), index=False)

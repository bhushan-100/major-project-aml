import torch
import pandas as pd
from train_util import AddEgoIds, extract_param, add_arange_ids, get_loaders, evaluate_model, evaluate_hetero, load_model
from training import get_model
from torch_geometric.nn import to_hetero, summary
import logging
import os
import sys
import time
from torch.utils.tensorboard import SummaryWriter

script_start = time.time()



def infer_gnn(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, args, data_config):
    #set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter()

    #define a model config dictionary and wandb logging at the same time
    config={
            "epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "model": args.model,
            "data": args.data,
            "num_neighbors": args.num_neighs,
            "lr": extract_param("lr", args),
            "n_hidden": extract_param("n_hidden", args),
            "n_gnn_layers": extract_param("n_gnn_layers", args),
            "loss": "ce",
            "w_ce1": extract_param("w_ce1", args),
            "w_ce2": extract_param("w_ce2", args),
            "dropout": extract_param("dropout", args),
            "final_dropout": extract_param("final_dropout", args),
            "n_heads": extract_param("n_heads", args) if args.model == 'gat' else None
    }

    #set the transform if ego ids should be used
    if args.ego:
        transform = AddEgoIds()
    else:
        transform = None

    #add the unique ids to later find the seed edges
    add_arange_ids([tr_data, val_data, te_data])
    # save_edge_ids_to_csv([tr_data, val_data, te_data], 'edge_ids.csv')

    tr_loader, val_loader, te_loader = get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args)

    #get the model
    sample_batch = next(iter(tr_loader))
    model = get_model(sample_batch, config, args)

    if args.reverse_mp:
        model = to_hetero(model, te_data.metadata(), aggr='mean')

    if args.load_ckpt:
        model, optimizer, ckpt_epochs, ckpt_best_f1 = load_model(model, device, args, config, data_config)
    else:
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    model.to(device)

    if not args.reverse_mp:
        te_f1, te_prec, te_rec = evaluate_homo(te_loader, te_inds, model, te_data, device, args, precrec=True)
    else:
        te_f1, te_prec, te_rec = evaluate_hetero(te_loader, te_inds, model, te_data, device, args, precrec=True)

    print(f"F1: {te_f1*100:.2f}")
    print(f"Precision: {te_prec*100:.2f}")
    print(f"Recall: {te_rec*100:.2f}")
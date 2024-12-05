import torch
import tqdm
from torch_geometric.transforms import BaseTransform
from typing import Union
from torch_geometric.data import Data, HeteroData
from torch_geometric.loader import LinkNeighborLoader
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
import json
import os
import pandas as pd
import csv



def save_edge_data_to_csv(data, prefix, output_dir='edge_data'):
    """
    Save edge data to CSV files.
    
    Args:
        data: PyG Data or HeteroData object
        prefix: String prefix for the filename (e.g., 'tr', 'val', 'te')
        output_dir: Directory to save the CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if isinstance(data, HeteroData):
        for edge_type in [('node', 'to', 'node'), ('node', 'rev_to', 'node')]:
            edge_attr = data[edge_type].edge_attr
            edge_index = data[edge_type].edge_index
            
            df = pd.DataFrame(
                edge_attr.numpy(),
                columns=['edge_id'] + [f'feature_{i}' for i in range(edge_attr.shape[1]-1)]
            )
            df['source'] = edge_index[0].numpy()
            df['target'] = edge_index[1].numpy()
            
            filename = f'{output_dir}/{prefix}_{edge_type[1]}.csv'
            df.to_csv(filename, index=False)
    else:
        edge_attr = data.edge_attr
        edge_index = data.edge_index
        
        df = pd.DataFrame(
            edge_attr.numpy(),
            columns=['edge_id'] + [f'feature_{i}' for i in range(edge_attr.shape[1]-1)]
        )
        df['source'] = edge_index[0].numpy()
        df['target'] = edge_index[1].numpy()
        
        filename = f'{output_dir}/{prefix}.csv'
        df.to_csv(filename, index=False)

class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
            
        return data

def extract_param(parameter_name: str, args) -> float:
    """
    Extract the value of the specified parameter for the given model.
    
    Args:
    - parameter_name (str): Name of the parameter (e.g., "lr").
    - args (argparser): Arguments given to this specific run.
    
    Returns:
    - float: Value of the specified parameter.
    """
    file_path = './model_settings.json'
    with open(file_path, "r") as file:
        data = json.load(file)

    return data.get(args.model, {}).get("params", {}).get(parameter_name, None)

def add_arange_ids(data_list):
    '''
    Add the index as an id to the edge features to find seed edges in training, validation and testing.

    Args:
    - data_list (str): List of tr_data, val_data and te_data.
    '''
    for i, data in enumerate(data_list):
        prefix = ['tr', 'val', 'te'][i]
        save_edge_data_to_csv(data, f"{prefix}_before", 'edge_data_before')
        
    for data in data_list:
        if isinstance(data, HeteroData):
            data['node', 'to', 'node'].edge_attr = torch.cat([torch.arange(data['node', 'to', 'node'].edge_attr.shape[0]).view(-1, 1), data['node', 'to', 'node'].edge_attr], dim=1)
            offset = data['node', 'to', 'node'].edge_attr.shape[0]
            data['node', 'rev_to', 'node'].edge_attr = torch.cat([torch.arange(offset, data['node', 'rev_to', 'node'].edge_attr.shape[0] + offset).view(-1, 1), data['node', 'rev_to', 'node'].edge_attr], dim=1)
        else:
            data.edge_attr = torch.cat([torch.arange(data.edge_attr.shape[0]).view(-1, 1), data.edge_attr], dim=1)
            
    for i, data in enumerate(data_list):
        prefix = ['tr', 'val', 'te'][i]
        save_edge_data_to_csv(data, f"{prefix}_after", 'edge_data_after')

def get_loaders(tr_data, val_data, te_data, tr_inds, val_inds, te_inds, transform, args):
    if isinstance(tr_data, HeteroData):
        tr_edge_label_index = tr_data['node', 'to', 'node'].edge_index
        tr_edge_label = tr_data['node', 'to', 'node'].y


        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), tr_edge_label_index), 
                                    edge_label=tr_edge_label, batch_size=args.batch_size, shuffle=True, transform=transform)
        
        val_edge_label_index = val_data['node', 'to', 'node'].edge_index[:,val_inds]
        val_edge_label = val_data['node', 'to', 'node'].y[val_inds]


        val_loader =  LinkNeighborLoader(val_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), val_edge_label_index), 
                                    edge_label=val_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform)
        
        te_edge_label_index = te_data['node', 'to', 'node'].edge_index[:,te_inds]
        te_edge_label = te_data['node', 'to', 'node'].y[te_inds]


        te_loader =  LinkNeighborLoader(te_data, num_neighbors=args.num_neighs, 
                                    edge_label_index=(('node', 'to', 'node'), te_edge_label_index), 
                                    edge_label=te_edge_label, batch_size=args.batch_size, shuffle=False, transform=transform)
    else:
        tr_loader =  LinkNeighborLoader(tr_data, num_neighbors=args.num_neighs, batch_size=args.batch_size, shuffle=True, transform=transform)
        val_loader = LinkNeighborLoader(val_data,num_neighbors=args.num_neighs, edge_label_index=val_data.edge_index[:, val_inds],
                                        edge_label=val_data.y[val_inds], batch_size=args.batch_size, shuffle=False, transform=transform)
        te_loader =  LinkNeighborLoader(te_data,num_neighbors=args.num_neighs, edge_label_index=te_data.edge_index[:, te_inds],
                                edge_label=te_data.y[te_inds], batch_size=args.batch_size, shuffle=False, transform=transform)
        
    return tr_loader, val_loader, te_loader

@torch.no_grad()
def evaluate_homo(loader, inds, model, data, device, args):
    '''Evaluates the model performane for homogenous graph data.'''
    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #add the seed edges that have not been sampled to the batch
        missing = ~torch.isin(batch_edge_ids, batch.edge_attr[:, 0].detach().cpu())

        if missing.sum() != 0 and (args.data == 'Small_J' or args.data == 'Small_Q'):
            missing_ids = batch_edge_ids[missing].int()
            n_ids = batch.n_id
            add_edge_index = data.edge_index[:, missing_ids].detach().clone()
            node_mapping = {value.item(): idx for idx, value in enumerate(n_ids)}
            add_edge_index = torch.tensor([[node_mapping[val.item()] for val in row] for row in add_edge_index])
            add_edge_attr = data.edge_attr[missing_ids, :].detach().clone()
            add_y = data.y[missing_ids].detach().clone()
        
            batch.edge_index = torch.cat((batch.edge_index, add_edge_index), 1)
            batch.edge_attr = torch.cat((batch.edge_attr, add_edge_attr), 0)
            batch.y = torch.cat((batch.y, add_y), 0)

            mask = torch.cat((mask, torch.ones(add_y.shape[0], dtype=torch.bool)))

        #remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = out[mask]
            pred = out.argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch.y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)

    return f1

@torch.no_grad()
def evaluate_hetero(loader, inds, model, data, device, args, precrec=False):
    '''Evaluates the model performane for heterogenous graph data.'''
    preds = []
    ground_truths = []
    csv_data = []
    
    i = 0
    
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        i += 1
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch['node', 'to', 'node'].input_id.detach().cpu()]
        batch_edge_ids = loader.data['node', 'to', 'node'].edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch['node', 'to', 'node'].edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #remove the unique edge id from the edge features, as it's no longer needed
        edge_ids = batch['node', 'to', 'node'].edge_attr[mask, 0].detach().cpu().numpy()
        edge_attrs = batch['node', 'to', 'node'].edge_attr[mask].detach().cpu().numpy() 
        batch['node', 'to', 'node'].edge_attr = batch['node', 'to', 'node'].edge_attr[:, 1:]
        batch['node', 'rev_to', 'node'].edge_attr = batch['node', 'rev_to', 'node'].edge_attr[:, 1:]
        
        with torch.no_grad():
            batch.to(device)
            out = model(batch.x_dict, batch.edge_index_dict, batch.edge_attr_dict)
            out = out[('node', 'to', 'node')]
            out = out[mask]
            # pred = out.argmax(dim=-1)
            probs = torch.softmax(out, dim=-1)

            pred = (probs[:, 1] > 0.9).long()
            preds.append(pred)
            ground_truths.append(batch['node', 'to', 'node'].y[mask])
            
            edge_list = batch['node', 'to', 'node'].edge_index[:, mask].detach().cpu().numpy().T
            truth_list = batch['node', 'to', 'node'].y[mask].detach().cpu().numpy()
            pred_list = pred.cpu().numpy()

            csv_data.extend([
                [edge_id, edge[0], edge[1], truth, pred, i]
                for edge_id, edge, truth, pred in zip(edge_ids, edge_list, truth_list, pred_list)
            ])
            
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    with open("evaluation_results.csv", "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Edge ID", "Source Node", "Target Node", "Ground Truth", "Prediction", "batch"])
        csv_writer.writerows(csv_data)
        
    f1 = f1_score(ground_truth, pred)
    acc = accuracy_score(ground_truth, pred)
    prec = precision_score(ground_truth, pred)
    rec = recall_score(ground_truth, pred)

    return f1, prec, rec

# @torch.no_grad()
def evaluate_model(loader, inds, model, data, device, args):
    '''Evaluates the model performane for homogenous graph data.'''
    preds = []
    ground_truths = []
    for batch in tqdm.tqdm(loader, disable=not args.tqdm):
        #select the seed edges from which the batch was created
        inds = inds.detach().cpu()
        batch_edge_inds = inds[batch.input_id.detach().cpu()]
        batch_edge_ids = loader.data.edge_attr.detach().cpu()[batch_edge_inds, 0]
        mask = torch.isin(batch.edge_attr[:, 0].detach().cpu(), batch_edge_ids)

        #remove the unique edge id from the edge features, as it's no longer needed
        batch.edge_attr = batch.edge_attr[:, 1:]

        with torch.no_grad():
            batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            out = out[mask]
            pred = out.argmax(dim=-1)
            preds.append(pred)
            ground_truths.append(batch.y[mask])
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    f1 = f1_score(ground_truth, pred)
    acc = accuracy_score(ground_truth, pred)

    return f1, acc

def save_model(model, optimizer, epoch, best_f1, args, data_config):
    # Save the model in a dictionary
    torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1
                }, f'./{data_config["paths"]["model_to_save"]}/checkpoint_{args.unique_name}{"" if not args.finetune else "_finetuned"}.tar')

def load_model(model, device, args, config, data_config):
    checkpoint_path = f'.{data_config["paths"]["model_to_load"]}/checkpoint_{args.unique_name}.tar'

    if not os.path.exists(checkpoint_path):
        # If no checkpoint found, return the original model, optimizer, and epoch
        print(f"No checkpoint found for {args.unique_name}, starting training from 0...\n\n")
        return model, torch.optim.Adam(model.parameters(), lr=config["lr"]), 0, 0

    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint["epoch"]
    best_f1 = checkpoint["best_f1"] if "best_f1" in checkpoint else 0
    print(f"\n\nCheckpoint found for {args.unique_name}, epoch: {epoch+1}\n\n")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer, epoch, best_f1
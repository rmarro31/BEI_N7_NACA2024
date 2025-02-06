import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
import os
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay
from tqdm import tqdm
from sklearn.metrics import r2_score
import random
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

# === 📌 1. Calcul des min/max globaux ===
def compute_global_min_max(data_dir):
    all_files = sorted(os.listdir(data_dir))
    p_min, p_max = float("inf"), float("-inf")
    
    for file in all_files:
        df = pd.read_csv(os.path.join(data_dir, file), skiprows=1, sep=',', skipinitialspace=True,
                         names=["id", "x", "y", "pressure", "volume"])
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
        df = df.dropna()
        if df.empty:
            continue
        p_min, p_max = min(p_min, df["pressure"].min()), max(p_max, df["pressure"].max())
    
    return p_min, p_max

# === 📌 2. Charger un fichier et appliquer Min-Max global ===
def load_fluent_data(file_path, p_min, p_max):
    df = pd.read_csv(file_path, skiprows=1, sep=',', skipinitialspace=True,
                     names=["id", "x", "y", "pressure", "volume"])
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
    df = df.dropna()
    if df.empty:
        raise ValueError(f"Erreur : les données de {file_path} sont vides après nettoyage !")

    pressures = (df["pressure"].values - p_min) / (p_max - p_min)
    x_features = np.vstack((df["x"].values, df["y"].values)).T
    points = np.vstack((df["x"].values, df["y"].values)).T
    tri = Delaunay(points)
    edges = [(simplex[i], simplex[j]) for simplex in tri.simplices for i in range(3) for j in range(i + 1, 3)]
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    
    graph = Data(x=torch.tensor(x_features, dtype=torch.float),
                 edge_index=edge_index,
                 y=torch.tensor(pressures, dtype=torch.float))
    return graph

# === 📌 3. Chargement du dataset ===
def load_dataset(data_dir, num_train, num_test):
    p_min, p_max = compute_global_min_max(data_dir)
    all_files = sorted(os.listdir(data_dir))
    selected_files = np.random.choice(all_files, num_train + num_test, replace=False)
    graphs = [load_fluent_data(os.path.join(data_dir, file), p_min, p_max) for file in selected_files]
    return graphs[:num_train], graphs[num_train:], (p_min, p_max)

# === 📌 4. Définition du GNN ===
class PressureGNN(nn.Module):
    def __init__(self):
        super(PressureGNN, self).__init__()
        self.layers = nn.ModuleList()
    
    def add_layer(self, layer_type, in_channels, out_channels, activation=None, batch_norm=False, dropout=0.0):
        layers = [layer_type(in_channels, out_channels)]
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        if activation:
            layers.append(activation())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.layers.append(nn.Sequential(*layers))
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
            if isinstance(layer[0], GCNConv):
                x = layer[0](x, edge_index)
                x = layer[1:](x)
            else:
                x = layer(x)
        return x  # 🚀 Suppression de l'activation finale !

    # === 📌 5. Entraînement et évaluation ===
def train_gnn(model, train_loader, test_loader, optimizer, loss_fn, epochs, scheduler):
    train_losses, test_r2_scores = [], []
    a = 0
    for epoch in tqdm(range(epochs), desc="Entraînement du modèle"):
        model.train()
        total_loss = 0

        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out.squeeze(), batch.y)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Ajuster le taux d'apprentissage
        scheduler.step(avg_loss)

        test_r2 = evaluate_gnn(model, test_loader)
        test_r2_scores.append(test_r2)

        if a == 1:
            tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Test R²: {test_r2:.4f}")
            a = 0
        else:
            a += 1

    return train_losses, test_r2_scores

# === 📌 5. Correction de la dénormalisation ===
def plot_test_case(graph, predictions, p_min, p_max, title="Comparaison Prédiction vs Réel"):
    predictions_denorm = predictions * (p_max - p_min) + p_min
    targets_denorm = graph.y * (p_max - p_min) + p_min
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    sc1 = axes[0].scatter(graph.x[:, 0].numpy(), graph.x[:, 1].numpy(), c=targets_denorm.numpy(), cmap="viridis", s=10)
    axes[0].set_title("Pression réelle")
    fig.colorbar(sc1, ax=axes[0])
    sc2 = axes[1].scatter(graph.x[:, 0].numpy(), graph.x[:, 1].numpy(), c=predictions_denorm.numpy(), cmap="viridis", s=10)
    axes[1].set_title("Pression prédite")
    fig.colorbar(sc2, ax=axes[1])
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

def evaluate_gnn(model, data_loader):
    model.eval()
    all_targets, all_preds = [], []
    with torch.no_grad():
        for batch in data_loader:
            preds = model(batch).squeeze()
            all_preds.append(preds)
            all_targets.append(batch.y)
    return r2_score(torch.cat(all_targets).numpy(), torch.cat(all_preds).numpy())

# === 📌 6. Main ===
if __name__ == "__main__":
    data_dir = "./Premier_resu/Test"
    num_train_graphs, num_test_graphs = 359, 40
    epochs = 40

    train_graphs, test_graphs, p_range = load_dataset(data_dir, num_train_graphs, num_test_graphs)
    train_loader = DataLoader(train_graphs, batch_size=10, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=10, shuffle=False)
    
    model = PressureGNN()
    # Ajout des couches avec options avancées
    model.add_layer(GCNConv, 2, 64, activation=nn.ReLU, batch_norm=True, dropout=0.2)
    model.add_layer(GCNConv, 64, 64, activation=nn.ReLU, batch_norm=True, dropout=0.3)
    model.add_layer(GCNConv, 64, 64, activation=nn.ReLU, batch_norm=True, dropout=0.4)
    model.add_layer(GCNConv, 64, 64, activation=nn.ReLU, batch_norm=True, dropout=0.4)
    model.add_layer(GCNConv, 64, 64, activation=nn.ReLU, batch_norm=True, dropout=0.4)
    model.add_layer(GCNConv, 64, 32, activation=nn.ReLU, batch_norm=True, dropout=0.4)
    model.add_layer(GCNConv, 32, 16, activation=nn.ReLU, batch_norm=True, dropout=0.3)
    model.add_layer(GCNConv, 16, 16, activation=nn.ReLU, batch_norm=True, dropout=0.3)

    # Ajout de couches denses
    model.add_layer(nn.Flatten, 0, 0)
    model.add_layer(nn.Linear, 16, 16, activation=nn.ReLU, batch_norm=True, dropout=0.3)
    model.add_layer(nn.Linear, 16, 16, activation=nn.ReLU, batch_norm=True, dropout=0.3)
    model.add_layer(nn.Linear, 16, 16, activation=nn.ReLU, batch_norm=True, dropout=0.3)
    model.add_layer(nn.Linear, 16, 8, activation=nn.ReLU, batch_norm=True, dropout=0.2)
    model.add_layer(nn.Linear, 8, 8, activation=nn.ReLU, batch_norm=True, dropout=0.2)
    model.add_layer(nn.Linear, 8, 4, activation=nn.ReLU, batch_norm=True, dropout=0.2)
    model.add_layer(nn.Linear, 4, 2, activation=nn.ReLU, batch_norm=True, dropout=0.1)
    model.add_layer(nn.Linear, 2, 1, activation=nn.Sigmoid, batch_norm=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')  # Ajuster le taux d'apprentissage
    loss_fn = nn.MSELoss()
    train_losses, test_r2_scores = train_gnn(model, train_loader, test_loader, optimizer, loss_fn, epochs, scheduler)

    test_r2 = evaluate_gnn(model, test_loader)
    print(f"R² sur les données test : {test_r2:.4f}")
    # Affichage des courbes de perte et R²
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Perte d'entraînement")
    plt.plot(test_r2_scores, label="R² sur test")
    plt.xlabel("Épochs")
    plt.ylabel("Valeur")
    plt.title("Évolution de la perte et du R²")
    plt.legend()
    plt.grid()
    plt.show()

    test_graph = test_graphs[0]
    model.eval()
    with torch.no_grad():
        predictions = model(test_graph)
    p_min, p_max = p_range
    plot_test_case(test_graph, predictions.squeeze(), p_min, p_max)

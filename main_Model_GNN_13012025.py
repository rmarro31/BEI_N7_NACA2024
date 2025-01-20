## Importé le 06/01/2025

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score
import networkx as nx
from scipy.spatial import Delaunay
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter


class PressureGraph():
    def __init__(self, P0, x_c, y_c, a, b , epsilon, k, x_min=0, x_max=10, y_min=0, y_max=10):
        # --- Champ de pression ---
        # x_c, y_c = 5, 5  # Centre de l'aile dans le plan (10x10)
        # a, b = 5, 3      # Demi-axes de l'ellipse
        # P_0 = 200        # Pression maximale
        # epsilon = 10     # Amplitude de la perturbation
        # k = 2 * np.pi / 10  # Fréquence de la perturbation
        
        
        #self.n_points = n_points
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.P0 = P0
        self.x_c = x_c
        self.y_c = y_c
        self.a = a
        self.b = b
        self.epsilon = epsilon
        self.k = k
        self.forme_aile = None
        
        
    def calculate_pressure(self, x, y):
        return self.P0 * np.exp(-(((x - self.x_c) ** 2) / self.a ** 2 + ((y - self.y_c) ** 2) / self.b ** 2)) + self.epsilon * np.sin(self.k * y)
    
    # mettre une fonction qui génénère une forme : un certcle, une ellipse...
    def generate_foil(self, num_points):
        #pour l'entrainement sans données, générer une forme a qui représene une aile
        theta = np.linspace(0, 2 * np.pi, num_points)
        x_foil = self.x_c + self.a * np.cos(theta)
        y_foil = self.y_c + self.b * np.sin(theta)
        self.forme_aile = np.vstack((x_foil, y_foil)).T
    
    def generate_graph(self, num_points, x_min=0, x_max=10, y_min=0, y_max=10):
        #à faire :
        # ne pas mettre des points dans la forme
        # ajouter un raffinement autour de la forme
        
        
        G = nx.Graph()
        # nodes_x = np.random.uniform(x_min, x_max, num_points)
        # nodes_y = np.random.uniform(y_min, y_max, num_points)
        
        nodes_x = np.random.uniform(x_min, x_max, num_points)
        nodes_y = np.random.uniform(y_min, y_max, num_points)
        
        # Densifier les points autour de l'aile
        if self.forme_aile is not None:
            num_points_foil = int(num_points * 0.3)  # 30% des points autour de l'aile
            theta = np.linspace(0, 2 * np.pi, num_points_foil)
            x_foil = self.x_c + (self.a * 0.5) * np.cos(theta) + np.random.normal(0, 0.1, num_points_foil)
            y_foil = self.y_c + (self.b * 0.5) * np.sin(theta) + np.random.normal(0, 0.1, num_points_foil)
            nodes_x = np.concatenate((nodes_x, x_foil))
            nodes_y = np.concatenate((nodes_y, y_foil))
    
        
        pressures = [self.calculate_pressure(nodes_x[i], nodes_y[i]) for i in range(num_points)]
        
        # Normaliser les données
        x_coords = np.vstack((nodes_x, nodes_y)).T
        x_normalized, pressures_normalized, x_min_vals, x_max_vals, y_min_val, y_max_val = normalize_data(x_coords, np.array(pressures))
    
        
        
        

        
        
        
        # Création des nœuds
        for i in range(num_points):
            G.add_node(i, pos=(x_normalized[i][0], x_normalized[i][1]), pressure=pressures_normalized[i])
            
        # Création des arêtes via triangulation
        tri = Delaunay(x_coords)
        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i + 1, 3):
                    G.add_edge(simplex[i], simplex[j])
                    
        # Retourner les données normalisées et les paramètres de normalisation
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        x_tensor = torch.tensor(x_normalized, dtype=torch.float)
        y_tensor = torch.tensor(pressures_normalized, dtype=torch.float)
        
        return (Data(x=x_tensor, edge_index=edge_index, y=y_tensor), (x_min_vals, x_max_vals, y_min_val, y_max_val))

    def generate_graphs(self, num_train_graphs, num_test_graphs, num_points_train, num_points_test):
        train_graphs, test_graphs = [], []
        train_params, test_params = [], []
        for _ in range(num_train_graphs):
            graph, params = self.generate_graph(num_points_train)
            train_graphs.append(graph)
            train_params.append(params)
        for _ in range(num_test_graphs):
            graph, params = self.generate_graph(num_points_test)
            test_graphs.append(graph)
            test_params.append(params)
            
        return train_graphs, test_graphs, train_params, test_params





# --- Normalisation ---
def normalize_data(x, y):
    x_min, x_max = x.min(axis=0), x.max(axis=0)
    y_min, y_max = y.min(), y.max()
    x_normalized = (x - x_min) / (x_max - x_min)
    y_normalized = (y - y_min) / (y_max - y_min)
    return x_normalized, y_normalized, x_min, x_max, y_min, y_max

def denormalize_data(y_normalized, y_min, y_max):
    return y_normalized * (y_max - y_min) + y_min


# --- Modèle GNN ---
class PressureGNN(nn.Module):
    def __init__(self):
        super(PressureGNN, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []

    def add_layer(self, layer_class, in_channels, out_channels, activation=None, batch_norm=False, dropout=0.0, **kwargs):
        """
        Ajoute une couche avec activation, batch normalization et dropout optionnels.
        """
        layer = layer_class(in_channels, out_channels, **kwargs)
        self.layers.append(layer)
        self.activations.append(activation if activation else lambda x: x)
        
        if batch_norm:
            self.layers.append(BatchNorm(out_channels))  # Ajouter BatchNorm
            self.activations.append(lambda x: x)  # BatchNorm n'a pas d'activation directe

        if dropout > 0.0:
            self.layers.append(nn.Dropout(p=dropout))  # Ajouter Dropout
            self.activations.append(lambda x: x)  # Dropout n'a pas d'activation propre

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer, activation in zip(self.layers, self.activations):
            x = activation(layer(x, edge_index) if isinstance(layer, GCNConv) else layer(x))
        return x




def train_gnn_with_validation(model, train_loader, test_loader, test_params, optimizer, loss_fn, epochs, scheduler, best_loss, early_stop_patience, no_improve_count):
    model.train()
    train_losses = []
    test_r2_scores = []

    for epoch in tqdm(range(epochs), desc="Entraînement"):
        # Entraînement
        epoch_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out.squeeze(), batch.y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        # Validation sur les données de test
        _, test_r2 = evaluate_gnn(model, test_loader, test_params)
        test_r2_scores.append(test_r2)

        # Scheduler step
        scheduler.step(epoch_loss / len(train_loader))

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count >= early_stop_patience:
            print(f"Early stopping at epoch {epoch}")
            break
        
        # Log metrics to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss / len(train_loader), epoch)
        writer.add_scalar('R2/test', test_r2, epoch)

    writer.close()

    return train_losses, test_r2_scores



# --- Évaluation ---
def evaluate_gnn(model, loader, params_list):
    model.eval()
    all_targets, all_predictions = [], []
    with torch.no_grad():
        for batch, params in zip(loader, params_list):
            out = model(batch)
            y_pred_denormalized = denormalize_data(out.squeeze().numpy(), params[2], params[3])
            y_target_denormalized = denormalize_data(batch.y.numpy(), params[2], params[3])
            all_targets.extend(y_target_denormalized)
            all_predictions.extend(y_pred_denormalized)
    mse = mean_squared_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)
    return mse, r2

# --- Tracer les courbes ---
def plot_training_and_validation(train_losses, test_r2_scores):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Courbe de perte d'entraînement
    color = 'tab:blue'
    ax1.set_xlabel("Époque")
    ax1.set_ylabel("Perte (MSE)", color=color)
    ax1.plot(train_losses, label="Perte d'entraînement", color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.legend(loc="upper left")

    # Courbe de R² sur les données de test
    ax2 = ax1.twinx()
    color = 'tab:orange'
    ax2.set_ylabel("R² (Données de test)", color=color)
    ax2.plot(test_r2_scores, label="R² (Test)", color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.legend(loc="upper right")

    fig.tight_layout()
    plt.title("Perte d'entraînement et R² (Test)")
    plt.grid(True)
    plt.show()

# --- Main ---
if __name__ == "__main__":
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir='./logs')
    
    num_train_graphs = 80
    num_test_graphs = 20
    num_points_train = 1000
    num_points_test = 1000
    epochs = 400

    pressure_graph = PressureGraph(P0=200, 
                                   x_c=5, 
                                   y_c=5, 
                                   a=5, 
                                   b=3, 
                                   epsilon=10, 
                                   k=2 * np.pi / 10)
    
    train_graphs, test_graphs, train_params, test_params = pressure_graph.generate_graphs(num_train_graphs,
                                                                                          num_test_graphs, 
                                                                                          num_points_train, 
                                                                                          num_points_test)                                                    

    train_loader = DataLoader(train_graphs, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=1, shuffle=False)



    # -- Création du modèle --
    model = PressureGNN()

    # Ajout des couches avec options avancées
    model.add_layer(GCNConv, 2, 16, activation=F.relu, batch_norm=True, dropout=0.2)
    model.add_layer(GCNConv, 16, 32, activation=F.relu, batch_norm=True, dropout=0.3)
    model.add_layer(GCNConv, 32, 64, activation=F.relu, batch_norm=True, dropout=0.4)
    model.add_layer(GCNConv, 64, 128, activation=F.relu, batch_norm=True, dropout=0.4)
    model.add_layer(GCNConv, 128, 64, activation=F.relu, batch_norm=True, dropout=0.3)
    model.add_layer(GCNConv, 64, 32, activation=F.relu, batch_norm=True, dropout=0.2)
    model.add_layer(GCNConv, 32, 16, activation=F.relu, batch_norm=True)
    #model.add_layer(GCNConv, 16, 1, activation=torch.sigmoid)
    
    #ajouter dense layers ?
    model.add_layer(nn.Linear, 16, 64, activation=F.relu)
    model.add_layer(nn.Linear, 64, 1, activation=torch.sigmoid)

    # Affichage du modèle
    print(model)
    

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) #pas utile adam a déjà
    loss_fn = nn.MSELoss()


    best_loss = float('inf')
    early_stop_patience = 10
    no_improve_count = 0

    train_losses, test_r2_scores = train_gnn_with_validation(
        model=model, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        test_params=test_params, 
        optimizer=optimizer, 
        loss_fn=loss_fn, 
        epochs=epochs,
        scheduler=scheduler, 
        best_loss=best_loss, 
        early_stop_patience=early_stop_patience, 
        no_improve_count=no_improve_count
    )

    train_mse, train_r2 = evaluate_gnn(model, train_loader, train_params)
    test_mse, test_r2 = evaluate_gnn(model, test_loader, test_params)

    print(f"Train MSE: {train_mse:.4f}, Train R²: {train_r2:.4f}")
    print(f"Test MSE: {test_mse:.4f}, Test R²: {test_r2:.4f}")

    # Tracer les courbes de perte et de R²
    plot_training_and_validation(train_losses, test_r2_scores)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    # model = PressureGNN() #in_channels=2, hidden_channels=128, out_channels=1, num_layers=num_layers, activation_fn=activation_fn)
    
    # # première couche
    # model.add_layer(torch.relu, 2, 16)
    
    # # couches cachées
    # # montée
    # model.add_layer(torch.relu, 16, 32)    
    # model.add_layer(torch.relu, 32, 64)
    # model.add_layer(torch.relu, 64, 128)
    # # descente
    # model.add_layer(torch.relu, 128, 64)
    # model.add_layer(torch.relu, 64, 32)
    # model.add_layer(torch.relu, 32, 16)
    
    # # couche de sortie
    # model.add_layer(torch.sigmoid, 16, 1)
    
    #############" Idées amélioration"#############
    #tester d'autres structures
    #mettre early stoppage
    #rajouter R² train
    
    #prochaine fois 
    #essayer meême maillage, loi qui change
    
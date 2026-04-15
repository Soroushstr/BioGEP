from seq_encoder import build_dataset
from gnn_models import CustomGAT
from gnn_models import WeightedGCN 
from pipeline import train
from pipeline import test
import torch

graphs, vocab = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/pool80_genes.fasta", 
                              "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/pool80_labels.txt", k=4)
config = torch.load("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/models/config.pth")
device = "cpu"
model = WeightedGCN(**config).to(device)
results = train(graphs, model, model_path="pool_model.pt")

# Ara
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/arabidopsis_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/arabidopsis_labels.txt", k=4)
results_test = test(graphs=test_graphs, model=model, model_path="pool_model.pt")
print("ARABIDOTEST")
print(results_test)

# Ele
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/elegans_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/elegans_labels.txt", k=4)
results_test = test(graphs=test_graphs, model=model, model_path="pool_model.pt")
print("ELEGANSTEST")
print(results_test)

# MEL
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/melanogaster_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/melanogaster_labels.txt", k=4)
results_test = test(graphs=test_graphs, model=model, model_path="pool_model.pt")
print("MELANOTEST")
print(results_test)

# Hom
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/sapiens_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/sapiens_labels.txt", k=4)
results_test = test(graphs=test_graphs, model=model, model_path="pool_model.pt")
print("HOMOTEST")
print(results_test)

# Sac
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/saccharomyces_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/saccharomyces_labels.txt", k=4)
results_test = test(graphs=test_graphs, model=model, model_path="pool_model.pt")
print("SACCHAROTEST")
print(results_test)

# Bas
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/bacillus_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/bacillus_labels.txt", k=4)
results_test = test(graphs=test_graphs, model=model, model_path="pool_model.pt")
print("BASITEST")
print(results_test)

# Mar
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/maripaludis_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/maripaludis_labels.txt", k=4)
results_test = test(graphs=test_graphs, model=model, model_path="pool_model.pt")
print("MARITEST")
print(results_test)

# pool
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/pool20_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/pool20_labels.txt", k=4)
results_test = test(graphs=test_graphs, model=model, model_path="pool_model.pt")
print("POOLTEST")
print(results_test)

# Mus
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/musculus_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/musculus_labels.txt", k=4)
results_test = test(graphs=test_graphs, model=model, model_path="pool_model.pt")
print("MUSCULTEST")
print(results_test)
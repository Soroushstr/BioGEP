from old_seq_encoder import build_dataset
from gnn_models import CustomGAT
from gnn_models import WeightedGCN 
from new_pipeline import train
from new_pipeline import test
import torch

graphs, vocab = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/pool80_genes.fasta", 
                              "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/pool80_labels.txt", k=4)
config = torch.load("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/models/config.pth")
device = "cuda"
model = WeightedGCN(**config).to(device)
results = train(graphs, model, model_path="baseline_model.pt")

# Bacillus
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/bacillus_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/bacillus_labels.txt", k=4,
                               vocab = vocab)
results_test = test(graphs=test_graphs, model=model, model_path="baseline_model.pt")
print("Bacillus test")
print(results_test)

# Pool20
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/pool20_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/pool20_labels.txt", k=4,
                               vocab = vocab)
results_test = test(graphs=test_graphs, model=model, model_path="baseline_model.pt")
print("Pool20 test")
print(results_test)

# Mus
test_graphs, _ = build_dataset("/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/musculus_genes.fasta", 
                               "/gpfs/data/fs72129/imansamiei/soroush/GEPNew/data/musculus_labels.txt", k=4,
                               vocab = vocab)
results_test = test(graphs=test_graphs, model=model, model_path="baseline_model.pt")
print("musculus test")
print(results_test)

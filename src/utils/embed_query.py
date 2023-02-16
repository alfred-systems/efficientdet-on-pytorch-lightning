import torch


class Embeddings:
    def __init__(self, batch_embeds, batch_captions) -> None:
        self.embeds = []
        self.captions = []
    
    def query(self, embed_vec: torch.Tensor):
        pass
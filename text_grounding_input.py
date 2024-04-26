import torch


class GroundingNetInput:
    def __init__(self):
        self.set = False 

    def prepare(self, batch):
        """
        batch should be the output from dataset.
        Please define here how to process the batch and prepare the 
        input only for the ground tokenizer. 
        """

        self.set = True
        boxes = batch['boxes']
        masks = batch['masks']
        positive_embeddings = batch["text_embeddings"]


        fg_index = batch["fg_index"] if hasattr(batch,"fg_index") else None
        text_length = batch["text_length"] if hasattr(batch,"text_length") else None
        self.batch, self.max_box, self.in_dim = positive_embeddings.shape
        self.device = positive_embeddings.device
        self.dtype = positive_embeddings.dtype

        return {"boxes": boxes, "masks": masks,  "fg_index": fg_index,
                "positive_embeddings": positive_embeddings, "text_length": text_length}

    def get_null_input(self, batch=None, device=None, dtype=None):
        """
        Guidance for training (drop) or inference, 
        please define the null input for the grounding tokenizer 
        """

        assert self.set, "not set yet, cannot call this function"
        batch = self.batch if batch is None else batch
        device = self.device if device is None else device
        dtype = self.dtype if dtype is None else dtype

        boxes = torch.zeros(batch, self.max_box, 4,).type(dtype).to(device)
        masks = torch.zeros(batch, self.max_box).type(dtype).to(device)
        positive_embeddings = torch.zeros(batch, self.max_box, self.in_dim).type(dtype).to(device)

        return {"boxes": boxes, "masks": masks, "positive_embeddings": positive_embeddings}

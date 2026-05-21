import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class FinetunedLLM(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout_p: float = 0.0):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(input_dim, num_classes)

    def forward(self, batch):
        x = batch["features"]
        z = self.dropout(x)
        z = self.fc1(z)
        return z

    @torch.inference_mode()
    def predict(self, batch):
        self.eval()
        z = self(batch)
        y_pred = torch.argmax(z, dim=1).cpu().numpy()
        return y_pred

    @torch.inference_mode()
    def predict_proba(self, batch):
        self.eval()
        z = self(batch)
        y_probs = F.softmax(z, dim=1).cpu().numpy()
        return y_probs

    def save(self, dp):
        with open(Path(dp, "args.json"), "w") as fp:
            contents = {
                "dropout_p": self.dropout_p,
                "input_dim": self.input_dim,
                "num_classes": self.num_classes,
            }
            json.dump(contents, fp, indent=4, sort_keys=False)
        torch.save(self.state_dict(), os.path.join(dp, "model.pt"))

    @classmethod
    def load(cls, args_fp, state_dict_fp):
        with open(args_fp, "r") as fp:
            kwargs = json.load(fp=fp)
        model = cls(**kwargs)
        model.load_state_dict(torch.load(state_dict_fp, map_location=torch.device("cpu")))
        return model

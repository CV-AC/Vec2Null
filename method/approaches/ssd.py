import torch
from typing import Dict, List


class SelectiveSynapseDampening:
    def __init__(
        self,
        model,
        criterion,
        opt,
        device="cuda" if torch.cuda.is_available() else "cpu",
        parameters=None,
    ):
        self.model = model
        self.criterion = criterion
        self.opt = opt
        self.device = device
        self.alpha = None
        self.xmin = None

        self.lower_bound = parameters["lower_bound"]
        self.exponent = parameters["exponent"]
        self.dampening_constant = parameters["dampening_constant"]
        self.selection_weighting = parameters["selection_weighting"]

    def zerolike_params_dict(self, model: torch.nn) -> Dict[str, torch.Tensor]:


        return dict([(k, torch.zeros_like(p, device=p.device)) for k, p in model.named_parameters()])

    def calc_importance(self, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, torch.Tensor]:


        importances = self.zerolike_params_dict(self.model)
        for batch in dataloader:
            x, identity, y = batch
            self.opt.zero_grad()
            x, y = x.to(device), y.to(device)
            out = self.model(x)


            if isinstance(out, tuple):
                assert isinstance(self.criterion, torch.nn.CrossEntropyLoss)
                out = out[1]

            if type(self.criterion) == torch.nn.BCEWithLogitsLoss:
                loss = self.criterion(out, y.float())
            else:
                loss = self.criterion(out, y)
            loss.backward()

            for (k1, p), (k2, imp) in zip(self.model.named_parameters(), importances.items()):
                if p.grad is not None:
                    imp.data += p.grad.data.clone().pow(2)


        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        return importances

    def modify_weight(
        self,
        original_importance: List[Dict[str, torch.Tensor]],
        forget_importance: List[Dict[str, torch.Tensor]],
    ) -> None:


        num_locations = 0
        with torch.no_grad():
            for (n, p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(),
                original_importance.items(),
                forget_importance.items(),
            ):

                oimp_norm = oimp.mul(self.selection_weighting)
                locations = torch.where(fimp > oimp_norm)
                num_locations += len(locations[0])


                weight = ((oimp.mul(self.dampening_constant)).div(fimp)).pow(self.exponent)
                update = weight[locations]


                min_locs = torch.where(update > self.lower_bound)
                num_locations -= len(min_locs[0])
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)

        print(f"Number of locations updated: {num_locations}")

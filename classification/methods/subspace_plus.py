import torch
import torch.nn as nn
from methods.base import TTAMethod
from utils.registry import ADAPTATION_REGISTRY
from utils.losses import Entropy
import numpy as np
import torch.nn.functional as F
from sklearn.decomposition import PCA
from copy import deepcopy
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@ADAPTATION_REGISTRY.register()
class Subspace_plus(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.softmax_entropy = Entropy()
        self.softmax_entropy = Entropy()
        self.batch_num = 0
        self.moving_W = []
        self.w_num = cfg.SUBSPACE.W_NUM
        self.n_components = cfg.SUBSPACE.N_COMPONENTS
        self.batch_step = cfg.SUBSPACE.BATCH_STEP
        self.temp = None
        self.e_margin = 2.763102111592855
        self.current_model_probs = None

    def loss_calculation(self, x):
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        loss = self.softmax_entropy(outputs).mean(0)
        return outputs, loss

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            imgs_test = x[0]
            out = self.model(imgs_test)
            entropys = self.softmax_entropy(out)
            filter_ids_1 = torch.where(entropys < self.e_margin)
            entropys = entropys[filter_ids_1]

            if self.current_model_probs is not None:
                cosine_similarities = F.cosine_similarity(self.current_model_probs.unsqueeze(dim=0),
                                                          out[filter_ids_1].softmax(1), dim=1)
                filter_ids_2 = torch.where(torch.abs(cosine_similarities) < 0.05)
                entropys = entropys[filter_ids_2]
                updated_probs = update_model_probs(self.current_model_probs,
                                                   out[filter_ids_1][filter_ids_2].softmax(1))
            else:
                updated_probs = update_model_probs(self.current_model_probs, out[filter_ids_1].softmax(1))
            coeff = 1 / (torch.exp(entropys.clone().detach() - self.e_margin))
            entropys = entropys.mul(coeff)
            self.current_model_probs = updated_probs
            loss = entropys.mean(0)
            loss.backward()

            model_state, optimizer_state = \
                copy_model_and_optimizer(self.model, self.optimizer)
            if self.batch_num % self.batch_step == 0 and not self.batch_num == 0:
                self.moving_W.append(get_model_grad_vec1(self.model))
                if len(self.moving_W) > self.w_num:
                    self.moving_W.pop(0)
                if self.moving_W[0] is None:
                    self.moving_W.pop(0)
            elif self.batch_num == 0:
                self.moving_W.append(self.temp)


            if len(self.moving_W) >= self.n_components:
                W = np.array(self.moving_W)
                W = W - np.mean(W, axis=0)
                pca = PCA(n_components=self.n_components)
                pca.fit_transform(W)
                P = np.array(pca.components_)
                P = torch.from_numpy(P).to(device)
                load_model_and_optimizer(self.model, self.optimizer,
                                         model_state, optimizer_state)
                gk = get_model_grad_vec(self.model)
                P_SGD(self.model, self.optimizer, gk, P)
            else:
                self.optimizer.step()
            self.batch_num += 1
            self.optimizer.zero_grad()
        return out

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
            elif isinstance(m, nn.BatchNorm1d):
                m.train()
                m.requires_grad_(True)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                m.requires_grad_(True)

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def P_SGD(model, optimizer, grad, P, retrun_grad=False):
  gk = torch.mm(P, grad.reshape(-1, 1))
  grad_proj = torch.mm(P.transpose(0, 1), gk)
  update_grad(model, grad_proj)
  gk_proj = get_model_grad_vec(model)
  optimizer.step()
  return gk_proj if retrun_grad else None

def update_grad(model, grad_vec):
  idx = 0
  for name, param in model.named_parameters():
    if param.requires_grad:
        if param.grad is None:
          param.grad = torch.zeros_like(param)
        arr_shape = param.grad.shape
        size = 1
        for i in range(len(list(arr_shape))):
          size *= arr_shape[i]
        param.grad.data = grad_vec[idx:idx + size].reshape(arr_shape)
        idx += size

def get_model_grad_vec1(model):
  vec = []
  for name, param in model.named_parameters():
    if param.requires_grad:
        vec.append(param.grad.detach().cpu().numpy().reshape(-1))
  return np.concatenate(vec, 0)


def get_model_grad_vec(model):

  vec = []
  for name, param in model.named_parameters():
    if param.requires_grad:
        vec.append(param.grad.detach().reshape(-1))
  return torch.cat(vec, 0)


def update_model_probs(current_model_probs, new_probs):
    if current_model_probs is None:
        if new_probs.size(0) == 0:
            return None
        else:
            with torch.no_grad():
                return new_probs.mean(0)
    else:
        if new_probs.size(0) == 0:
            with torch.no_grad():
                return current_model_probs
        else:
            with torch.no_grad():
                return 0.9 * current_model_probs + (1 - 0.9) * new_probs.mean(0)
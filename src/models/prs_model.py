import os
import torch
import torch.nn as nn
import src.models as mdl

class PRS_classifier(nn.Module):
    def __init__(self, opt, num_classes=10, pretrain=True):
        super().__init__()
        if pretrain == True:
            model_path = os.path.join(opt.save_folder, opt.ckpt)
            model_info = torch.load(model_path, weights_only=False)
            self.encoder = mdl.SupConResNet(
                name=opt.model, 
                head=opt.head, 
                feat_dim=opt.embedding_size
            )
            self.encoder.load_state_dict(model_info["model"])
            self.classifier = mdl.LinearClassifier(
                dim_in=opt.embedding_size, 
                num_classes=num_classes
            )
        else:
            self.encoder = mdl.SupConResNet(
                name=opt.model_name, 
                head=opt.head, 
                feat_dim=opt.embedding_size
            )
            self.classifier = mdl.LinearClassifier(
                dim_in=opt.embedding_size, 
                num_classes=num_classes
            )

    def forward(self, x):
        encode = self.encoder(x)
        outputs = self.classifier(encode)
        return outputs
    

class PRS_Model(nn.Module):
    def __init__(self, model, head, embedding_size, num_classes=10):
        super().__init__()
        self.encoder = mdl.SupConResNet(
            name=model, 
            head=head, 
            feat_dim=embedding_size
        )
        self.classifier = mdl.LinearClassifier(
            dim_in=embedding_size, 
            num_classes=num_classes
        )

    def forward(self, x):
        encode = self.encoder(x)
        outputs = self.classifier(encode)
        return outputs

   


#extra class 
class MoEClassifier(nn.Module):
    def __init__(self, dim_in, num_classes, num_experts=4, hidden_dim=256, k=2):
        super().__init__()
        self.num_experts = num_experts
        self.k = k

        # Gating network (same idea, but we’ll do top-k selection in forward)
        self.gate = nn.Linear(dim_in, num_experts)

        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
            for _ in range(num_experts)
        ])

    # def forward(self, x):
    #     # -----------------
    #     # 1) Compute raw gate scores
    #     # -----------------
    #     gate_logits = self.gate(x)                     # [batch, num_experts]
    #     gate_probs = torch.softmax(gate_logits, dim=-1)

    #     # -----------------
    #     # 2) Top-k selection (sparse gating)
    #     # -----------------
    #     topk_vals, topk_idx = torch.topk(gate_probs, self.k, dim=-1)  # [batch, k]

    #     # Normalize top-k probs so they sum to 1
    #     topk_vals = topk_vals / torch.sum(topk_vals, dim=-1, keepdim=True)

    #     # -----------------
    #     # 3) Compute outputs from selected experts only
    #     # -----------------
    #     batch_size = x.size(0)
    #     outputs = torch.zeros(batch_size, self.experts[0][-1].out_features, device=x.device)

    #     for i in range(self.k):
    #         idx = topk_idx[:, i]     # [batch]
    #         weight = topk_vals[:, i] # [batch]

    #         # Compute expert outputs
    #         expert_outs = torch.zeros_like(outputs)
    #         for exp_id in range(self.num_experts):
    #             mask = (idx == exp_id)
    #             if mask.any():
    #                 expert_outs[mask] = self.experts[exp_id](x[mask])

    #         # Weighted sum
    #         outputs += weight.unsqueeze(-1) * expert_outs

    #     # -----------------
    #     # 4) Auxiliary load-balancing loss (from Switch/Google MoE)
    #     # Encourage uniform expert usage
    #     # -----------------
    #     expert_prob_mean = torch.mean(gate_probs, dim=0)   # [num_experts]
    #     # load_balancing_loss = torch.mean(expert_prob_mean * self.num_experts)
    #     # Encourage all experts to be used equally
    #     load_balancing_loss = self.num_experts * torch.sum(expert_prob_mean ** 2)


    #     return outputs, load_balancing_loss
    def forward(self, x, return_gates=False):  # add return_gates flag
        gate_logits = self.gate(x)
        gate_probs = torch.softmax(gate_logits, dim=-1)
        topk_vals, topk_idx = torch.topk(gate_probs, self.k, dim=-1)
        topk_vals = topk_vals / torch.sum(topk_vals, dim=-1, keepdim=True)

        batch_size = x.size(0)
        outputs = torch.zeros(batch_size, self.experts[0][-1].out_features, device=x.device)

        for i in range(self.k):
            idx = topk_idx[:, i]
            weight = topk_vals[:, i]
            expert_outs = torch.zeros_like(outputs)
            for exp_id in range(self.num_experts):
                mask = (idx == exp_id)
                if mask.any():
                    expert_outs[mask] = self.experts[exp_id](x[mask])
            outputs += weight.unsqueeze(-1) * expert_outs

        # Aux loss
        expert_prob_mean = torch.mean(gate_probs, dim=0)
        load_balancing_loss = self.num_experts * torch.sum(expert_prob_mean ** 2)

        if return_gates:  # allow returning gate info
            return outputs, load_balancing_loss, topk_idx
        else:
            return outputs, load_balancing_loss


# --------------------------
# PRS Classifier with MoE
# --------------------------
class PRS_classifier2(nn.Module):
    def __init__(self, opt, num_classes=10, pretrain=True, num_experts=2, hidden_dim=256):
        super().__init__()
        if pretrain:
            model_path = os.path.join(opt.save_folder, opt.ckpt)
            model_info = torch.load(model_path)
            self.encoder = mdl.SupConResNet(
                name=opt.model,
                head=opt.head,
                feat_dim=opt.embedding_size
            )
            self.encoder.load_state_dict(model_info["model"])
        else:
            self.encoder = mdl.SupConResNet(
                name=opt.model_name,
                head=opt.head,
                feat_dim=opt.embedding_size
            )

        self.classifier = MoEClassifier(
            dim_in=opt.embedding_size,
            num_classes=num_classes,
            num_experts=num_experts,
            hidden_dim=hidden_dim
        )

    def forward(self, x, return_gates=False):   #  accept return_gates
        encode = self.encoder(x)
        return self.classifier(encode, return_gates=return_gates)  #  pass through


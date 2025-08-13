import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from torch.utils.checkpoint import checkpoint


class Adaptive_Score_Generation(nn.Module):
    def __init__(
        self,
        dim,
        num_slices=5,
        d_state=16,
        d_conv=4,
        expand=2,
        number_gs=3,
        length_gs=2048,
    ):
        super(Adaptive_Score_Generation, self).__init__()
        self.individual_score_generator = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                bimamba_type="v3",
                nslices=num_slices,
            ),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )
        self.group_score_embeddings = nn.ParameterList(
            [nn.Parameter(torch.randn(length_gs)) for _ in range(number_gs)]
        )

    def forward(self, x, mask=None):
        individual_score = self.individual_score_generator(x)[:, :, 0]
        b, s, d = x.shape
        group_scores = []
        for gse in self.group_score_embeddings:
            gse_interp = F.interpolate(
                gse[None, None],  # shape: (1, 1, L)
                size=s,
                mode="linear",
                align_corners=True,
            )
            group_scores.append(torch.sigmoid(gse_interp)[0, 0])  # shape: (s,)
        return group_scores, individual_score


class Mamba_Layer(nn.Module):
    def __init__(self, dim, num_slices, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.relu = nn.ReLU(inplace=True)
        # self.mamba = Mamba(
        #     d_model=dim,  # Model dimension d_model
        #     d_state=d_state,  # SSM state expansion factor
        #     d_conv=d_conv,  # Local convolution width
        #     expand=expand,  # Block expansion factor
        #     bimamba_type="v3",
        #     nslices=num_slices,
        # )
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
            bimamba_type="v1",
        )

    def forward(self, x):
        x_skip = x
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        x_mamba = self.mamba(x)
        out = x_mamba + x_skip
        return out


class Adaptive_Scan_Mamba(nn.Module):
    def __init__(
        self,
        in_dim,
        if_lmls=False,
        depths=2,
        num_slices_stack=[5, 5],
        num_slices_score=5,
        number_gs=3,
        length_gs=2048,
    ):
        super(Adaptive_Scan_Mamba, self).__init__()
        self.in_dim = in_dim
        # if_lmls: whether to enable "low memory, low speed" mode
        self.if_lmls = if_lmls
        self.number_gs = number_gs
        self.depths = depths
        self.num_slices_stack = num_slices_stack
        self.scores_generation = Adaptive_Score_Generation(
            in_dim,
            num_slices=num_slices_score,
            number_gs=number_gs,
            length_gs=length_gs,
        )
        assert (
            len(num_slices_stack) == depths
        ), "the length of num_slices_stack should be equal to depths"

        if if_lmls:
            self.stacked_mamba_layers = self.construct_stack_mambas1()
        else:
            self.stacked_mamba_layers = self.construct_stack_mambas2()
        self.proj_g = nn.Sequential(
            nn.Linear(in_dim + 1, in_dim),
            nn.LayerNorm(in_dim),
            # nn.ReLU(inplace=True)
        )

    def construct_stack_mambas1(self,):
        stacked_mamba_layers = []
        for i in range(self.number_gs):
            mamba_layers = []
            for i in range(self.depths):
                mamba_layers.append(Mamba_Layer(self.in_dim + 1, num_slices=self.num_slices_stack[i]))
                mamba_layers.append(nn.LayerNorm(self.in_dim + 1))
            mamba_layers = nn.Sequential(*mamba_layers)
            stacked_mamba_layers.append(mamba_layers)
        return nn.ParameterList(stacked_mamba_layers)

    def construct_stack_mambas2(self,):
        mamba_layers = []
        for i in range(self.depths):
            mamba_layers.append(Mamba_Layer(self.in_dim + 1, num_slices=self.num_slices_stack[i]))
            mamba_layers.append(nn.LayerNorm(self.in_dim + 1))
        mamba_layers = nn.Sequential(*mamba_layers)
        return mamba_layers

    def flatten_input_tensor(self, input_tensor):
        len_flag = len(input_tensor.shape)
        if len_flag == 3:
            return input_tensor, 3, input_tensor.shape
        elif len_flag == 4:
            b, d, w, h = input_tensor.shape
            input_tensor = input_tensor.permute(0, 2, 3, 1).reshape(b, w * h, d)
            return input_tensor, 4, (b, d, w, h)
        elif len_flag == 5:
            b, d, l, w, h = input_tensor.shape
            input_tensor = input_tensor.permute(0, 2, 3, 4, 1).reshape(b, l * w * h, d)
            return input_tensor, 5, (b, d, l, w, h)
        else:
            raise ValueError(f"Unsupported input shape: {input_tensor.shape}")

    def recover_input_tensor(self, tensor, len_flag, original_shape):
        if len_flag == 3:
            return tensor
        elif len_flag == 4:
            b, d, w, h = original_shape
            return tensor.permute(0, 2, 1).reshape(b, d, w, h)
        elif len_flag == 5:
            b, d, l, w, h = original_shape
            return tensor.permute(0, 2, 1).reshape(b, d, l, w, h)
        else:
            raise ValueError(f"Unsupported len_flag: {len_flag}")

    def forward1(self, input_tensor):
        input_tensor, len_flag, original_shape = self.flatten_input_tensor(input_tensor)
        group_scores, individual_score = self.scores_generation(input_tensor)
        outputs = []
        for i in range(self.number_gs):
            adpative_score = group_scores[i].repeat(original_shape[0], 1) + individual_score
            _, indices = torch.sort(adpative_score, dim=1)
            output = torch.cat((input_tensor, adpative_score.unsqueeze(-1)), dim=2)
            output = torch.gather(
                output,
                1,
                indices.unsqueeze(-1).expand(-1, -1, output.size(2)),
            )
            def forward_fn(x_): return self.stacked_mamba_layers[i](x_)
            output = checkpoint(forward_fn, output, use_reentrant=False)
            restore_indices = torch.argsort(indices, dim=1)
            output = torch.gather(
                output,
                1,
                restore_indices.unsqueeze(-1).expand(
                    -1, -1, output.size(2)
                ),
            )
            outputs.append(output)
        outputs = torch.mean(torch.stack(outputs), dim=0)
        outputs = self.proj_g(outputs)
        outputs = self.recover_input_tensor(outputs, len_flag, original_shape)
        return outputs

    def forward2(self, input_tensor):
        input_tensor, len_flag, original_shape = self.flatten_input_tensor(input_tensor)
        group_scores, individual_score = self.scores_generation(input_tensor)
        gathered_inputs = []
        restore_indices_list = []
        for i in range(self.number_gs):
            # 1. adaptive score
            adpative_scores = group_scores[i].repeat(original_shape[0], 1) + individual_score  # (B, L)
            # 2. sort
            _, indices = torch.sort(adpative_scores, dim=1)  # (B, L)
            restore_indices = torch.argsort(indices, dim=1)
            # 3. gather input
            output = torch.cat((input_tensor, adpative_scores.unsqueeze(-1)), dim=2)  # (B, L, D+1)
            gathered = torch.gather(output, 1, indices.unsqueeze(-1).expand(-1, -1, output.size(2)))  # (B, L, D+1)
            # 4. collect values
            gathered_inputs.append(gathered)
            restore_indices_list.append(restore_indices)

        #Run Mamba
        batched_input = torch.cat(gathered_inputs, dim=0)  # (B×N, L, D+1)
        batched_output = checkpoint(self.stacked_mamba_layers, batched_input, use_reentrant=False)  # (B×N, L, D+1)

        outputs = []
        B = original_shape[0]
        for i in range(self.number_gs):
            out_i = batched_output[i*B:(i+1)*B]  # (B, L, D+1)
            restore_idx = restore_indices_list[i]  # (B, L)
            out_i = torch.gather(out_i, 1, restore_idx.unsqueeze(-1).expand(-1, -1, out_i.size(2)))  # (B, L, D+1)
            outputs.append(out_i)

        outputs = torch.mean(torch.stack(outputs), dim=0)
        outputs = self.proj_g(outputs)
        outputs = self.recover_input_tensor(outputs, len_flag, original_shape)
        return outputs

    def forward(self, input_tensor):
        if self.if_lmls:
            outputs = self.forward1(input_tensor)
        else:
            outputs = self.forward2(input_tensor)
        return outputs
"""
Modified from https://github.com/microsoft/Graphormer
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .tokengt_graph_encoder import init_graphormer_params
from .performer_pytorch import ProjectionUpdater
from .tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer
from .tokenizer import GraphFeatureTokenizer


class TokenGTGraphDecoder(nn.Module):
    def __init__(
            self,
            graph_feature: GraphFeatureTokenizer = None,
            stochastic_depth: bool = False,

            performer: bool = False,
            performer_finetune: bool = False,
            performer_nb_features: int = None,
            performer_feature_redraw_interval: int = 1000,
            performer_generalized_attention: bool = False,
            performer_auto_check_redraw: bool = True,

            num_decoder_layers: int = 12,
            embedding_dim: int = 768,
            ffn_embedding_dim: int = 768,
            num_attention_heads: int = 32,
            dropout: float = 0.1,
            attention_dropout: float = 0.1,
            activation_dropout: float = 0.1,
            layerdrop: float = 0.0,
            encoder_normalize_before: bool = False,
            layernorm_style: str = "postnorm",
            apply_graphormer_init: bool = False,
            activation_fn: str = "gelu",
            embed_scale: float = None,
            freeze_embeddings: bool = False,
            n_trans_layers_to_freeze: int = 0,
            export: bool = False,
            traceable: bool = False,
            q_noise: float = 0.0,
            qn_block_size: int = 8,

            return_attention: bool = False,
            masked: bool = True
    ) -> None:
        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable
        self.performer = performer
        self.performer_finetune = performer_finetune

        self.performer_finetune = performer_finetune
        self.embed_scale = embed_scale
        self.graph_feature = graph_feature
        if masked:
            self.mask_token = nn.Embedding(1, embedding_dim)

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if layernorm_style == "prenorm":
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])

        if stochastic_depth:
            assert layernorm_style == 'prenorm'  # only for residual nets

        self.cached_performer_options = None
        if self.performer_finetune:
            assert self.performer
            self.cached_performer_options = (
                performer_nb_features,
                performer_generalized_attention,
                performer_auto_check_redraw,
                performer_feature_redraw_interval
            )
            self.performer = False
            performer = False
            performer_nb_features = None
            performer_generalized_attention = False
            performer_auto_check_redraw = False
            performer_feature_redraw_interval = None

        self.layers.extend(
            [
                self.build_tokengt_graph_decoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    decoder_layers=num_decoder_layers,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    drop_path=(0.1 * (layer_idx + 1) / num_decoder_layers) if stochastic_depth else 0,
                    performer=performer,
                    performer_nb_features=performer_nb_features,
                    performer_generalized_attention=performer_generalized_attention,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    layernorm_style=layernorm_style,
                    return_attention=return_attention,
                )
                for layer_idx in range(num_decoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

        if performer:
            # keeping track of when to redraw projections for all attention layers
            self.performer_auto_check_redraw = performer_auto_check_redraw
            self.performer_proj_updater = ProjectionUpdater(self.layers, performer_feature_redraw_interval)

    def performer_fix_projection_matrices_(self):
        self.performer_proj_updater.feature_redraw_interval = None

    def performer_finetune_setup(self):
        assert self.performer_finetune
        (
            performer_nb_features,
            performer_generalized_attention,
            performer_auto_check_redraw,
            performer_feature_redraw_interval
        ) = self.cached_performer_options

        for layer in self.layers:
            layer.performer_finetune_setup(performer_nb_features, performer_generalized_attention)

        self.performer = True
        self.performer_auto_check_redraw = performer_auto_check_redraw
        self.performer_proj_updater = ProjectionUpdater(self.layers, performer_feature_redraw_interval)

    def build_tokengt_graph_decoder_layer(
            self,
            embedding_dim,
            ffn_embedding_dim,
            decoder_layers,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            drop_path,
            performer,
            performer_nb_features,
            performer_generalized_attention,
            activation_fn,
            export,
            q_noise,
            qn_block_size,
            layernorm_style,
            return_attention,
    ):
        return TokenGTGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            encoder_layers=decoder_layers,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            drop_path=drop_path,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            layernorm_style=layernorm_style,
            return_attention=return_attention
        )

    def forward(
            self,
            x: torch.Tensor,
            padded_index: torch.Tensor,
            padding_mask: torch.Tensor,
            perturb=None,
            last_state_only: bool = True,
            token_embeddings: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            masked_tokens: Optional[torch.Tensor] = None,
    ):
        is_tpu = False

        if masked_tokens is not None:
            x[masked_tokens] = self.mask_token.weight.expand(*x[masked_tokens].size()) + token_embeddings[masked_tokens]

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        if attn_mask is not None:
            raise NotImplementedError

        attn_dict = {'maps': {}, 'padded_index': padded_index}
        for i in range(len(self.layers)):
            layer = self.layers[i]
            x, attn = layer(x, self_attn_padding_mask=padding_mask, self_attn_mask=attn_mask, self_attn_bias=None)
            if not last_state_only:
                inner_states.append(x)
            attn_dict['maps'][i] = attn

        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), graph_rep, attn_dict
        else:
            return inner_states, graph_rep, attn_dict

import logging

import torch
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import LayerNorm
from fairseq.utils import safe_hasattr

from tokengt.models import TokenGTModel
from tokengt.models.tokengt import TokenGTEncoder, base_architecture
from tokengt.modules import TokenGTGraphDecoder

logger = logging.getLogger(__name__)


@register_model("graph_coder_masked")
class GraphCoderMaskedModel(TokenGTModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        graph_coder_masked_base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        logger.info(args)

        encoder = GraphCoderAutoEncoder(args)
        return cls(args, encoder)

    def get_targets(self, sample, net_output):
        tokens_num = [node_num + edge_num for node_num, edge_num in zip(sample["net_input"]["batched_data"]["node_num"], sample["net_input"]["batched_data"]["edge_num"])]
        max_n = max(tokens_num)

        targets = torch.cat([F.pad(target[None, ...], (0, max_n - target.size(0)), value=-100) for target in sample["target"]])

        return targets[sample["net_input"]["masked_tokens"]]


class GraphCoderAutoEncoder(TokenGTEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.decoder_layer_norm = LayerNorm(args.encoder_embed_dim)

        if args.prenorm:
            layernorm_style = "prenorm"
        elif args.postnorm:
            layernorm_style = "postnorm"
        else:
            raise NotImplementedError

        self.graph_decoder = TokenGTGraphDecoder(
            # <
            graph_feature=self.graph_encoder.graph_feature,
            stochastic_depth=args.stochastic_depth,
            performer=args.performer,
            performer_finetune=args.performer_finetune,
            performer_nb_features=args.performer_nb_features,
            performer_feature_redraw_interval=args.performer_feature_redraw_interval,
            performer_generalized_attention=args.performer_generalized_attention,
            num_decoder_layers=args.decoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.act_dropout,
            encoder_normalize_before=args.encoder_normalize_before,
            layernorm_style=layernorm_style,
            apply_graphormer_init=args.apply_graphormer_init,
            activation_fn=args.activation_fn,
            return_attention=args.return_attention
            # >
        )

    def encoder(self, batched_data, perturb=None, masked_tokens=None):
        inner_states, _, attn_dict = self.graph_encoder(
            batched_data,
            perturb=perturb,
            masked_tokens=masked_tokens
        )

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        x = self.layer_norm(x)

        return x, attn_dict

    def decoder(self, output, padded_index, padding_mask, masked_tokens=None):
        inner_states, _, attn_dict = self.graph_decoder(
            output,
            padded_index,
            padding_mask,
        )

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        if masked_tokens is not None:
            x = self.decoder_layer_norm(x[masked_tokens])
        else:
            x = self.decoder_layer_norm(x)

        return x, attn_dict

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        x, attn_dict = self.encoder(batched_data, perturb=perturb, masked_tokens=masked_tokens)
        x = self.lm_head_transform_weight(x)
        x, attn_dict = self.decoder(x, padded_index=attn_dict["padded_index"], padding_mask=attn_dict["padding_mask"], masked_tokens=masked_tokens)

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
                self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        if self.return_attention:
            return x, attn_dict
        else:
            return x


@register_model_architecture("graph_coder_masked", "graph_coder_masked_base")
def graph_coder_masked_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.dropout = getattr(args, "dropout", 0.0)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.act_dropout = getattr(args, "act_dropout", 0.1)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.apply_graphormer_init = getattr(args, "apply_graphormer_init", True)
    args.share_encoder_input_output_embed = getattr(args, "share_encoder_input_output_embed", False)
    args.prenorm = getattr(args, "prenorm", False)
    args.postnorm = getattr(args, "postnorm", False)

    args.rand_node_id = getattr(args, "rand_node_id", False)
    args.rand_node_id_dim = getattr(args, "rand_node_id_dim", 64)
    args.orf_node_id = getattr(args, "orf_node_id", False)
    args.orf_node_id_dim = getattr(args, "orf_node_id_dim", 64)
    args.lap_node_id = getattr(args, "lap_node_id", True)
    args.lap_node_id_k = getattr(args, "lap_node_id_k", 16)
    args.lap_node_id_sign_flip = getattr(args, "lap_node_id_sign_flip", True)
    args.lap_node_id_eig_dropout = getattr(args, "lap_node_id_eig_dropout", 0.2)
    args.type_id = getattr(args, "type_id", True)

    args.stochastic_depth = getattr(args, "stochastic_depth", False)

    args.performer = getattr(args, "performer", False)
    args.performer_finetune = getattr(args, "performer_finetune", False)
    args.performer_nb_features = getattr(args, "performer_nb_features", None)
    args.performer_feature_redraw_interval = getattr(args, "performer_feature_redraw_interval", 1000)
    args.performer_generalized_attention = getattr(args, "performer_generalized_attention", False)

    args.return_attention = getattr(args, "return_attention", False)
    args.special_tokens = getattr(args, "special_tokens", False)
    args.masked = getattr(args, "masked", True)
    base_architecture(args)


@register_model_architecture("graph_coder_masked", "graph_coder_masked_ablated")
def graph_coder_masked_ablated_architecture(args):
    args.lap_node_id = getattr(args, "lap_node_id", False)
    args.type_id = getattr(args, "type_id", False)
    graph_coder_masked_base_architecture(args)


@register_model_architecture("graph_coder_masked", "graph_coder_masked_tiny")
def graph_coder_masked_tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    graph_coder_masked_base_architecture(args)

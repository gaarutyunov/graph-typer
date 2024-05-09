import torch
from torch.nn import functional as F

from tokengt.models import TokenGTModel
from tokengt.models.tokengt import base_architecture


class GraphCoderMaskedModel(TokenGTModel):
    def get_targets(self, sample, net_output):
        tokens_num = [node_num + edge_num for node_num, edge_num in zip(sample["net_input"]["batched_data"]["node_num"], sample["net_input"]["batched_data"]["edge_num"])]
        max_n = max(tokens_num)

        targets = torch.cat([F.pad(target[None, ...], (0, max_n - target.size(0)), value=-100) for target in sample["target"]])

        return targets[sample["net_input"]["masked_tokens"]]

    @staticmethod
    def add_args(parser):
        TokenGTModel.add_args(parser)
        parser.add_argument("--masked", type=bool, help="Randomly mask tokens")
        parser.add_argument("--special-tokens", type=bool, help="Use special tokens")


def graph_coder_masked_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
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


def graph_coder_masked_ablated_architecture(args):
    args.lap_node_id = getattr(args, "lap_node_id", False)
    args.type_id = getattr(args, "type_id", False)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    graph_coder_masked_base_architecture(args)


def graph_coder_masked_tiny_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 64)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 128)
    graph_coder_masked_base_architecture(args)
import logging

from fairseq.models import register_model, register_model_architecture

from tokengt.models import TokenGTModel
from tokengt.models.tokengt import base_architecture

logger = logging.getLogger(__name__)


@register_model("graph_coder")
class GraphCoderModel(TokenGTModel):
    pass


@register_model_architecture("graph_coder", "graph_coder_base")
def graphcoder_base_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 768)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 32)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 768)
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
    base_architecture(args)


@register_model_architecture("graph_coder", "graph_coder")
def graphcoder_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    graphcoder_base_architecture(args)


@register_model_architecture("graph_coder", "graph_coder_ablated")
def graphcoder_ablated_architecture(args):
    args.lap_node_id = getattr(args, "lap_node_id", False)
    args.type_id = getattr(args, "type_id", False)
    graphcoder_base_architecture(args)


@register_model_architecture("graph_coder", "graph_coder_big")
def graphcoder_big_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 8192)
    graphcoder_base_architecture(args)


@register_model_architecture("graph_coder", "graph_coder_deep")
def graphcoder_deep_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 64)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    graphcoder_base_architecture(args)


@register_model_architecture("graph_coder", "graph_coder_final")
def graphcoder_ablated_architecture(args):
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    graphcoder_base_architecture(args)

import logging

import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import LayerNorm
from fairseq.utils import safe_hasattr

from .base import GraphCoderMaskedModel, graph_coder_masked_base_architecture, graph_coder_masked_ablated_architecture, \
    graph_coder_masked_tiny_architecture
from tokengt.models.tokengt import TokenGTEncoder
from tokengt.modules import TokenGTGraphDecoder

logger = logging.getLogger(__name__)


@register_model("graph_coder_autoencoder")
class GraphCoderAutoencoderModel(GraphCoderMaskedModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        graph_coder_masked_base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        logger.info(args)

        encoder = GraphCoderAutoencoder(args)
        return cls(args, encoder)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--decoder-layers", type=int, help="Number of decoder layers")
        GraphCoderMaskedModel.add_args(parser)


class GraphCoderAutoencoder(TokenGTEncoder):
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
            return_attention=args.return_attention,
            autoencoder=args.autoencoder
            # >
        )

    def encoder(self, batched_data, perturb=None, masked_tokens=None):
        inner_states, token_embeddings, attn_dict = self.graph_encoder(
            batched_data,
            perturb=perturb,
            masked_tokens=masked_tokens,
            return_embedding=True,
        )

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        x = self.layer_norm(x)

        return x, token_embeddings, attn_dict

    def decoder(self, output, padded_index, padding_mask, token_embeddings=None, masked_tokens=None):
        inner_states, _, attn_dict = self.graph_decoder(
            output,
            padded_index,
            padding_mask,
            token_embeddings=token_embeddings,
            masked_tokens=masked_tokens
        )

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        x = self.decoder_layer_norm(x)

        return x, attn_dict

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        x, token_embeddings, attn_dict = self.encoder(batched_data, perturb=perturb, masked_tokens=masked_tokens)
        x = self.lm_head_transform_weight(x)
        x, attn_dict = self.decoder(
            x,
            padded_index=attn_dict["padded_index"],
            padding_mask=attn_dict["padding_mask"],
            masked_tokens=masked_tokens,
            token_embeddings=token_embeddings
        )

        x = self.activation_fn(x)

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


@register_model_architecture("graph_coder_autoencoder", "graph_coder_autoencoder_base")
def base_architecture(args):
    args.autoencoder = getattr(args, "autoencoder", True)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    graph_coder_masked_base_architecture(args)


@register_model_architecture("graph_coder_autoencoder", "graph_coder_autoencoder_ablated")
def ablated_architecture(args):
    args.autoencoder = getattr(args, "autoencoder", True)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    graph_coder_masked_ablated_architecture(args)


@register_model_architecture("graph_coder_autoencoder", "graph_coder_autoencoder_symmetric")
def symmetric_architecture(args):
    args.autoencoder = getattr(args, "autoencoder", True)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    graph_coder_masked_base_architecture(args)


@register_model_architecture("graph_coder_autoencoder", "graph_coder_autoencoder_tiny")
def tiny_architecture(args):
    args.autoencoder = getattr(args, "autoencoder", True)
    args.decoder_layers = getattr(args, "decoder_layers", 2)
    graph_coder_masked_tiny_architecture(args)

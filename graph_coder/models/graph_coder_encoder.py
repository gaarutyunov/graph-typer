import logging

import torch
from fairseq.models import register_model, register_model_architecture
from fairseq.utils import safe_hasattr
from torch.nn import functional as F

from graph_coder.models.base import GraphCoderMaskedModel, graph_coder_masked_base_architecture, \
    graph_coder_masked_ablated_architecture, graph_coder_masked_tiny_architecture
from tokengt.models.tokengt import TokenGTEncoder

torch.autograd.set_detect_anomaly(True)


logger = logging.getLogger(__name__)


@register_model("graph_coder_encoder")
class GraphCoderEncoderModel(GraphCoderMaskedModel):
    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure all arguments are present in older models
        graph_coder_masked_base_architecture(args)

        if not safe_hasattr(args, "max_nodes"):
            args.max_nodes = args.tokens_per_sample

        logger.info(args)

        encoder = GraphCoderEncoder(args)
        return cls(args, encoder)


class GraphCoderEncoder(TokenGTEncoder):
    def __init__(self, args):
        super().__init__(args)

    def encoder(self, batched_data, perturb=None, masked_tokens=None):
        inner_states, _, attn_dict = self.graph_encoder(
            batched_data,
            perturb=perturb,
            masked_tokens=masked_tokens,
            return_embedding=True,
        )

        x = inner_states[-1].transpose(0, 1)  # B x T x C

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        return x, attn_dict

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        x, attn_dict = self.encoder(batched_data, perturb=perturb, masked_tokens=masked_tokens)

        # project back to size of vocabulary
        if self.share_input_output_embed and hasattr(
                self.graph_encoder.embed_tokens, "weight"
        ):
            x = F.linear(x, self.graph_encoder.embed_tokens.weight)
        elif self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        if masked_tokens is not None:
            x = x[masked_tokens]

        if self.return_attention:
            return x, attn_dict
        else:
            return x


@register_model_architecture("graph_coder_encoder", "graph_coder_encoder_base")
def base_architecture(args):
    graph_coder_masked_base_architecture(args)


@register_model_architecture("graph_coder_encoder", "graph_coder_encoder_not_masked")
def not_masked_architecture(args):
    args.masked = getattr(args, "masked", False)
    graph_coder_masked_base_architecture(args)


@register_model_architecture("graph_coder_encoder", "graph_coder_encoder_ablated")
def ablated_architecture(args):
    graph_coder_masked_ablated_architecture(args)


@register_model_architecture("graph_coder_encoder", "graph_coder_encoder_tiny")
def tiny_architecture(args):
    args.masked = getattr(args, "masked", False)
    graph_coder_masked_tiny_architecture(args)



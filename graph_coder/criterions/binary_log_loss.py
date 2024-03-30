import torch
import torch.nn.functional as F
from fairseq import metrics
from fairseq.criterions import register_criterion, FairseqCriterion
from fairseq.dataclass import FairseqDataclass

from graph_coder.utils import get_padded_node_mask


@register_criterion("binary_log_loss", dataclass=FairseqDataclass)
class BinaryLogLoss(FairseqCriterion):
    """
    Implementation for the binary log loss used in tokengt model training.
    """

    def forward(self, model, sample, perturb=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sample_size = sample["nsamples"]

        with torch.no_grad():
            natoms = max(sample["net_input"]["batched_data"]["node_num"])

            padded_node_mask = get_padded_node_mask(sample)

        logits = model(**sample["net_input"], perturb=perturb)
        targets = model.get_targets(sample, [logits])

        loss = F.binary_cross_entropy_with_logits(
            logits[padded_node_mask],
            F.one_hot(targets, num_classes=model.args.num_classes).type(logits.dtype),
            reduction="sum"
        )

        logging_output = {
            "loss": loss.data,
            "sample_size": logits.size(0),
            "nsentences": sample_size,
            "ntokens": natoms,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=6)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

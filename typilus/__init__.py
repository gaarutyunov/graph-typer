from typing import NamedTuple, Tuple, Dict

from dpu_utils.mlutils import Vocabulary


class Annotation(NamedTuple):
    provenance: str
    node_id: int
    name: str
    location: Tuple[int, int]
    original_annotation: str
    annotation_type: str
    predicted_annotation_logprob_dist: Dict[str, float]


def class_id_to_class(metadata: Dict[str, Vocabulary], class_id: int) -> str:
    name = metadata['annotation_vocab'].get_name_for_id(class_id)
    if metadata['annotation_vocab'].is_unk(name):
        return 'typing.Any'
    return name

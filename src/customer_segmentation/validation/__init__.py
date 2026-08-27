"""Validação de qualidade e contratos de dados (entrada e saída)."""

from customer_segmentation.validation.contracts import (
    list_input_contracts,
    list_output_contracts,
    load_contract,
)
from customer_segmentation.validation.quality import (
    assert_valid,
    validate_dataframe,
    validate_feature_table,
    validate_referential_integrity,
)

__all__ = [
    "assert_valid",
    "list_input_contracts",
    "list_output_contracts",
    "load_contract",
    "validate_dataframe",
    "validate_feature_table",
    "validate_referential_integrity",
]

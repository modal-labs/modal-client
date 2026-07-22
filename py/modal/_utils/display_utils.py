# Copyright Modal Labs 2026

from decimal import Decimal


# todo(ayush): make this locale-aware
def pretty_decimal(x: Decimal) -> str:
    """Pretty representation of a Decimal

    This formats by
    - adding "," for thousands places, and
    - trimming to at most 2 decimal places
    """

    return f"{x:,.2f}"

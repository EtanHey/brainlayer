from importlib import import_module
from typing import cast

import pytest

from tests.benchmarks.wave5_contract import LedgerFactory, OccurrenceLedger


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--wave5-ledger-factory",
        metavar="MODULE:ATTRIBUTE",
        help="Run the Wave 5 ledger contract against an external factory.",
    )


def _load_ledger_factory(reference: str) -> LedgerFactory:
    module_name, separator, attribute_name = reference.partition(":")
    if not module_name or not separator or not attribute_name:
        raise pytest.UsageError("--wave5-ledger-factory must use MODULE:ATTRIBUTE syntax")
    factory = getattr(import_module(module_name), attribute_name, None)
    if not callable(factory):
        raise pytest.UsageError(f"--wave5-ledger-factory target is not callable: {reference}")
    return cast(LedgerFactory, factory)


@pytest.fixture
def ledger_factory(request: pytest.FixtureRequest) -> LedgerFactory:
    reference = request.config.getoption("--wave5-ledger-factory")
    return _load_ledger_factory(reference) if reference else OccurrenceLedger

from importlib import import_module
from typing import cast

import pytest

from tests.benchmarks.wave5_contract import (
    CandidateProducer,
    LedgerFactory,
    OccurrenceLedger,
    oracle_candidate_producer,
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


def _load_candidate_producer(reference: str) -> CandidateProducer:
    module_name, separator, attribute_name = reference.partition(":")
    if not module_name or not separator or not attribute_name:
        raise pytest.UsageError("--wave5-candidate-producer must use MODULE:ATTRIBUTE syntax")
    producer = getattr(import_module(module_name), attribute_name, None)
    if not callable(producer):
        raise pytest.UsageError(f"--wave5-candidate-producer target is not callable: {reference}")
    return cast(CandidateProducer, producer)


@pytest.fixture
def candidate_producer(request: pytest.FixtureRequest) -> CandidateProducer:
    reference = request.config.getoption("--wave5-candidate-producer")
    return _load_candidate_producer(reference) if reference else oracle_candidate_producer

"""Fail closed if a retained local or retired public path imports the controller."""

import builtins
import importlib


def forbid_enrichment_controller(monkeypatch):
    original_import = builtins.__import__
    original_import_module = importlib.import_module

    def reject(name, fromlist=()):
        if name == "brainlayer.enrichment_controller" or (
            name == "brainlayer" and "enrichment_controller" in (fromlist or ())
        ):
            raise AssertionError("retained path imported the retired enrichment controller")

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        reject(name, fromlist)
        return original_import(name, globals, locals, fromlist, level)

    def guarded_import_module(name, package=None):
        reject(name)
        return original_import_module(name, package)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(importlib, "import_module", guarded_import_module)

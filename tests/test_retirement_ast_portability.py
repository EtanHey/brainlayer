"""Portable transport identities retain every reviewed policy decision."""

import json
import tempfile
import unittest
from pathlib import Path

from scripts.retirement_inventory import check_policy, inventory, python_sites
from scripts.retirement_policy_generate import render

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "empty_lists": "httpx.post(endpoint)",
    "nested_keywords": "httpx.post(endpoint, data=encode(payload(), mode='json'), follow_redirects=False)",
    "empty_dict": "httpx.post(endpoint, json={})",
    "expanded_keywords": "httpx.post(endpoint, *args, **kwargs)",
    "callable_reference": "send = httpx.post",
    "none_literal": "httpx.post(endpoint, data=None)",
    "lambda_arguments": "httpx.post(endpoint, data=lambda **kw: None)",
}
# Captured from the reviewed Python 3.12 serializer before this repair.
LEGACY_SHA256 = {
    "empty_lists": "90ab52e1b440f7c3b6a8668e4a1491436545208ae4b55ac4ddd7e92dd62235f7",
    "nested_keywords": "f5abfa867c8c4e0851e14ef075b14bce39efceaa3bcb80532acb169900e6b30b",
    "empty_dict": "39dde0abb601b14ab019a84d29420e46a197a451a1ed5eaad31396b219135763",
    "expanded_keywords": "c9804162142904981a593a30b7affb327c7d05185e09d214ac1bc2eb9b50787c",
    "callable_reference": "a55d4792f97374c250ec28af9c01e919e38128aeabfee0e9fbc01c9a59ed17da",
    "none_literal": "9f32dae88ccd26d3ffe7ff2d3ecd6c0f66e6f4e2ae77ef1a8899e0013f9a26c1",
    "lambda_arguments": "3776dc4790df80d6330e9b143ae4f737310d8569d369b43a3e497dd45861f96e",
}


class TestASTPortability(unittest.TestCase):
    def site(self, code):
        return next(
            site
            for site in python_sites("import httpx\n" + code, "src/brainlayer/probe.py")
            if site["kind"] != "import_binding"
        )

    def test_reviewed_ast_fingerprints(self):
        for name, code in CASES.items():
            with self.subTest(form=name):
                self.assertEqual(self.site(code)["call_sha256"], LEGACY_SHA256[name])

    def test_full_inventory_retains_every_reviewed_identity_and_purpose(self):
        policy = json.loads((ROOT / "scripts/retirement_policy.json").read_text())
        expected = {key: entry["site"] for key, entry in policy["sites"].items()}
        self.assertEqual(inventory(ROOT), expected)
        self.assertEqual(check_policy(ROOT)["unclassified"], [])
        self.assertEqual(render(ROOT), (ROOT / "scripts/retirement_policy.json").read_text())
        self.assertEqual(render(ROOT), render(ROOT))

    def test_harmless_edits_preserve_admission(self):
        code = "import httpx\nhttpx.post(endpoint, data=payload())"
        harmless = (
            '\n"""Historical documentation."""\nimport httpx\n# comment\nhttpx.post( endpoint, data = payload( ) )\n'
        )
        self.assertEqual(python_sites(code, "probe.py"), python_sites(harmless, "probe.py"))

    def test_semantic_mutations_change_identity_and_remain_unclassified(self):
        baseline = "import httpx\nsend = httpx.post\nhttpx.post(endpoint, data=payload())"
        mutations = [
            "import httpx\nsend = httpx.post\nhttpx.post(other_endpoint, data=payload())",
            "import httpx\nsend = httpx.post\nhttpx.post(endpoint, data=payload('private'))",
            "import httpx\nsend = httpx.post\nhttpx.post(endpoint, data=payload(), follow_redirects=True)",
            "import httpx\nsend = httpx.post\nhttpx.put(endpoint, data=payload())",
            "import httpx\nsend = httpx.put\nhttpx.post(endpoint, data=payload())",
        ]
        with tempfile.TemporaryDirectory(prefix="ast-portability-") as directory:
            root = Path(directory)
            source = root / "src/brainlayer/probe.py"
            source.parent.mkdir(parents=True)
            source.write_text(baseline)
            reviewed = inventory(root)
            policy = root / "scripts/retirement_policy.json"
            policy.parent.mkdir()
            policy.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "sites": {
                            key: {
                                "site": site,
                                "non_cloud_purpose": "Private synthetic admission control; never executed",
                            }
                            for key, site in reviewed.items()
                        },
                    }
                )
            )
            self.assertEqual(check_policy(root)["unclassified"], [])
            for code in mutations:
                with self.subTest(code=code):
                    source.write_text(code)
                    self.assertNotEqual(inventory(root), reviewed)
                    self.assertTrue(check_policy(root)["unclassified"])
            source.write_text(baseline)
            self.assertEqual(check_policy(root)["unclassified"], [])

    def test_callable_reference_mutation_changes_fingerprint(self):
        self.assertNotEqual(self.site("send = httpx.post"), self.site("send = httpx.put"))


if __name__ == "__main__":
    unittest.main()

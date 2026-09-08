# Source assertion and corpus verification

Status: library gate under qualification. No model is qualified by its unit tests,
and the existing backfill runner does not call it yet. It has no database writer,
retriever, scheduler or model fallback.

`verify_relation` first applies every existing backfill structural gate, then
asks a separate model call to assess the original quote in its source and every
supplied reference. The caller supplies bounded eligible windows and a durable
raw-response callback. A recording, transport or parse error stops review;
there is no synthetic empty result or correction that hides the first response.
The callback receives the primary and ordered reference windows, including
their source IDs, origins, timestamps and original text, even for invalid output.
Transport responses may be text or UTF-8 bytes. The callback retains original
bytes (freezing bytearrays) before decoding; parsing and hashing use the same text.
Missing or unrecognized source classes fail before inference. Ordinary CLI,
subagent and fleet-coordination evidence remains eligible.
Desktop evidence and all memory-reader classes from the shared ingestion
constant are excluded from both primary and reference windows before inference.

`depends_on` means a software/runtime requirement: necessary code, a service or
an artifact. An adopted mandatory policy is a real `governed_by` relationship.
When the graph lacks its policy target, `GOVERNED_BY_UNBOUND` preserves the policy
quote without creating an ID or changing the proposed service endpoint. The
legacy source and relation types remain unchanged.
Both supporting and policy verdicts must retain the original proposal quote.
In v1, `mandatory_policy` returns before reference chronology is assessed: a
contradicted or expired policy can still be UNBOUND. This status confirms neither
policy validity nor current applicability, and remains unauthorized for writes.

Outcomes distinguish a corroborated source assertion, rejection, unresolved
evidence, policy awaiting binding, and a relationship that subsequently ended.
All outcomes explicitly leave current graph truth unverified and authorize no
canonical write. A dated later correction and a legitimate ending are distinct;
an earlier denial cannot automatically refute a later assertion. Uncertain or
missing dates remain unresolved. Observation timestamps are not effective dates.

The caller must resolve underlying evidence origins, including forwarded or
summarized origins; different session IDs alone do not establish independence.
Quotes copied from any primary-source span, same-origin records and unknown
origins cannot corroborate. This conservative copy check can also withhold
independent accounts that happen to use identical wording.
Short or common reviewer quotes amplify this loss. During qualification, inspect
this filter before attributing low independent-support counts to retrieval.
No search hits is UNKNOWN. The model must review every retrieved record in order,
but this does not prove retrieval recall or absence of unreturned contradictions.
Corroboration requires known independent support and no unresolved supplied
evidence. The model still makes semantic judgments; fabricated interpretations
of exact quotes are caught by qualification, not by pretending structure proves
meaning. A model can fail this gate's gold set even when these unit tests pass.

The original frozen extraction FAIL remains immutable. Requalification uses a
new round with explicit disclosure of exposed sources and a frozen fresh holdout.
Raw extraction, structural acceptance, semantic verification, candidate recall
and reference retrieval recall have separate denominators. A verifier cannot
hide a failed raw-extraction bar. The permanent Sol/Astra bulk and delta worker
must use the same qualified gate, retain provenance/configuration fingerprints,
and stop before writes on a failed canary. A canary pass is not that night's
measured precision. Production integration, policy binding and scheduling remain
separate work; no canonical run is authorized by this library slice.

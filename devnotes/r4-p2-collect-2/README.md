# `r4/p2-modelx-collect-2`: one commit on 4f21476

This session may push only `r4/p2-modelx-collect`, so the branch is delivered
as a patch. `git checkout -b r4/p2-modelx-collect-2 4f21476 && git am
devnotes/r4-p2-collect-2/*.patch` gives commit 8219b21's tree.

Behaviour: an ItemSpace has no `collect` (or `_mx_collect`) in live and
exported models alike: reading it raises AttributeError, `hasattr` is False.
Suite: 3.13 1963 passed, 7 skipped; 3.14t 1963 passed, 6 skipped; the one
failure in both is upstream's `lifelib/test_deserializer.py`.

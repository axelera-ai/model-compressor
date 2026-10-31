"""Define the API to validate that a graph has been fully adapted.

Where `validate.validate` checks a *graph-global, topological* invariant on an ``fx.GraphModule``
(canonical form), and `validate_operators` checks *operator-local, static* invariants on an
operator's schema, this module checks the *post-condition of the `adapt` pass* on the adapted
``networkx.MultiDiGraph``: that every data-type mismatch has been reconciled into an explicit
``mark_cast`` marker, so the graph is data-type-coherent.

`adapt` is non-destructive: it preserves both the ``source`` (original) and ``chosen`` (optimized)
annotations, splicing ``mark_cast`` markers that are *identities in the source view* and *casts in
the chosen view*. So the graph carries two data-type-coherent programs at once, and a consumer picks
the view it needs - learning-time reads ``chosen`` (the fake-quantized program), compile-/inference-
time reads ``source`` (the true program). The post-condition is therefore **dual arc-coherence**:
along every arc the producer's output data type equals the consumer's input data type, checked
*independently in each view*. Coherence of the ``chosen`` view means every required cast was inserted;
coherence of the ``source`` view means the original program is intact and recoverable (the markers
are transparent there). Arcs incident to schema-less nodes (``"placeholder"`` / ``"get_attr"`` /
``"output"``) carry no data type and are skipped.

Concretely, this is meant to be executed *after* `adapt.adapt`.

"""

import importlib

import networkx as nx

# ``passes.adapt.__init__`` re-exports the ``adapt`` *function*, which shadows the ``adapt`` submodule
# as an attribute of the package; ``import ...adapt.adapt as _adapt`` would bind the function, not the
# module. ``importlib.import_module`` returns the submodule regardless of the shadow.
_adapt = importlib.import_module("model_compressor.passes.adapt")


def validate_adapted(graph: nx.MultiDiGraph) -> None:
    """Verify that ``graph`` satisfies the post-condition of the `adapt` pass: dual arc-coherence.

    Along every arc, the producer's output data type must equal the consumer's input data type in
    *both* the ``source`` view (markers are identities, so the original program is intact) and the
    ``chosen`` view (markers are casts, so every required cast was inserted). Arcs incident to
    schema-less nodes (``"placeholder"`` / ``"get_attr"`` / ``"output"``) carry no data type and are
    skipped.

    Args:
        graph: An adapted ``networkx.MultiDiGraph`` (the output of `adapt.adapt`), whose nodes carry
            ``source`` and ``chosen`` schema annotations.

    Raises:
        ValueError: an arc is data-type-incoherent in the ``source`` view (the original program was
            corrupted) or the ``chosen`` view (a required cast is missing).

    """
    for view in ("source", "chosen"):
        for producer, consumer, (output_port, input_port) in graph.edges(keys=True):
            producer_schema = getattr(graph.nodes[producer]["annotation"], view)
            consumer_schema = getattr(graph.nodes[consumer]["annotation"], view)
            if (producer_schema is None) or (consumer_schema is None):
                continue  # a schema-less endpoint ("placeholder" / "get_attr" / "output")
            produced = _adapt.get_output_type(producer_schema, output_port)
            expected = _adapt.get_input_type(consumer_schema, input_port)
            if produced != expected:
                raise ValueError(
                    f"validate_adapted: arc {producer!r} -> {consumer!r} is data-type-incoherent in "
                    f"the {view!r} view - producer emits {produced!r} at output port {output_port!r} "
                    f"but consumer expects {expected!r} at input port {input_port!r}"
                )

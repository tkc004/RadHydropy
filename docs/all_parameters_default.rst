Complete Nested Parameter Template
===================================

The file below is the complete annotated template for the nested RadHydropy
runtime configuration. It lists the available ``par`` groups, default values,
short explanations, and supported options where a parameter has multiple
modes.

Use it as a reference when creating an example configuration. Keep runtime
parameters under their appropriate ``par.<group>`` section; put initial-
condition inputs under ``initial_condition`` and plotting or comparison
settings under ``example``.

.. literalinclude:: ../example/all_parameters_default.yaml
   :language: yaml

The template contains placeholders such as ``null`` and example paths. Replace
those values for a concrete run and validate the resulting file with the
normal nested example loader.

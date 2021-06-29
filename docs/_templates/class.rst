:mod:`{{module}}`.{{objname}}
{{ underline }}==============

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   .. automethod:: fit
   .. automethod:: refit
   .. automethod:: fit_transform
   .. automethod:: transform
   {% endblock %}

.. raw:: html

    <div style='clear:both'></div>

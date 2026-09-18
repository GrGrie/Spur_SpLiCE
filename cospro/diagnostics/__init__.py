"""Quality metrics and dashboards for CoSpRo concept groups and teacher graphs.

Two tiers keep the label-free protocol auditable:

- label-free metrics read only artifacts and the SpLiCE cache;
- post-hoc metrics also read the class label ``y`` and the spurious attribute ``a``; every one of
  them receives its labels from :mod:`cospro.diagnostics.labels`.

Post-hoc metrics explain the method and support manual selection. Automatic tuning uses the
label-free tier only.
"""

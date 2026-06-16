# Result Consistency Notes

## Results that align with the final report

The MCDA and VRS DEA results align with the final report:

- Supplier H ranks first in both Product Quality and Customer Service.
- Supplier G has the weakest Product Quality score.
- Supplier F has the weakest Customer Service score.
- Suppliers A, B, D, E, G, H, I and K are VRS-efficient.
- Suppliers C, F, J and L are VRS-inefficient and require improvement targets.

## Cross-efficiency note

The Python implementation calculates cross-efficiency using a CCR multiplier model. Depending on output scaling, solver tolerance and multiplier-weight selection, the exact cross-efficiency values may differ from the final written report.

For portfolio use, the project focuses on the stable business conclusion rather than exact cross-efficiency replication:

- Supplier H remains a strong benchmark.
- Supplier B remains a strong supplier.
- Supplier G requires careful interpretation because its efficiency is partly driven by low input values rather than strong Product Quality or Customer Service outcomes.

## Total purchase scaling

The DEA output matrix uses total purchase values scaled to thousands. This matches the scale shown in the report appendix and improves numerical stability. Scaling an entire output column by a constant does not change the main VRS efficient/inefficient classification.

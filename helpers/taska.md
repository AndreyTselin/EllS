# Task Log

## 2026-02-24
- Added a PyQt UI for ellipsometry modeling with parameter controls and substrate file loading.
- Switched plotting to interactive `pyqtgraph` views (zoom/pan, linked X-axis, legends).
- Fixed core modeling issues in `IsoLayer`/`EllModel` related to interpolation, matrix generation, and plot save paths.
- Added expanded automated tests for different layer models and material stack structures.

## 2025-11-28
- Created `AirLayer` as a child of `IsoLayer` with semi-infinite thickness (`None`) and `n=1`, `k=0`.
- Added `AirLayer` method for `D_0` matrix handling in total transfer matrix workflow.
 

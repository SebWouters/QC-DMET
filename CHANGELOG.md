# Changelog

## [1.0.0] - 2026-03-28
### Added
- New `FCI` solver using PySCF's `pyscf.fci` module
- New `DMRG` solver using [Block2](https://github.com/block-hczhai/block2-preview)

### Changed
- Ported entire codebase from Python 2.7 to Python 3.10
- Updated PySCF API calls for PySCF 2.0+ compatibility
- Centralized all helper scripts and backend solvers into the `src/solvers/` subdirectory
- Replaced depreciated CheMPS2 as default `DMRG` solver with Block2 ([CheMPS2](https://github.com/SebWouters/CheMPS2) remains available as a non-default option).
- `ED` method now calls PySCF FCI

### Removed
- Dependency on deprecated `pyscf.future` module
- Dependency on deprecated `pyscf.tools.rhf_newtonraphson`
- Dependency on deprecated `pyscf.tools.localizer`

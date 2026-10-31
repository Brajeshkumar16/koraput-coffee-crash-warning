# Data and code: "Coffee yield crash prediction by developing Machine Learning models using Coffee Dataset of Koraput India"

Revised manuscript submitted to *Discover Sustainability* (submission ID d99a7a07-abcf-4c43-9e90-3a8dbd821ddd).

## Access and licence

All files are openly available. Downloading them needs no account, registration or request to the authors.
Data files: CC BY 4.0 (`LICENSE-DATA.md`). Code: MIT licence (`LICENSE`). To cite the data and code, see `CITATION.cff`.

## Files

| File | What it is |
|---|---|
| `data/source_spreadsheet_as_supplied.xlsx` | The spreadsheet as the authors received it (file name "Koraput Coffee Data Set- SO Agriculture Koraput .xlsx"), 472 rows, unchanged. Up to 7 rows per block and year. |
| `data/koraput_yearly_means.csv` | 182 block-years: for each block and year present in the source, the mean of its rows. |
| `data/koraput_panel_analysis_file_flagged.csv` | The 200-row analysis file (10 blocks x 2006-2025) used in the paper, with three added columns: `filled` (1 = no row for this block-year in the source), `fill_method`, `copied_row`. |
| `data/coffee_board_odisha.csv` | Coffee Board of India figures for the whole of Odisha (final production, planted area), each with the edition of the Board's public *Database on Coffee* it was read from. Used only for the magnitude comparison in Table 2. |
| `data/original_submission/` | Files of the original submission: the analysis file it read (`final_clustered_data.csv`, which also carries the cluster label of the companion study), two of its output tables and its Fig. 1 schematic. |
| `code/` | `ewcore.py`, `reanalysis.py`, `experiments2.py`, `provenance.py`, `figures.py`, `summarise.py`: reproduce every table and figure of the revised paper. `requirements.txt` lists the package versions used. |
| `code/original_submission/earlywarning_v2_koraput.py` | The code of the original submission, unchanged. It reads `final_clustered_data.csv` from the folder set in `BASE` (line 58). |

## How the analysis file was derived

1. For each block and year, production, export, area and farmers are the **means** of the source rows for that block-year (1 to 7 rows). This is why many values are not integers.
2. Eighteen block-years have no source row. Fifteen were filled by linear interpolation between the neighbouring years; Semiliguda 2025 is a copy of 2024; Jeypore 2025 and Laxmipur 2006 are edge years whose filling rule could not be identified. All are flagged.
3. Yield = production / area; export ratio = export / production; productivity per farmer = production / farmers.

## Known problems (stated in the paper)

- The product measured (fresh cherry, parchment or clean coffee) and the units could not be confirmed against original returns.
- The median district total is about 39 times the Coffee Board of India's figure for all of Odisha, on about 1.2 times the area, so yields are far above published figures.
- In the source rows, production, area, export share and farmers per hectare are spread evenly between round limits, production is unrelated to area (r = 0.04), and block and year explain no more variance than chance.
- The values should therefore not be used as measurements of coffee production in Koraput.

## Reproduce

Python 3.13 with the packages in `code/requirements.txt` (`pip install -r code/requirements.txt`). Run from the `code` folder:

```
python reanalysis.py      # outputs/tables/T01..T14 (about 20 minutes)
python experiments2.py    # outputs/tables/X01..X04: origin-referenced labels, persistence, equivalence, detectability
python provenance.py      # outputs/tables/P01..P05: derivation, filled block-years, source distributions, magnitudes
python figures.py         # outputs/figures/
python summarise.py       # outputs/results_summary.md
```

Results appear in `code/outputs/`. The random seed is fixed at 42, so repeated runs give identical tables.

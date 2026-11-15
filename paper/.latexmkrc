# Same layout as presentations/.latexmkrc: pdf beside the .tex, everything else in build/.
$out_dir = '.';
$aux_dir = 'build';
$pdf_mode = 1;
# BibTeX runs inside build/ under MiKTeX and cannot see a .bib beside the .tex.
$bibtex = 'bibtex --include-directory=.. %O %S';

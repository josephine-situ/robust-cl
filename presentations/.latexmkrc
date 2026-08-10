# The .pdf stays in presentations/ next to its .tex (so an editor's viewer finds
# it with no config, and .synctex.gz -- which must sit beside the pdf for
# click-to-source -- comes with it). Every other artifact goes to build/.
#
# latexmk reads this automatically for any run whose cwd is this directory --
# including editor compile-on-save (VS Code LaTeX Workshop's default recipe),
# which is what kept dropping .fls/.fdb_latexmk/.aux at the top level.
# NOTE: a distinct $aux_dir needs -aux-directory, which is MiKTeX-only.
$out_dir = '.';
$aux_dir = 'build';
$pdf_mode = 1;    # pdflatex

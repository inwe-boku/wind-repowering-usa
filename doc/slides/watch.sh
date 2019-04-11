#!/bin/bash
xdg-open slides.pdf
while true; do
    inotifywait -e modify . &&  pdflatex -interaction nonstopmode -halt-on-error -file-line-error slides.tex
done

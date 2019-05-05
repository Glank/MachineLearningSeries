paper.pdf: paper.tex
	pdflatex paper
	-bibtex paper
	pdflatex paper
	pdflatex paper

clean:
	-rm paper.aux
	-rm paper.log
	-rm paper.pdf
	-rm paper.bbl
	-rm paper.blg

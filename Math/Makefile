book.pdf: book.tex book.bib
	pdflatex book
	-bibtex book
	pdflatex book
	pdflatex book

clean:
	-rm book.aux
	-rm book.log
	-rm book.pdf
	-rm book.bbl
	-rm book.blg
	-rm book.toc

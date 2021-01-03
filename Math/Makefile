book.pdf: book.tex book.bib
	pdflatex book
	-bibtex book
	pdflatex book
	pdflatex book

spellcheck: book.tex
	aspell -p ./aspell_dict.txt -t -c book.tex

clean:
	-rm book.aux
	-rm book.log
	-rm book.pdf
	-rm book.bbl
	-rm book.blg
	-rm book.toc
	-rm book.out
	-rm book.tex.bak

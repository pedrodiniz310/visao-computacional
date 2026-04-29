import fitz
pdfs = ['9 - Segmentação de Imagens.pdf', '10a - Extração de Características.pdf', '7b - Filtros.pdf', '7c - Morfologia.pdf']
for f in pdfs:
    doc = fitz.open(f)
    text = '\n'.join(p.get_text() for p in doc)
    print(f'=== {f} ===')
    print(text[:4000])
    print()

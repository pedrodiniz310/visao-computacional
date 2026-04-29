"""
Extração de Texto de PDFs
Varre o diretório atual em busca de todos os arquivos .pdf,
extraí o texto de cada página usando a biblioteca PyMuPDF (fitz)
e imprime o conteúdo no terminal.
"""

import fitz   # PyMuPDF: biblioteca para leitura e manipulação de PDFs
import os

# Diretório base: mesmo diretório onde este script está localizado
# os.path.dirname(__file__) retorna o caminho do arquivo atual
pdf_dir = os.path.dirname(__file__)

# Lista todos os arquivos .pdf no diretório, ordenados alfabética/numericamente
# os.listdir() retorna todos os nomes de arquivo (sem caminho completo)
# sorted() garante ordem consistente de processamento
pdfs = sorted([f for f in os.listdir(pdf_dir) if f.endswith('.pdf')])

# Processa cada PDF encontrado
for pdf_name in pdfs:
    # Constrói o caminho absoluto do arquivo PDF
    path = os.path.join(pdf_dir, pdf_name)

    # fitz.open() abre o documento PDF e retorna um objeto Document
    # Esse objeto permite iterar sobre as páginas como uma lista
    doc = fitz.open(path)

    # Cabeçalho separador para identificar o PDF no terminal
    print(f'\n{"="*80}')
    print(f'PDF: {pdf_name}')
    print(f'{"="*80}')

    # Itera sobre cada página do documento
    # enumerate(doc) fornece (número_da_pagina, objeto_pagina)
    for i, page in enumerate(doc):
        # page.get_text() extrai o conteúdo de texto da página como string
        # Retorna string vazia se a página for puramente gráfica (imagem escaneada)
        text = page.get_text()

        # Só imprime páginas que têm texto real (ignora páginas em branco/gráficas)
        if text.strip():
            print(f'\n--- Página {i+1} ---')  # i é base-0, por isso +1
            print(text)

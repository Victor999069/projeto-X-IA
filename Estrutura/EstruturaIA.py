import tkinter as tk
from tkinter import filedialog, Text
from transformers import pipeline, GPTNeoForCausalLM, GPT2Tokenizer
import PyPDF2
import torch
from docx import Document
import os

# qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-2.7B").to(device)
tokenize = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-2.7B")

# def carregar_arquivo():
#     global contexto_texto
#     filename = filedialog.askopenfilename(
#         initialdir="/",
#         title="Selecione o Arquivo",
#         filetypes=(("Arquivos PDF", "*.pdf"), ("Documentos do Word", "*.docx"), ("Planilhas Excel", "*.xlsx"))
#     )
#     if filename:
#         texto_box.insert(tk.END, f"Arquivo carregado: {filename}\n")
#         if filename.endswith(".pdf"):
#             contexto_texto = ler_pdf(filename)
#         elif filename.endswith(".docx"):
#             contexto_texto = ler_docx(filename)
#         else:
#             contexto_texto = "Formato de arquivo não suportado ainda!"
            
#         texto_box.insert(tk.END, "Texto carregado:\n")
#         texto_box.insert(tk.END, contexto_texto[:500] + "...\n\n")
        
# def ler_pdf(caminho):
#     texto = ""
#     with open(caminho, "rb") as file:
#         leitor_pdf = PyPDF2.PdfFileReader(file)
#         for pagina in range(leitor_pdf.numPages):
#             texto += leitor_pdf.getPage(pagina).extract_text()
#     return texto

# def ler_docx(caminho):
#     doc = Document(caminho)
#     texto = "\n".join([paragrafo.text for paragrafo in doc.paragraphs])
#     return texto
def enviar_pergunta():
    pergunta = pergunta_entry.get()
    
    texto_box.insert(tk.END, "Gerando resposta...\n")
    root.update_idletasks()  # Atualizar a interface

    try:
        resposta = responder_pergunta_geral(pergunta)
    except Exception as e:
        resposta = f"Erro ao gerar resposta: {e}"

    texto_box.insert(tk.END, f"Pergunta: {pergunta}\n")
    texto_box.insert(tk.END, f"Resposta: {resposta}\n")
    texto_box.insert(tk.END, "-"*50 + "\n")


# def enviar_pergunta():
#     pergunta = pergunta_entry.get()
#     if contexto_texto:
#         resposta = responder_pergunta(pergunta)
#     else:
#         resposta = responder_pergunta_geral(pergunta)
#     texto_box.insert(tk.END, f"Pergunta: {pergunta}\n")
#     texto_box.insert(tk.END, f"Resposta: {resposta}\n")

# def responder_pergunta(pergunta):
#     resposta = qa_pipeline({
#         'question': pergunta,
#         'context': contexto_texto
#     })
#     return resposta['answer']

def responder_pergunta_geral(pergunta):
    inputs = tokenize.encode(pergunta, return_tensors='pt').to(device)
    attention_mask = torch.ones(inputs.shape, dtype=torch.long).to(device)

    # Gerar a resposta
    outputs = model.generate(
        inputs, 
        attention_mask=attention_mask,
        max_length=50, 
        num_return_sequences=1, 
        top_k=50,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenize.eos_token_id
    )
    
    # Decodificar a resposta
    resposta = tokenize.decode(outputs[0], skip_special_tokens=True)
    return resposta

root = tk.Tk()
root.title("Assistente Local para Empresas")
root.geometry("600x500")

# Frame para o título e descrição
frame_topo = tk.Frame(root)
frame_topo.pack(pady=10)

titulo_label = tk.Label(frame_topo, text="Assistente Local", font=("Arial", 16, "bold"))
titulo_label.pack()

descricao_label = tk.Label(frame_topo, text="Este assistente ajuda você a se familiarizar com a empresa e responder perguntas.")
descricao_label.pack()

# Frame para os botões
frame_botoes = tk.Frame(root)
frame_botoes.pack(pady=10)

# botao_carregar = tk.Button(frame_botoes, text="Carregar Arquivo", padx=10, pady=5, command=carregar_arquivo)
# botao_carregar.grid(row=0, column=0, padx=5)

botao_enviar = tk.Button(frame_botoes, text="Enviar Pergunta", padx=10, pady=5, command=enviar_pergunta)
botao_enviar.grid(row=0, column=1, padx=5)

# Frame para o campo de entrada e área de texto
frame_conteudo = tk.Frame(root)
frame_conteudo.pack(pady=10, expand=True, fill="both")

pergunta_entry = tk.Entry(frame_conteudo, width=50)
pergunta_entry.pack(pady=5)

texto_box = tk.Text(frame_conteudo, wrap="word", bg="lightgrey", height=15)
texto_box.pack(padx=10, pady=10, expand=True, fill="both")

# Adicionando barra de rolagem para a área de texto
scrollbar = tk.Scrollbar(frame_conteudo, command=texto_box.yview)
scrollbar.pack(side="right", fill="y")
texto_box.config(yscrollcommand=scrollbar.set)

# Inicializando o contexto vazio
contexto_texto = ""

# Loop da interface
root.mainloop()
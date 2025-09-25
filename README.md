# Otimização do Processamento de Imagens de Ovos de Galinha  
### Interface Gráfica Interativa + Paralelismo em Python (TCC)

> **Resumo:** Este repositório contém o código e a documentação do meu TCC, que automatiza a **detecção e análise de ovos** em imagens, e **acelera** o processamento usando **paralelismo por processos**. Inclui uma **GUI** para operar o pipeline.

---

## ✨ Destaques

- **Paralelismo por processos** (`concurrent.futures.ProcessPoolExecutor`) para acelerar etapas **CPU-bound**.  
- **Separação clara**: **CPU em paralelo** e **I/O sequencial** (evita disputa de disco).  
- **Medição de desempenho** com `time.perf_counter()` + decorator `@timing`.  
- **Resultados reprodutíveis**: tabelas comparando **sem** e **com** paralelismo em máquinas diferentes.  
- **GUI** para selecionar imagens/pastas e acompanhar o progresso. 

---

## 🏗️ Arquitetura do Pipeline

Entrada (imagens) ─▶ detecção de ovos (findeggs)
└▶ recortes por ovo (subimagens em RAM)
▼
jobs = [(id, bytes, shape, l0..c1, pixFactor), ...]
▼
ProcessPoolExecutor (CPU-bound em paralelo)
└─ cpu_process_egg(job)
└─ process(...)
├─ segmentação (HSV/contorno)
├─ medidas A–D + pontos
└─ séries x/y para ajuste
▼
resultados em memória (sem I/O no pool)
▼
I/O sequencial (seguro e previsível)
├─ salvar PNGs / mosaico final
├─ Relatorio.dad (medidas)
├─ gráficos de ajuste (matplotlib)
└─ planilha Excel (openpyxl)


---

## 🧰 Requisitos

- **Python 3.10+**
- Bibliotecas principais:
  - NumPy, SciPy, **scikit-image**, **imageio**
  - matplotlib, pandas
  - openpyxl, Pillow
  - (GUI atual) Tkinter (builtin)
  - opencv-python

**`requirements.txt` (exemplo):**
```txt
annotated-types==0.7.0
cloudpickle==3.1.1
contourpy==1.3.1
cycler==0.12.1
et_xmlfile==2.0.0
fonttools==4.56.0
gprof2dot==2025.4.14
imageio==2.37.0
Jinja2==3.1.6
kiwisolver==1.4.8
lazy_loader==0.4
markdown-it-py==3.0.0
MarkupSafe==3.0.2
matplotlib==3.10.1
mdurl==0.1.2
networkx==3.4.2
numpy==2.2.3
nvidia-ml-py==12.570.86
opencv-python==4.11.0.86
openpyxl==3.1.5
packaging==24.2
pandas==2.2.3
pillow==11.1.0
psutil==7.0.0
py-spy==0.4.0
pydantic==2.11.3
pydantic_core==2.33.1
Pygments==2.19.1
pyparsing==3.2.1
python-dateutil==2.9.0.post0
pytz==2025.1
rich==14.0.0
scalene==1.5.51
scikit-image==0.25.2
scipy==1.15.2
six==1.17.0
snakeviz==2.2.2
tifffile==2025.2.18
tornado==6.4.2
typing-inspection==0.4.0
typing_extensions==4.13.2
tzdata==2025.1
urlopen==1.0.0
```

---
## 🚀 Instalação

```bash
git clone https://github.com/vinisilvf/tcc-ovos.git
cd tcc-ovos
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate
pip install -r requirements.txt
```

---

## ⚙️ Paralelismo (como funciona)

- Jobs: 1 ovo = 1 job → (id, subimagem.tobytes(), shape, l0..c1, pixFactor)
- Executor: ProcessPoolExecutor(max_workers=cpu_count()-1)
- No worker (cpu_process_egg): reconstrói imagem (np.frombuffer), segmenta, mede A–D, gera x/y, sem I/O.
- Pós-pool: I/O sequencial (PNGs, .dad, gráficos, Excel, mosaico).

### Medindo Tempo:

```py
import time

def timing(fn):
    def w(*a, **k):
        t0 = time.perf_counter()
        r = fn(*a, **k)
        t1 = time.perf_counter()
        print(f"[TIMER] {fn.__name__}: {t1 - t0:.3f}s")
        return r
    return w
```

---

## 📈 Resultados (exemplo de leitura)

- i5-6200U (2C/4T): tempo médio caiu de ~155,6 s → ~80,4 s (-52%).
- i3-N305 (8C): tempo médio caiu de ~82,8 s → ~57,1 s (-31%).
- Interpretação: ganhos consistentes; mais núcleos reduzem tempo absoluto (i3), enquanto em hardware mais modesto o ganho relativo pode ser maior (i5). O I/O sequencial define o limite do speedup.

--- 

## GUI - Interface Grafica 

<img width="377" height="552" alt="Menu2" src="https://github.com/user-attachments/assets/5cc03f5c-8276-4ecd-8d43-c4e09b888bce" />

## Exemplo de cartela utilizada para analise do processamento paralelo

<img width="800" height="800" alt="Captura de tela 2025-07-14 164457" src="https://github.com/user-attachments/assets/2e1cca7e-816c-4752-8b38-2abc625e9731" />

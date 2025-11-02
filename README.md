# Estimativa de Esforço (Story Points) com Deep-SE e LLMs

Este repositório reúne os cadernos (Jupyter Notebooks) usados no estudo/artigo sobre estimativa de esforço de histórias de usuário (story points) combinando:

- EDA do conjunto de dados Deep-SE
- Fine-tuning de modelos baseados em Transformers (DistilBERT e Gemma 3 4B)
- Quantização para acelerar inferência
- Experimentos com LLMs em modo zero-shot e few-shot
- Geração de gráficos e tabelas para o artigo

O objetivo é facilitar a reprodução dos experimentos e a geração das figuras do trabalho.

> Observação: alguns notebooks têm nomes muito similares (variações com hífen e sublinhado, `v1` e `v2`). Mantivemos todos para preservar o histórico dos experimentos; use a versão mais recente quando houver duplicidade.

## Requisitos

- Python 3.9+ (recomendado 3.10)
- Jupyter Notebook ou JupyterLab
- Pacotes principais:
	- numpy, pandas, scikit-learn, matplotlib, seaborn, plotly
	- torch, transformers, datasets, accelerate, evaluate
	- opcional (para técnicas específicas): peft, bitsandbytes
- GPU CUDA (opcional, mas recomendável para fine-tuning e LLMs)

Notas de ambiente (Windows):
- Quantização com `bitsandbytes` e algumas rotinas de fine-tuning de LLMs podem exigir Linux/WSL e drivers CUDA compatíveis.
- Caso encontre limitações no Windows, considere executar via WSL2/Ubuntu ou um ambiente em nuvem com GPU (ex.: Colab, SageMaker, etc.).


## Configuração do ambiente (exemplos)

Você pode usar conda ou venv. Abaixo, exemplos em PowerShell.

### Via conda

```powershell
conda create -n storypoints python=3.10 -y
conda activate storypoints

pip install -U pip
pip install jupyter numpy pandas scikit-learn matplotlib seaborn plotly
pip install torch torchvision torchaudio
pip install transformers datasets accelerate evaluate
# opcionais
pip install peft bitsandbytes
```

### Via venv (nativo do Python)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install -U pip
pip install jupyter numpy pandas scikit-learn matplotlib seaborn plotly
pip install torch torchvision torchaudio
pip install transformers datasets accelerate evaluate
# opcionais
pip install peft bitsandbytes
```

Se for usar GPU, instale as versões de PyTorch/Transformers compatíveis com sua versão do driver CUDA. Consulte a documentação oficial dessas bibliotecas.

## Estrutura do repositório

Principais notebooks e propósito:

- `deep_SE_EDA.ipynb` e `deep-SE-EDA.ipynb`
	- Análise exploratória do dataset Deep-SE: distribuição de story points, limpeza básica, gráficos descritivos.
- `fine_tuning_story_point distilbert-base-uncased v1.ipynb` e `fine_tuning_story_point distilbert-base-uncased v2.ipynb`
	- Fine-tuning supervisionado do DistilBERT para regressão de story points (v2 geralmente incorpora ajustes/otimizações).
- `distilbert_uncase_story_point_quantization.ipynb` e `distilbert-uncase-story-point_quantization.ipynb`
	- Quantização do modelo DistilBERT (por exemplo, 8-bit) para acelerar a inferência com baixo impacto em acurácia.
- `story fine tuning Gemma3_(4B) v1.ipynb` e `story fine tuning Gemma3_(4B) v2.ipynb`
	- Experimentos de fine-tuning com a família Gemma 3 (4B) aplicados à tarefa de story points.
- `Experimento LLM Estimativa de esforço.ipynb`
	- Protocolos de avaliação com LLMs (prompting few/zero-shot, formatos de entrada/saída, métricas).
- `experimento LLM Estimativa de esforço Zero Shot.ipynb`
	- Foco em zero-shot (sem exemplos), avaliando a generalização do LLM com instruções adequadas.
- `geração dos graficos no artigo.ipynb`
	- Scripts para gerar figuras e tabelas finais do artigo com base nos resultados salvos.

## Dados (Deep-SE)

- Os notebooks de EDA assumem o uso do conjunto de dados Deep-SE (histórias de usuário e seus story points). Coloque os arquivos em uma pasta dedicada (ex.: `data/`).
- Se necessário, ajuste os caminhos de entrada nos notebooks para apontar para seus arquivos locais.
- Garanta consistência de encoding (UTF-8) e separador (por exemplo, CSV com vírgula).

## Como executar

1) Configure o ambiente (seção acima) e abra o VS Code.
2) Abra o notebook desejado e execute célula a célula.
3) Se estiver usando GPU, selecione o kernel Python correto (da conda/venv) e confirme que o PyTorch detecta a GPU.

Sugestão de ordem para reproduzir o estudo completo:

1. EDA do dataset: `deep_SE_EDA.ipynb` (ou a variante com hífen)
2. Fine-tuning DistilBERT: `fine_tuning_story_point distilbert-base-uncased v2.ipynb`
3. Quantização para inferência: `distilbert_uncase_story_point_quantization.ipynb`
4. Experimentos com LLMs (zero/few-shot): `experimento LLM Estimativa de esforço Zero Shot.ipynb` e `Experimento LLM Estimativa de esforço.ipynb`
5. Geração dos gráficos do artigo: `geração dos graficos no artigo.ipynb`

## Métricas avaliadas

As principais métricas de regressão utilizadas para story points incluem:

- MAE: $\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$
- RMSE: $\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$
- R²: $R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$

Opcionalmente, dependendo do protocolo, também podem ser reportados MAPE ou “acurácia dentro de um delta” (ex.: % de predições dentro de ±1 ponto da verdade).

## Dicas e problemas comuns

- Se o kernel reiniciar durante o fine-tuning, reduza o tamanho do batch ou ative gradiente acumulado.
- Para quantização 8-bit/4-bit, valide compatibilidade de hardware e bibliotecas (pode exigir Linux/WSL).
- Em LLMs zero-shot, a formulação do prompt é crítica. Guarde versões de prompts que funcionaram melhor e fixe sementes aleatórias para reprodutibilidade.
- Salve os checkpoints e resultados (CSV/JSON) em pastas versionadas (`outputs/`, `runs/`) para alimentar o notebook de gráficos.

## Resultados e figuras

- Use `geração dos graficos no artigo.ipynb` para consolidar métricas e produzir figuras finais do artigo.
- Garanta que os arquivos de resultados apontados no notebook existam (por exemplo, métricas exportadas dos experimentos anteriores).

## Contribuição

Sinta-se à vontade para abrir issues e PRs com correções, melhorias de documentação ou novos experimentos.

## Licença

Nenhuma licença explícita foi incluída no repositório até o momento. Se você pretende reutilizar este trabalho, verifique com os autores/mantenedores ou adicione uma licença apropriada.

## Citação

Se este repositório for útil para sua pesquisa ou prática, por favor cite-o:

```
@inproceedings{Neo2025SBES,
	title        = {{Estimativa de Esforço em Story Points a partir de User Stories com Large Language Models}},
	author       = {Neo, Giseldo da Silva and Neo, Alana Viana Borges da Silva and Moura, Jos{\'{e}} Ant{\~{a}}o Beltr{\~{a}}o},
	year         = 2025,
	booktitle    = {Anais do XXXIX Simpósio Brasileiro de Engenharia de Software},
	location     = {Recife/PE},
	publisher    = {SBC},
	address      = {Porto Alegre, RS, Brasil},
	pages        = {720--726},
	doi          = {https://doi.org/10.5753/sbes.2025.11121},
	issn         = {2833-0633},
	url          = {https://sol.sbc.org.br/index.php/sbes/article/view/37050}
}

---

Contato e dúvidas: abra uma issue no GitHub ou email para giseldo@gmail.com.
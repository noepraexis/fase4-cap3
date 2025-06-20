# Guia de Preparação do Ambiente - Sistema Schierke

Este documento fornece instruções detalhadas para configurar o ambiente de desenvolvimento do **Sistema Schierke** para classificação automatizada de grãos de trigo.

## 📋 Pré-requisitos do Sistema

### Requisitos Mínimos
- **Sistema Operacional**: Windows 10+, macOS 10.14+, ou Linux (Ubuntu 18.04+)
- **Python**: 3.8 ou superior (recomendado 3.9-3.11)
- **RAM**: 4GB mínimo (8GB recomendado)
- **Espaço em disco**: 2GB livres
- **Git**: Para controle de versão

### Verificação dos Pré-requisitos

```bash
# Verificar versão do Python
python --version
# ou
python3 --version

# Verificar se pip está instalado
pip --version

# Verificar Git
git --version
```

**Saída esperada:**
```
Python 3.9.16
pip 23.0.1
git version 2.39.2
```

## 🔧 Configuração do Ambiente Virtual

### 1. Clone do Repositório

```bash
# Clonar o repositório
git clone https://github.com/noepraexis/fase4-cap3.git

# Navegar para o diretório do projeto
cd fase4-cap3

# Verificar estrutura do projeto
ls -la
```

### 2. Criação do Ambiente Virtual

#### Windows (PowerShell/CMD)
```powershell
# Criar ambiente virtual
python -m venv venv

# Ativar ambiente virtual
venv\Scripts\activate

# Verificar ativação (prompt deve mostrar (venv))
```

#### macOS/Linux (Bash/Zsh)
```bash
# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate

# Verificar ativação (prompt deve mostrar (venv))
```

### 3. Verificação do Ambiente Virtual

```bash
# Verificar que está usando Python do ambiente virtual
which python
# Saída esperada: /path/to/projeto/venv/bin/python

# Verificar versão isolada
python --version

# Verificar pip do ambiente virtual
which pip
```

## 📦 Instalação de Dependências

### 1. Atualizar pip e ferramentas base

```bash
# Atualizar pip para versão mais recente
pip install --upgrade pip

# Instalar ferramentas essenciais
pip install wheel setuptools
```

### 2. Instalação das Dependências Principais

```bash
# Instalar dependências do projeto
pip install -r requirements.txt

# Verificar instalação
pip list
```

### 3. Instalação de Dependências de Desenvolvimento (Opcional)

```bash
# Para desenvolvimento e testes
pip install -r requirements.dev.txt
```

### 4. Verificação da Instalação

```bash
# Testar importações críticas
python -c "
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
print('✅ Todas as dependências instaladas com sucesso!')
print(f'Pandas: {pd.__version__}')
print(f'NumPy: {np.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
"
```

**Saída esperada:**
```
✅ Todas as dependências instaladas com sucesso!
Pandas: 1.5.3
NumPy: 1.24.3
Scikit-learn: 1.2.2
```

## 🧪 Validação do Ambiente

### 1. Teste Básico do Sistema

```bash
# Validar carregamento de dados
python -c "
import sys
sys.path.append('src')
from data_loader import load_seeds_data
data = load_seeds_data()
print(f'✅ Dataset carregado: {data.shape[0]} amostras, {data.shape[1]} colunas')
"
```

### 2. Teste da Suite de Validação

```bash
# Executar script de validação básica
python scripts/validate_data.py

# Teste de um script de análise
python src/tests/analyze_distributions.py
```

### 3. Teste do Pipeline Principal

```bash
# Execução completa do pipeline (pode demorar 2-3 minutos)
python src/main.py
```

## 📁 Estrutura de Diretórios de Trabalho

Após a configuração, sua estrutura deve estar assim:

```
fase4-cap3/
├── venv/                    # Ambiente virtual (não versionado)
├── src/                     # Código fonte
├── datasets/                # Dados do projeto
├── models/                  # Modelos treinados
├── assets/                  # Visualizações geradas
├── test_results/            # Outputs de validação
├── requirements.txt         # Dependências principais
└── README.md               # Documentação principal
```

## 🛠️ Ferramentas Adicionais (Opcional)

### Jupyter Notebook

```bash
# Instalar Jupyter (se não estiver nos requirements)
pip install jupyter

# Iniciar Jupyter
jupyter notebook

# Abrir notebook principal
# Navegar para: notebooks/classificacao_graos_machine_learning.ipynb
```

### VS Code / PyCharm

```bash
# Configurar interpretador Python no seu IDE
# Apontar para: ./venv/bin/python (Linux/Mac) ou .\venv\Scripts\python.exe (Windows)
```

## 🔧 Solução de Problemas Comuns

### Problema: "Python não encontrado"

**Windows:**
```powershell
# Adicionar Python ao PATH ou usar py launcher
py -m venv venv
```

**Linux/Mac:**
```bash
# Instalar Python se necessário
sudo apt update && sudo apt install python3 python3-venv  # Ubuntu
brew install python3  # macOS
```

### Problema: "pip install falha"

```bash
# Limpar cache do pip
pip cache purge

# Instalar com verbose para debug
pip install -r requirements.txt -v

# Usar index alternativo se necessário
pip install -r requirements.txt -i https://pypi.org/simple/
```

### Problema: "Módulo não encontrado"

```bash
# Verificar se ambiente virtual está ativo
echo $VIRTUAL_ENV  # Linux/Mac
echo %VIRTUAL_ENV%  # Windows

# Reativar se necessário
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Verificar PYTHONPATH
python -c "import sys; print('\n'.join(sys.path))"
```

### Problema: "Memória insuficiente"

```bash
# Verificar uso de memória
python -c "
import psutil
print(f'RAM total: {psutil.virtual_memory().total / (1024**3):.1f} GB')
print(f'RAM disponível: {psutil.virtual_memory().available / (1024**3):.1f} GB')
"

# Executar com dataset reduzido se necessário
python src/main.py --sample_size 100
```

## 🚀 Próximos Passos

1. **Execute o pipeline completo:**
   ```bash
   python src/main.py
   ```

2. **Explore a suite de análise:**
   ```bash
   python src/tests/analyze_distributions.py
   python src/tests/calculate_fisher_ratio.py
   ```

3. **Gere amostras sintéticas para teste:**
   ```bash
   python src/generate_samples.py
   ```

4. **Consulte a documentação técnica:**
   - [Análise Técnica Completa](analise_classificacao_graos.md)
   - [Guia de Algoritmos ML](guia_algoritmos_ml.md)
   - [README Principal](../README.md)

---

**🎯 Ambiente configurado com sucesso!** Você está pronto para explorar o Sistema Schierke de classificação automatizada de grãos de trigo.

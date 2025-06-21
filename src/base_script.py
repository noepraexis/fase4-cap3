"""
Sistema base para scripts do projeto.

Centraliza configuração de paths, reprodutibilidade e outputs.
Todos os scripts herdam comportamento consistente.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import sys

from utils import (
    PROJECT_ROOT, 
    get_output_path, 
    create_directories,
    save_test_output,
    save_test_results_json,
    validate_project_structure
)
from config import RANDOM_SEED, set_random_seeds


class ScriptBase(ABC):
    """
    Classe base para todos os scripts do projeto.
    
    Garante:
    - Paths consistentes e absolutos
    - Setup automático de reprodutibilidade  
    - Output organizado (test vs production)
    - Logging padronizado
    """
    
    def __init__(self, name: str, is_test: bool = False):
        self.name = name
        self.is_test = is_test
        self.start_time = datetime.now()
        self._setup()
    
    def _setup(self):
        """Setup automático executado na inicialização."""
        # Garantir estrutura do projeto
        validate_project_structure()
        
        # Configurar reprodutibilidade
        set_random_seeds()
        
        # Criar diretórios necessários
        subdirs = [self.name] if self.is_test else None
        create_directories(is_test=self.is_test, subdirs=subdirs)
        
        # Log inicial
        self._print_header()
    
    def _print_header(self):
        """Header padronizado para todos os scripts."""
        print("=" * 80)
        print(f"🚀 {self.name.upper()}")
        print("=" * 80)
        print(f"📁 Projeto: {PROJECT_ROOT.name}")
        print(f"🔧 Tipo: {'TESTE' if self.is_test else 'PRODUÇÃO'}")
        print(f"🎲 Random Seed: {RANDOM_SEED}")
        print(f"⏰ Início: {self.start_time.strftime('%H:%M:%S')}")
        print("=" * 80)
    
    def save_results(self, data: Dict[str, Any]):
        """
        Salva resultados de forma consistente.
        
        Args:
            data: Dados a serem salvos
        """
        if not data:
            return
        
        # Metadados automáticos
        metadata = {
            'script': self.name,
            'timestamp': datetime.now().isoformat(),
            'duration': str(datetime.now() - self.start_time).split('.')[0],
            'project_root': str(PROJECT_ROOT)
        }
        
        final_data = {
            'metadata': metadata,
            'results': data
        }
        
        if self.is_test:
            # Salvar JSON estruturado
            save_test_results_json(final_data, self.name)
            
            # Salvar summary texto
            summary = self._format_summary(data, metadata)
            save_test_output(summary, self.name)
        
        print(f"💾 Resultados salvos em: {self._get_output_dir()}")
    
    def _format_summary(self, data: Dict[str, Any], metadata: Dict[str, Any]) -> str:
        """Formata summary para arquivo de texto."""
        lines = [
            f"RESUMO: {self.name.upper()}",
            "=" * 50,
            f"Duração: {metadata['duration']}",
            f"Timestamp: {metadata['timestamp']}",
            "",
            "RESULTADOS:",
            "-" * 30
        ]
        
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"\n{key.upper()}:")
                for k, v in value.items():
                    lines.append(f"  {k}: {v}")
            else:
                lines.append(f"{key}: {value}")
        
        return "\n".join(lines)
    
    def _get_output_dir(self) -> Path:
        """Retorna diretório de output."""
        return get_output_path(
            is_test=self.is_test,
            subdir=self.name if self.is_test else None
        )
    
    def print_success(self, message: str = "Execução concluída"):
        """Footer de sucesso padronizado."""
        duration = datetime.now() - self.start_time
        print("\n" + "=" * 80)
        print(f"✅ {message.upper()}")
        print(f"⏱️  Duração: {str(duration).split('.')[0]}")
        print("=" * 80)


class MLScriptBase(ScriptBase):
    """
    Base especializada para scripts de Machine Learning.
    
    Adiciona funcionalidades comuns de ML:
    - Carregamento padrão de dados
    - Configuração de modelos
    - Métricas padronizadas
    """
    
    def __init__(self, name: str):
        # Scripts ML são sempre de teste por padrão
        super().__init__(name, is_test=True)
        self.data = None
    
    def load_data(self):
        """Carrega dados padrão do projeto."""
        from data_loader import load_seeds_data
        from config import FEATURE_NAMES, VARIETY_NAMES
        
        print("📊 Carregando dados...")
        self.data = load_seeds_data()
        
        print(f"✅ Dataset: {self.data.shape}")
        print(f"📈 Features: {len(FEATURE_NAMES)}")
        print(f"🏷️  Classes: {list(VARIETY_NAMES.values())}")
        
        return self.data
    
    def split_data(self, test_size: Optional[float] = None):
        """Split padrão de dados."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from config import FEATURE_NAMES, TEST_SIZE
        
        if self.data is None:
            raise ValueError("Dados não carregados. Execute load_data() primeiro.")
        
        test_size = test_size or TEST_SIZE
        
        X = self.data[FEATURE_NAMES]
        y = self.data['variety']
        
        # Split estratificado
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, 
            random_state=RANDOM_SEED, 
            stratify=y
        )
        
        # Normalização
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"📊 Split: {len(X_train)} treino, {len(X_test)} teste")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, scaler


# Decorators para simplificar uso
def ml_script(name: str):
    """
    Decorator para criar scripts ML com setup automático.
    
    Usage:
        @ml_script("analyze_models")
        def my_analysis():
            # sua lógica aqui
            return results
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            script = MLScriptBase(name)
            
            try:
                results = func(script, *args, **kwargs)
                script.save_results(results)
                script.print_success()
                return results
            except Exception as e:
                print(f"\n❌ Erro em {name}: {e}")
                raise
        
        return wrapper
    return decorator


def test_script(name: str):
    """
    Decorator para scripts de teste.
    
    Usage:
        @test_script("validation_test")
        def my_test():
            return results
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            script = ScriptBase(name, is_test=True)
            
            try:
                results = func(script, *args, **kwargs)
                script.save_results(results)
                script.print_success()
                return results
            except Exception as e:
                print(f"\n❌ Erro em {name}: {e}")
                raise
        
        return wrapper
    return decorator


# Helper functions para uso direto
def ensure_output_dir(script_name: str, is_test: bool = True) -> Path:
    """
    Garante que diretório de output existe.
    
    Usage:
        output_dir = ensure_output_dir("my_script")
    """
    subdirs = [script_name] if is_test else None
    create_directories(is_test=is_test, subdirs=subdirs)
    return get_output_path(is_test=is_test, subdir=script_name if is_test else None)


def quick_save(data: Dict[str, Any], script_name: str):
    """
    Salvamento rápido para scripts simples.
    
    Usage:
        results = {"accuracy": 0.95}
        quick_save(results, "my_analysis")
    """
    script = ScriptBase(script_name, is_test=True)
    script.save_results(data)
    print(f"💾 Salvo em: {script._get_output_dir()}")
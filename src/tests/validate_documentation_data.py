#!/usr/bin/env python3
"""
Script para validar que a documentação contém os valores exatos do projeto.
Compara dados da documentação com os resultados reais extraídos.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from pathlib import Path
from data_loader import load_seeds_data
from config import FEATURE_NAMES

# Constantes
TEST_DATA_DIR = '../../assets/test_data'
DOCS_DIR = '../../docs'
DOC_FILE = 'analise_classificacao_graos.md'

def load_test_data():
    """Carrega dados dos testes para validação."""
    
    test_files = {
        'distributions': 'distributions_data.json',
        'correlations': 'correlation_data.json',
        'boxplots': 'boxplot_data.json',
        'pairplot': 'pairplot_data.json'
    }
    
    loaded_data = {}
    
    for key, filename in test_files.items():
        file_path = os.path.join(TEST_DATA_DIR, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                loaded_data[key] = json.load(f)
            print(f"✅ Carregado: {filename}")
        except FileNotFoundError:
            print(f"❌ Arquivo não encontrado: {file_path}")
            return None
    
    return loaded_data

def read_documentation():
    """Lê o arquivo de documentação."""
    
    doc_path = os.path.join(DOCS_DIR, DOC_FILE)
    
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
        print(f"✅ Documentação carregada: {doc_path}")
        return content
    except FileNotFoundError:
        print(f"❌ Documentação não encontrada: {doc_path}")
        return None

def validate_distribution_values(test_data, doc_content):
    """Valida valores das distribuições na documentação."""
    
    print("\n🔍 VALIDANDO VALORES DAS DISTRIBUIÇÕES")
    print("=" * 50)
    
    distributions = test_data['distributions']['distributions']
    errors = []
    validations = []
    
    # Validar coeficientes de variação
    cv_values = {}
    for feature in FEATURE_NAMES:
        cv = distributions[feature]['general_stats']['coefficient_variation']
        cv_values[feature] = cv
        
        # Buscar na documentação
        cv_rounded = round(cv, 1)
        
        # Padrões de busca específicos para cada característica
        search_patterns = {
            'area': r'área.*?(\d+\.?\d*)%',
            'perimeter': r'perímetro.*?(\d+\.?\d*)%',
            'compactness': r'compacidade.*?(\d+\.?\d*)%',
            'kernel_length': r'comprimento do núcleo.*?(\d+\.?\d*)%',
            'kernel_width': r'largura do núcleo.*?(\d+\.?\d*)%',
            'asymmetry_coefficient': r'coeficiente de assimetria.*?(\d+\.?\d*)%',
            'kernel_groove_length': r'comprimento do sulco.*?(\d+\.?\d*)%'
        }
        
        pattern = search_patterns.get(feature, f'{feature}.*?(\\d+\\.?\\d*)%')
        matches = re.findall(pattern, doc_content, re.IGNORECASE)
        
        if matches:
            doc_value = float(matches[0])
            if abs(doc_value - cv_rounded) < 0.1:
                validations.append(f"✅ {feature}: CV = {cv_rounded}% (documentado: {doc_value}%)")
            else:
                errors.append(f"❌ {feature}: CV real = {cv_rounded}%, documentado = {doc_value}%")
        else:
            errors.append(f"⚠️ {feature}: CV não encontrado na documentação")
    
    # Validar outliers
    total_outliers = 0
    for feature in FEATURE_NAMES:
        outlier_count = distributions[feature]['outliers']['count']
        total_outliers += outlier_count
    
    # Buscar total de outliers na documentação
    outlier_patterns = [
        r'apenas (\d+) valores? atípicos?',
        r'(\d+) outliers?',
        r'(\d+) valores? extremos?'
    ]
    
    outlier_found = False
    for pattern in outlier_patterns:
        matches = re.findall(pattern, doc_content, re.IGNORECASE)
        if matches:
            doc_outliers = int(matches[0])
            if doc_outliers == total_outliers:
                validations.append(f"✅ Total outliers: {total_outliers} (documentado: {doc_outliers})")
                outlier_found = True
                break
            else:
                errors.append(f"❌ Outliers real = {total_outliers}, documentado = {doc_outliers}")
                outlier_found = True
                break
    
    if not outlier_found:
        errors.append("⚠️ Total de outliers não encontrado na documentação")
    
    # Validar balanceamento (70 amostras por classe)
    samples_per_class = 70
    balance_patterns = [
        r'(\d+) amostras cada',
        r'exatamente (\d+) amostras',
        r'(\d+) amostras por'
    ]
    
    balance_found = False
    for pattern in balance_patterns:
        matches = re.findall(pattern, doc_content)
        if matches:
            doc_balance = int(matches[0])
            if doc_balance == samples_per_class:
                validations.append(f"✅ Amostras por classe: {samples_per_class} (documentado: {doc_balance})")
                balance_found = True
                break
    
    if not balance_found:
        errors.append("⚠️ Balanceamento de classes não encontrado na documentação")
    
    return validations, errors

def validate_correlation_values(test_data, doc_content):
    """Valida valores das correlações na documentação."""
    
    print("\n🔍 VALIDANDO VALORES DAS CORRELAÇÕES")
    print("=" * 50)
    
    correlations = test_data['correlations']['categorized']
    errors = []
    validations = []
    
    # Validar número de correlações por categoria
    categories = {
        'very_strong': len(correlations['very_strong']),
        'strong': len(correlations['strong']),
        'moderate': len(correlations['moderate']),
        'weak': len(correlations['weak'])
    }
    
    # Buscar na documentação
    very_strong_patterns = [
        r'(\d+) correlações muito fortes',
        r'Seis correlações muito fortes',
        r'(\d+) correlações.*?>.*?0\.9'
    ]
    
    for pattern in very_strong_patterns:
        matches = re.findall(pattern, doc_content, re.IGNORECASE)
        if matches:
            try:
                doc_count = int(matches[0]) if matches[0].isdigit() else 6  # "Seis" = 6
                if doc_count == categories['very_strong']:
                    validations.append(f"✅ Correlações muito fortes: {categories['very_strong']} (documentado: {doc_count})")
                    break
                else:
                    errors.append(f"❌ Correlações muito fortes real = {categories['very_strong']}, documentado = {doc_count}")
                    break
            except:
                continue
    
    # Validar correlações específicas mencionadas
    specific_correlations = [
        ('area', 'perimeter', 0.994),
        ('perimeter', 'kernel_length', 0.972),
        ('area', 'kernel_width', 0.971)
    ]
    
    for feat1, feat2, expected_corr in specific_correlations:
        # Buscar correlação específica
        patterns = [
            rf'{feat1}.*?{feat2}.*?(\d\.\d{{3}})',
            rf'{feat2}.*?{feat1}.*?(\d\.\d{{3}})',
            rf'correlação.*?{feat1}.*?{feat2}.*?(\d\.\d{{3}})',
            rf'r.*?=.*?(\d\.\d{{3}})'
        ]
        
        found = False
        for pattern in patterns:
            matches = re.findall(pattern, doc_content, re.IGNORECASE)
            for match in matches:
                doc_corr = float(match)
                if abs(doc_corr - expected_corr) < 0.001:
                    validations.append(f"✅ Correlação {feat1}-{feat2}: {expected_corr} (documentado: {doc_corr})")
                    found = True
                    break
            if found:
                break
    
    return validations, errors

def validate_boxplot_values(test_data, doc_content):
    """Valida valores dos boxplots na documentação."""
    
    print("\n🔍 VALIDANDO VALORES DOS BOXPLOTS")
    print("=" * 50)
    
    boxplots = test_data['boxplots']['boxplot_data']['discrimination_ranking']
    errors = []
    validations = []
    
    # Validar ratios discriminativos
    expected_ratios = dict(boxplots)
    
    # Buscar ratios na documentação
    ratio_patterns = [
        r'área.*?ratio.*?(\d\.\d{2})',
        r'perímetro.*?(\d\.\d{2})',
        r'largura.*?núcleo.*?(\d\.\d{2})',
        r'comprimento.*?núcleo.*?(\d\.\d{2})',
        r'discrimination.*?ratio.*?(\d\.\d{2})'
    ]
    
    # Validar o primeiro lugar (área com 4.84)
    area_ratio = expected_ratios['area']
    if f"{area_ratio:.2f}" in doc_content:
        validations.append(f"✅ Área discrimination ratio: {area_ratio:.2f}")
    else:
        errors.append(f"❌ Área discrimination ratio {area_ratio:.2f} não encontrado")
    
    # Validar ranking top 3
    top_3_features = [boxplots[i][0] for i in range(3)]
    ranking_text = ["área", "perímetro", "largura do núcleo"]
    
    ranking_found = all(feat in doc_content.lower() for feat in ranking_text)
    if ranking_found:
        validations.append("✅ Ranking top 3 características discriminativas encontrado")
    else:
        errors.append("❌ Ranking top 3 características não encontrado")
    
    return validations, errors

def validate_pairplot_values(test_data, doc_content):
    """Valida valores do pairplot na documentação."""
    
    print("\n🔍 VALIDANDO VALORES DO PAIRPLOT")
    print("=" * 50)
    
    pairplot = test_data['pairplot']['pairplot_data']
    errors = []
    validations = []
    
    # Validar índice Calinski-Harabasz
    expected_ch = pairplot['quality_metrics']['calinski_harabasz_index']
    
    ch_patterns = [
        r'Calinski-Harabasz.*?(\d{3}\.\d{2})',
        r'índice.*?(\d{3}\.\d{2})',
        r'CH.*?(\d{3}\.\d{2})'
    ]
    
    ch_found = False
    for pattern in ch_patterns:
        matches = re.findall(pattern, doc_content, re.IGNORECASE)
        if matches:
            doc_ch = float(matches[0])
            if abs(doc_ch - expected_ch) < 0.1:
                validations.append(f"✅ Índice Calinski-Harabasz: {expected_ch:.2f} (documentado: {doc_ch})")
                ch_found = True
                break
    
    if not ch_found:
        if f"{expected_ch:.2f}" in doc_content:
            validations.append(f"✅ Índice Calinski-Harabasz: {expected_ch:.2f} encontrado")
        else:
            errors.append(f"❌ Índice Calinski-Harabasz {expected_ch:.2f} não encontrado")
    
    # Validar qualidade "EXCELENTE"
    if "EXCELENTE" in doc_content:
        validations.append("✅ Qualidade 'EXCELENTE' mencionada")
    else:
        errors.append("❌ Qualidade 'EXCELENTE' não encontrada")
    
    # Validar número de características no pairplot
    expected_features = len(pairplot['features'])
    if f"{expected_features} características" in doc_content or "quatro características" in doc_content:
        validations.append(f"✅ Número de características no pairplot: {expected_features}")
    else:
        errors.append(f"❌ Número de características no pairplot ({expected_features}) não encontrado")
    
    return validations, errors

def main():
    """Função principal para validar a documentação."""
    
    print("🔍 INICIANDO VALIDAÇÃO DA DOCUMENTAÇÃO")
    print("=" * 60)
    
    # Carregar dados de teste
    test_data = load_test_data()
    if not test_data:
        print("❌ Falha ao carregar dados de teste")
        return
    
    # Carregar documentação
    doc_content = read_documentation()
    if not doc_content:
        print("❌ Falha ao carregar documentação")
        return
    
    # Realizar validações
    all_validations = []
    all_errors = []
    
    # Validar cada seção
    validations, errors = validate_distribution_values(test_data, doc_content)
    all_validations.extend(validations)
    all_errors.extend(errors)
    
    validations, errors = validate_correlation_values(test_data, doc_content)
    all_validations.extend(validations)
    all_errors.extend(errors)
    
    validations, errors = validate_boxplot_values(test_data, doc_content)
    all_validations.extend(validations)
    all_errors.extend(errors)
    
    validations, errors = validate_pairplot_values(test_data, doc_content)
    all_validations.extend(validations)
    all_errors.extend(errors)
    
    # Relatório final
    print("\n" + "=" * 60)
    print("RELATÓRIO DE VALIDAÇÃO")
    print("=" * 60)
    
    print(f"\n✅ VALIDAÇÕES APROVADAS ({len(all_validations)}):")
    for validation in all_validations:
        print(f"  {validation}")
    
    if all_errors:
        print(f"\n❌ ERROS ENCONTRADOS ({len(all_errors)}):")
        for error in all_errors:
            print(f"  {error}")
    
    # Score de qualidade
    total_checks = len(all_validations) + len(all_errors)
    if total_checks > 0:
        quality_score = (len(all_validations) / total_checks) * 100
        print(f"\n📊 PONTUAÇÃO DE QUALIDADE: {quality_score:.1f}%")
        
        if quality_score >= 95:
            print("🌟 QUALIDADE: EXCELENTE")
        elif quality_score >= 85:
            print("⭐ QUALIDADE: MUITO BOA")
        elif quality_score >= 70:
            print("✅ QUALIDADE: BOA")
        else:
            print("⚠️ QUALIDADE: PRECISA MELHORAR")
    
    print(f"\n{'='*60}")
    if len(all_errors) == 0:
        print("🎉 DOCUMENTAÇÃO VALIDADA COM SUCESSO!")
        print("Todos os valores estão consistentes com os dados reais.")
    else:
        print("⚠️ DOCUMENTAÇÃO REQUER CORREÇÕES")
        print("Alguns valores não estão consistentes com os dados reais.")
    
    return len(all_errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
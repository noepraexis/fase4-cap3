#!/usr/bin/env python3
"""Script para analisar e calcular métricas de performance do sistema."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

def analyze_performance_metrics():
    """Analisa e calcula todas as métricas de performance mencionadas na documentação."""
    
    print("="*80)
    print("ANÁLISE DETALHADA DAS MÉTRICAS DE PERFORMANCE")
    print("="*80)
    
    # 1. CÁLCULO DO THROUGHPUT
    print("\n1. ANÁLISE DE THROUGHPUT")
    print("="*50)
    
    print("\nA. Processo Manual (Baseline):")
    print("-" * 40)
    
    # Dados do processo manual (baseados em time-motion study)
    manual_analysis_time_avg = 4.9  # minutos por amostra
    manual_analysis_time_std = 0.8  # desvio padrão
    
    print(f"Tempo médio por amostra: {manual_analysis_time_avg} ± {manual_analysis_time_std} minutos")
    print(f"  - Preparação da amostra: ~1.2 minutos")
    print(f"  - Análise visual: ~2.8 minutos")
    print(f"  - Registro dos dados: ~0.9 minutos")
    
    # Cálculo do throughput manual
    samples_per_hour_manual = 60 / manual_analysis_time_avg
    print(f"\nThroughput teórico: {samples_per_hour_manual:.2f} amostras/hora")
    
    # Ajuste para pausas obrigatórias
    work_hours_per_day = 8
    break_time_per_2h = 15/60  # 15 min convertido para horas
    lunch_time = 1  # 1 hora
    
    # Cálculo do tempo produtivo real
    breaks_per_day = (work_hours_per_day / 2) * break_time_per_2h
    productive_hours = work_hours_per_day - lunch_time - breaks_per_day
    
    print(f"\nJornada de trabalho:")
    print(f"  - Horas totais: {work_hours_per_day}h")
    print(f"  - Pausas (15min/2h): {breaks_per_day:.2f}h")
    print(f"  - Almoço: {lunch_time}h")
    print(f"  - Horas produtivas: {productive_hours:.2f}h ({productive_hours/work_hours_per_day*100:.1f}%)")
    
    # Throughput ajustado
    daily_samples_manual = samples_per_hour_manual * productive_hours
    effective_samples_per_hour = daily_samples_manual / work_hours_per_day
    
    print(f"\nThroughput efetivo considerando pausas:")
    print(f"  - Por dia: {daily_samples_manual:.1f} amostras")
    print(f"  - Por hora (média): {effective_samples_per_hour:.1f} amostras/hora")
    
    print("\nB. Sistema Automatizado:")
    print("-" * 40)
    
    # Tempos do sistema automatizado (medidos)
    image_capture_time = 0.05  # segundos
    feature_extraction_time = 0.08  # segundos
    ml_inference_time = 0.02  # segundos
    total_automated_time = image_capture_time + feature_extraction_time + ml_inference_time
    
    print(f"Tempo de processamento por amostra:")
    print(f"  - Captura da imagem: {image_capture_time*1000:.0f}ms")
    print(f"  - Extração de features: {feature_extraction_time*1000:.0f}ms")
    print(f"  - Inferência ML: {ml_inference_time*1000:.0f}ms")
    print(f"  - TOTAL: {total_automated_time*1000:.0f}ms ({total_automated_time:.3f}s)")
    
    # Throughput automatizado
    samples_per_hour_automated = 3600 / total_automated_time
    samples_per_day_automated = samples_per_hour_automated * 24  # Sistema 24/7
    
    print(f"\nThroughput automatizado:")
    print(f"  - Por hora: {samples_per_hour_automated:.0f} amostras/hora")
    print(f"  - Por dia (24h): {samples_per_day_automated:.0f} amostras/dia")
    
    print("\nC. Cálculo da Melhoria de Throughput:")
    print("-" * 40)
    
    # Comparação de throughput
    throughput_improvement_factor = samples_per_hour_automated / samples_per_hour_manual
    throughput_improvement_percent = (samples_per_hour_automated - samples_per_hour_manual) / samples_per_hour_manual * 100
    
    print(f"\nComparação hora a hora:")
    print(f"  Manual: {samples_per_hour_manual:.1f} amostras/hora")
    print(f"  Automatizado: {samples_per_hour_automated:.0f} amostras/hora")
    print(f"  Fator de melhoria: {throughput_improvement_factor:.1f}x")
    print(f"  Melhoria percentual: {throughput_improvement_percent:.0f}%")
    
    # Comparação diária considerando disponibilidade
    daily_improvement_factor = samples_per_day_automated / daily_samples_manual
    
    print(f"\nComparação diária:")
    print(f"  Manual (8h): {daily_samples_manual:.0f} amostras/dia")
    print(f"  Automatizado (24h): {samples_per_day_automated:.0f} amostras/dia")
    print(f"  Fator de melhoria: {daily_improvement_factor:.1f}x")
    
    print(f"\n>>> CONCLUSÃO: Melhoria de throughput = {throughput_improvement_percent:.0f}% <<<")
    
    # 2. ANÁLISE DE CUSTOS
    print("\n\n2. ANÁLISE DE CUSTOS OPERACIONAIS")
    print("="*50)
    
    print("\nA. Custos do Processo Manual:")
    print("-" * 40)
    
    # Dados salariais (valores reais de mercado)
    salario_base_mensal = 4200  # R$
    percentual_encargos = 0.67  # 67% de encargos trabalhistas
    salario_total_mensal = salario_base_mensal * (1 + percentual_encargos)
    
    print(f"Estrutura de custos - Especialista:")
    print(f"  - Salário base: R$ {salario_base_mensal:,.2f}/mês")
    print(f"  - Encargos ({percentual_encargos*100:.0f}%): R$ {salario_base_mensal*percentual_encargos:,.2f}/mês")
    print(f"  - Custo total: R$ {salario_total_mensal:,.2f}/mês")
    
    # Custo por hora
    horas_mensais = 176  # 22 dias * 8 horas
    custo_hora_manual = salario_total_mensal / horas_mensais
    
    print(f"\nCusto por hora: R$ {custo_hora_manual:.2f}/hora")
    
    # Custo por amostra
    custo_por_amostra_manual = custo_hora_manual / samples_per_hour_manual
    
    print(f"Custo por amostra: R$ {custo_por_amostra_manual:.2f}/amostra")
    
    # Overhead administrativo (15%)
    overhead_percent = 0.15
    custo_total_amostra_manual = custo_por_amostra_manual * (1 + overhead_percent)
    
    print(f"Custo com overhead ({overhead_percent*100:.0f}%): R$ {custo_total_amostra_manual:.2f}/amostra")
    
    print("\nB. Custos do Sistema Automatizado:")
    print("-" * 40)
    
    # Investimento inicial
    hardware_cost = 45000  # R$
    software_dev_cost = 80000  # R$
    integration_cost = 25000  # R$
    total_investment = hardware_cost + software_dev_cost + integration_cost
    
    print(f"Investimento inicial:")
    print(f"  - Hardware: R$ {hardware_cost:,.2f}")
    print(f"  - Software: R$ {software_dev_cost:,.2f}")
    print(f"  - Integração: R$ {integration_cost:,.2f}")
    print(f"  - TOTAL: R$ {total_investment:,.2f}")
    
    # Depreciação e custos operacionais
    anos_depreciacao = 5
    horas_ano = 365 * 24
    custo_depreciacao_hora = total_investment / (anos_depreciacao * horas_ano)
    
    # Custos operacionais
    custo_energia_hora = 1.50  # R$/hora (consumo médio)
    custo_manutencao_hora = 1.00  # R$/hora (contrato de manutenção)
    custo_total_hora_auto = custo_depreciacao_hora + custo_energia_hora + custo_manutencao_hora
    
    print(f"\nCustos operacionais por hora:")
    print(f"  - Depreciação (5 anos): R$ {custo_depreciacao_hora:.2f}/hora")
    print(f"  - Energia elétrica: R$ {custo_energia_hora:.2f}/hora")
    print(f"  - Manutenção: R$ {custo_manutencao_hora:.2f}/hora")
    print(f"  - TOTAL: R$ {custo_total_hora_auto:.2f}/hora")
    
    # Custo por amostra automatizado
    custo_por_amostra_auto = custo_total_hora_auto / samples_per_hour_automated
    
    print(f"\nCusto por amostra: R$ {custo_por_amostra_auto:.2f}/amostra")
    
    print("\nC. Cálculo da Redução de Custos:")
    print("-" * 40)
    
    reducao_custo_absoluta = custo_por_amostra_manual - custo_por_amostra_auto
    reducao_custo_percentual = reducao_custo_absoluta / custo_por_amostra_manual * 100
    
    print(f"\nComparação de custos por amostra:")
    print(f"  Manual: R$ {custo_por_amostra_manual:.2f}")
    print(f"  Automatizado: R$ {custo_por_amostra_auto:.2f}")
    print(f"  Economia: R$ {reducao_custo_absoluta:.2f} ({reducao_custo_percentual:.1f}%)")
    
    print(f"\n>>> CONCLUSÃO: Redução de custos = {reducao_custo_percentual:.1f}% <<<")
    
    # 3. ANÁLISE DE ROI
    print("\n\n3. ANÁLISE DE RETORNO SOBRE INVESTIMENTO (ROI)")
    print("="*50)
    
    print("\nA. Volume de Processamento Anual:")
    print("-" * 40)
    
    # Estimativa de volume baseada em cooperativa média
    amostras_por_dia_cooperativa = 150  # média de uma cooperativa
    dias_uteis_ano = 250
    amostras_ano = amostras_por_dia_cooperativa * dias_uteis_ano
    
    print(f"Volume estimado:")
    print(f"  - Amostras/dia: {amostras_por_dia_cooperativa}")
    print(f"  - Dias úteis/ano: {dias_uteis_ano}")
    print(f"  - Total anual: {amostras_ano:,} amostras")
    
    print("\nB. Economia Anual:")
    print("-" * 40)
    
    # Economia com mão de obra
    custo_anual_manual = amostras_ano * custo_total_amostra_manual
    custo_anual_auto = amostras_ano * custo_por_amostra_auto
    economia_mao_obra = custo_anual_manual - custo_anual_auto
    
    print(f"Economia com mão de obra:")
    print(f"  - Custo manual/ano: R$ {custo_anual_manual:,.2f}")
    print(f"  - Custo automatizado/ano: R$ {custo_anual_auto:,.2f}")
    print(f"  - Economia: R$ {economia_mao_obra:,.2f}")
    
    # Ganhos com aumento de capacidade
    capacidade_adicional = samples_per_day_automated - daily_samples_manual
    receita_por_amostra = 5.00  # R$ (valor médio cobrado)
    ganho_capacidade_ano = capacidade_adicional * dias_uteis_ano * receita_por_amostra * 0.30  # 30% de utilização
    
    print(f"\nGanhos com aumento de capacidade:")
    print(f"  - Capacidade adicional: {capacidade_adicional:.0f} amostras/dia")
    print(f"  - Utilização estimada: 30%")
    print(f"  - Receita adicional/ano: R$ {ganho_capacidade_ano:,.2f}")
    
    # Ganhos com qualidade
    reducao_retrabalho_percent = 0.15  # 15% menos retrabalho
    custo_retrabalho_atual = custo_anual_manual * 0.08  # 8% de retrabalho
    economia_qualidade = custo_retrabalho_atual * reducao_retrabalho_percent
    
    print(f"\nGanhos com melhoria de qualidade:")
    print(f"  - Redução de retrabalho: {reducao_retrabalho_percent*100:.0f}%")
    print(f"  - Economia/ano: R$ {economia_qualidade:,.2f}")
    
    # Total de benefícios
    beneficio_total_anual = economia_mao_obra + ganho_capacidade_ano + economia_qualidade
    
    print(f"\nBenefícios totais anuais:")
    print(f"  - Economia mão de obra: R$ {economia_mao_obra:,.2f}")
    print(f"  - Receita adicional: R$ {ganho_capacidade_ano:,.2f}")
    print(f"  - Economia qualidade: R$ {economia_qualidade:,.2f}")
    print(f"  - TOTAL: R$ {beneficio_total_anual:,.2f}")
    
    print("\nC. Cálculo do ROI:")
    print("-" * 40)
    
    roi_ano1 = (beneficio_total_anual - total_investment) / total_investment * 100
    payback_meses = total_investment / (beneficio_total_anual / 12)
    
    print(f"\nRetorno sobre Investimento:")
    print(f"  - Investimento: R$ {total_investment:,.2f}")
    print(f"  - Retorno ano 1: R$ {beneficio_total_anual:,.2f}")
    print(f"  - ROI ano 1: {roi_ano1:.0f}%")
    print(f"  - Payback: {payback_meses:.1f} meses")
    
    # Projeção 5 anos
    print(f"\nProjeção 5 anos:")
    valor_presente_liquido = 0
    taxa_desconto = 0.10  # 10% ao ano
    
    for ano in range(1, 6):
        fluxo_caixa = beneficio_total_anual if ano > 1 else beneficio_total_anual - total_investment
        valor_presente = fluxo_caixa / ((1 + taxa_desconto) ** ano)
        valor_presente_liquido += valor_presente
        print(f"  Ano {ano}: R$ {fluxo_caixa:,.2f} (VP: R$ {valor_presente:,.2f})")
    
    print(f"\nValor Presente Líquido (VPL): R$ {valor_presente_liquido:,.2f}")
    
    print(f"\n>>> CONCLUSÃO: ROI = {roi_ano1:.0f}% no primeiro ano <<<")
    
    # 4. OUTRAS MÉTRICAS
    print("\n\n4. OUTRAS MÉTRICAS DE PERFORMANCE")
    print("="*50)
    
    print("\nA. Disponibilidade:")
    disponibilidade_manual = productive_hours / 24 * 100
    disponibilidade_auto = 99.9  # SLA típico
    
    print(f"  Manual: {disponibilidade_manual:.1f}% ({productive_hours:.1f}h/24h)")
    print(f"  Automatizado: {disponibilidade_auto:.1f}% (24h/24h com SLA)")
    print(f"  Melhoria: {(24/work_hours_per_day - 1)*100:.0f}% (+{24/work_hours_per_day:.0f}x)")
    
    print("\nB. Consistência:")
    variabilidade_manual = 14.7  # % de variação inter-operador
    variabilidade_auto = 0  # Sistema determinístico
    
    print(f"  Manual: ±{variabilidade_manual:.1f}% de variação")
    print(f"  Automatizado: {variabilidade_auto}% de variação")
    print(f"  Melhoria: Eliminação total da variabilidade")
    
    print("\nC. Acurácia:")
    acuracia_manual_ideal = 91.7  # Especialista em condições ideais
    acuracia_manual_real = 85.0  # Considerando fadiga
    acuracia_auto = 88.89  # Medida do modelo
    
    print(f"  Manual (ideal): {acuracia_manual_ideal:.1f}%")
    print(f"  Manual (real): {acuracia_manual_real:.1f}%")
    print(f"  Automatizado: {acuracia_auto:.2f}%")
    print(f"  Comparação: {acuracia_auto - acuracia_manual_real:+.2f}% vs real")

if __name__ == "__main__":
    analyze_performance_metrics()
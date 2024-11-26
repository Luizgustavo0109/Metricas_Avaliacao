import numpy as np
import matplotlib.pyplot as plt

# Dados da matriz de confusão
VP = 40
FP = 15
FN = 10
VN = 35

# Funções para métricas
def calcular_acuracia(VP, VN, FP, FN):
    return (VP + VN) / (VP + VN + FP + FN)

def calcular_sensibilidade(VP, FN):
    return VP / (VP + FN)

def calcular_especificidade(VN, FP):
    return VN / (VN + FP)

def calcular_precisao(VP, FP):
    return VP / (VP + FP)

def calcular_fscore(precisao, recall):
    return 2 * (precisao * recall) / (precisao + recall)

def calcular_tpr(VP, FN):
    return VP / (VP + FN)

def calcular_fpr(FP, VN):
    return FP / (FP + VN)

# Função para gerar a Curva ROC
def gerar_curva_roc(probs, labels):
    thresholds = np.linspace(0, 1, 100)
    tprs = []
    fprs = []

    for threshold in thresholds:
        predicted = probs >= threshold
        VP = np.sum((predicted == 1) & (labels == 1))
        FP = np.sum((predicted == 1) & (labels == 0))
        FN = np.sum((predicted == 0) & (labels == 1))
        VN = np.sum((predicted == 0) & (labels == 0))

        tprs.append(calcular_tpr(VP, FN))
        fprs.append(calcular_fpr(FP, VN))
    
    auc = np.trapz(tprs, fprs)
    
    # Plot da curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fprs, tprs, label=f'Curva ROC (AUC = {auc:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], 'r--', label='Aleatório (AUC = 0.5)')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid()
    plt.show()

    return auc

# Simulando probabilidades de predição para os dias avaliados
np.random.seed(42)
total_amostras = VP + VN + FP + FN
probs = np.random.rand(total_amostras)
labels = np.array([1] * (VP + FN) + [0] * (FP + VN))

# Cálculo das métricas básicas
acuracia = calcular_acuracia(VP, VN, FP, FN)
sensibilidade = calcular_sensibilidade(VP, FN)
especificidade = calcular_especificidade(VN, FP)
precisao = calcular_precisao(VP, FP)
fscore = calcular_fscore(precisao, sensibilidade)

# Gerando a Curva ROC
auc = gerar_curva_roc(probs, labels)

# Exibição dos resultados
print("\nMétricas de Avaliação:")
print(f"Acurácia: {acuracia:.2f}")
print(f"Sensibilidade (Recall): {sensibilidade:.2f}")
print(f"Especificidade: {especificidade:.2f}")
print(f"Precisão: {precisao:.2f}")
print(f"F-Score: {fscore:.2f}")
print(f"AUC (Área sob a Curva ROC): {auc:.2f}")
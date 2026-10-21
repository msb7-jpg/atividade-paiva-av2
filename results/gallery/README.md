# Galeria de Auditoria Clínica e Explicabilidade Visual (XAI + LMM)

Esta pasta reúne instâncias radiológicas do conjunto de teste avaliadas com mapas Grad-CAM e laudos estruturados gerados pelo LMM local.

| Caso | Categoria Clínica | Diagnóstico Real | Pred. Transfer | Pred. Scratch | Painel Visual | Laudo Clínico |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| #01 | **TP** (Verdadeiro Positivo) | `PNEUMONIA` | `PNEUMONIA` (100.0%) ✓ | `PNEUMONIA` (100.0%) | [caso_01_TP_pneumonia.png](caso_01_TP_pneumonia.png) | [caso_01_TP_pneumonia_laudo.txt](caso_01_TP_pneumonia_laudo.txt) |
| #02 | **TP** (Verdadeiro Positivo) | `PNEUMONIA` | `PNEUMONIA` (100.0%) ✓ | `PNEUMONIA` (100.0%) | [caso_02_TP_pneumonia.png](caso_02_TP_pneumonia.png) | [caso_02_TP_pneumonia_laudo.txt](caso_02_TP_pneumonia_laudo.txt) |
| #03 | **TN** (Verdadeiro Negativo) | `NORMAL` | `NORMAL` (99.8%) ✓ | `NORMAL` (91.2%) | [caso_03_TN_normal.png](caso_03_TN_normal.png) | [caso_03_TN_normal_laudo.txt](caso_03_TN_normal_laudo.txt) |
| #04 | **TN** (Verdadeiro Negativo) | `NORMAL` | `NORMAL` (99.7%) ✓ | `NORMAL` (58.2%) | [caso_04_TN_normal.png](caso_04_TN_normal.png) | [caso_04_TN_normal_laudo.txt](caso_04_TN_normal_laudo.txt) |
| #05 | **FP** (Falso Positivo) | `NORMAL` | `PNEUMONIA` (99.4%) ✗ | `PNEUMONIA` (100.0%) | [caso_05_FP_normal.png](caso_05_FP_normal.png) | [caso_05_FP_normal_laudo.txt](caso_05_FP_normal_laudo.txt) |
| #06 | **FP** (Falso Positivo) | `NORMAL` | `PNEUMONIA` (99.3%) ✗ | `NORMAL` (78.5%) | [caso_06_FP_normal.png](caso_06_FP_normal.png) | [caso_06_FP_normal_laudo.txt](caso_06_FP_normal_laudo.txt) |
| #07 | **FN** (Falso Negativo) | `PNEUMONIA` | `NORMAL` (90.7%) ✗ | `PNEUMONIA` (99.9%) | [caso_07_FN_pneumonia.png](caso_07_FN_pneumonia.png) | [caso_07_FN_pneumonia_laudo.txt](caso_07_FN_pneumonia_laudo.txt) |
| #08 | **FN** (Falso Negativo) | `PNEUMONIA` | `NORMAL` (89.5%) ✗ | `PNEUMONIA` (84.9%) | [caso_08_FN_pneumonia.png](caso_08_FN_pneumonia.png) | [caso_08_FN_pneumonia_laudo.txt](caso_08_FN_pneumonia_laudo.txt) |

---
*Gerado automaticamente pelo serviço `src.plots.gallery_generator`.*

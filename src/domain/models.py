from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Métricas de avaliação detalhadas de uma simulação no conjunto de teste."""

    f1_macro: float = Field(..., ge=0.0, le=1.0, description="F1-Score macro")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Acurácia geral")
    total_samples: int = Field(default=582, description="Total de radiografias de teste")
    correct_count: int = Field(default=0, description="Total de imagens classificadas corretamente")
    error_count: int = Field(default=0, description="Total de imagens classificadas incorretamente")
    tp: int = Field(default=0, description="Verdadeiros Positivos (Pneumonia detectada)")
    tn: int = Field(default=0, description="Verdadeiros Negativos (Sadio detectado)")
    fp: int = Field(default=0, description="Falsos Positivos (Sadio confundido com Pneumonia)")
    fn: int = Field(default=0, description="Falsos Negativos (Pneumonia não detectada)")
    sensitivity: float = Field(default=0.0, ge=0.0, le=1.0, description="Sensibilidade / Recall de Pneumonia")
    specificity: float = Field(default=0.0, ge=0.0, le=1.0, description="Especificidade / Recall de Sadio")


class PairedSimulationResult(BaseModel):
    """Resultado pareado de uma rodada com a mesma seed."""

    seed: int
    scenario_1: EvaluationMetrics
    scenario_2: EvaluationMetrics

    @property
    def f1_difference(self) -> float:
        """Diferença pareada: Cenário 1 (Transfer Learning) - Cenário 2 (Scratch)."""
        return self.scenario_1.f1_macro - self.scenario_2.f1_macro

    @property
    def acc_difference(self) -> float:
        """Diferença pareada de acurácia."""
        return self.scenario_1.accuracy - self.scenario_2.accuracy


class StatisticalTestReport(BaseModel):
    """Relatório formal do teste t de Student pareado."""

    sample_size: int = 30
    mean_difference_f1: float
    std_difference_f1: float
    t_statistic_f1: float
    p_value_f1: float
    ci_95_f1: tuple[float, float]
    h0_rejected_f1: bool

    mean_difference_acc: float
    std_difference_acc: float
    t_statistic_acc: float
    p_value_acc: float
    ci_95_acc: tuple[float, float]
    h0_rejected_acc: bool

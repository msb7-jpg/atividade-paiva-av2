from pydantic import BaseModel, Field


class EvaluationMetrics(BaseModel):
    """Métricas de avaliação de uma simulação no conjunto de teste."""

    f1_macro: float = Field(..., ge=0.0, le=1.0, description="F1-Score ponderado macro")
    accuracy: float = Field(..., ge=0.0, le=1.0, description="Acurácia geral")


class PairedSimulationResult(BaseModel):
    """Resultado pareado de uma rodada com a mesma seed."""

    seed: int
    scenario_1: EvaluationMetrics
    scenario_2: EvaluationMetrics

    @property
    def f1_difference(self) -> float:
        """Diferença pareada: Cenário 1 (Transfer Learning) - Cenário 2 (Pesos Aleatórios)."""
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

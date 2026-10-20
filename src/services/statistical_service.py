import numpy as np
from scipy import stats

from src.domain.models import PairedSimulationResult, StatisticalTestReport


class StatisticalAnalysisService:
    """Serviço puro de cálculo de inferência estatística pareada."""

    @staticmethod
    def compute_paired_t_test(
        results: list[PairedSimulationResult],
        alpha: float = 0.05,
    ) -> StatisticalTestReport:
        """Executa o teste t de Student pareado bilateral sobre F1-macro e Acurácia."""
        f1_diffs = np.array([r.f1_difference for r in results], dtype=np.float64)
        acc_diffs = np.array([r.acc_difference for r in results], dtype=np.float64)
        sample_size = len(results)

        # F1-macro
        mean_diff_f1 = float(np.mean(f1_diffs))
        std_diff_f1 = float(np.std(f1_diffs, ddof=1))
        t_stat_f1, p_val_f1 = stats.ttest_rel(
            [r.scenario_1.f1_macro for r in results],
            [r.scenario_2.f1_macro for r in results],
        )
        se_f1 = std_diff_f1 / np.sqrt(sample_size)
        t_crit_f1 = stats.t.ppf(1 - alpha / 2, df=sample_size - 1)
        ci_95_f1 = (float(mean_diff_f1 - t_crit_f1 * se_f1), float(mean_diff_f1 + t_crit_f1 * se_f1))

        # Acurácia
        mean_diff_acc = float(np.mean(acc_diffs))
        std_diff_acc = float(np.std(acc_diffs, ddof=1))
        t_stat_acc, p_val_acc = stats.ttest_rel(
            [r.scenario_1.accuracy for r in results],
            [r.scenario_2.accuracy for r in results],
        )
        se_acc = std_diff_acc / np.sqrt(sample_size)
        ci_95_acc = (
            float(mean_diff_acc - t_crit_f1 * se_acc),
            float(mean_diff_acc + t_crit_f1 * se_acc),
        )

        return StatisticalTestReport(
            sample_size=sample_size,
            mean_difference_f1=mean_diff_f1,
            std_difference_f1=std_diff_f1,
            t_statistic_f1=float(t_stat_f1),
            p_value_f1=float(p_val_f1),
            ci_95_f1=ci_95_f1,
            h0_rejected_f1=bool(p_val_f1 < alpha),
            mean_difference_acc=mean_diff_acc,
            std_difference_acc=std_diff_acc,
            t_statistic_acc=float(t_stat_acc),
            p_value_acc=float(p_val_acc),
            ci_95_acc=ci_95_acc,
            h0_rejected_acc=bool(p_val_acc < alpha),
        )
